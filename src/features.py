"""Extraction des caractéristiques (features) par fenêtre.

Trois familles sont calculées pour chaque fenêtre :

* **temporelles** : moyenne, écart-type, variance, min, max, amplitude, RMS,
  énergie, asymétrie (skewness), aplatissement (kurtosis), longueur de ligne ;
* **fréquentielles** : puissance des bandes Delta/Theta/Alpha/Beta, puissances
  relatives, rapports (theta/alpha, delta/alpha), fréquence dominante, entropie
  et centroïde spectrales ;
* **avancées** : paramètres de Hjorth, entropie de permutation, énergie en
  ondelettes, et **synchronie inter-canaux** (corrélation et cohérence moyennes
  entre canaux EEG — une absence est généralisée).

Tout est vectorisé par lots de fenêtres pour éviter les boucles Python lentes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pywt
from scipy import stats

from . import config
from .windowing import window_params, window_starts, window_labels_from_samples
from .io import list_mat_files, load_patient_signal, patient_id_from_stem
from .labeling import patient_seizures, sample_labels

EEG = "EEG"

#: Caractéristiques calculées pour chaque canal.
TIME_FEATS = (
    "mean", "std", "var", "min", "max", "amp", "rms", "energy",
    "skew", "kurtosis", "line_length",
)
BAND_NAMES = tuple(config.BANDS.keys())                    # Delta, Theta, Alpha, Beta
SPEC_FEATS = tuple(f"pow_{b}" for b in BAND_NAMES) + \
             tuple(f"rel_{b}" for b in BAND_NAMES) + \
             ("theta_alpha", "delta_alpha", "dom_freq", "spec_entropy", "spec_centroid")
HJORTH_FEATS = ("hjorth_activity", "hjorth_mobility", "hjorth_complexity")
OTHER_FEATS = ("perm_entropy",) + tuple(f"wav_e{i}" for i in range(1, 5))
#: Toutes les caractéristiques par canal (ordre stable).
CHANNEL_FEATS = TIME_FEATS + SPEC_FEATS + HJORTH_FEATS + OTHER_FEATS
#: Caractéristiques globales (indépendantes du canal).
GLOBAL_FEATS = ("corr_mean", "coh_delta", "coh_theta", "coh_mean")


# ---------------------------------------------------------------------------
# Blocs de caractéristiques (entrée x : (n_fenêtres, W, n_canaux))
# ---------------------------------------------------------------------------
def _time_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Caractéristiques temporelles, chacune de forme (n_fenêtres, n_canaux)."""
    mean = x.mean(axis=1)
    var = x.var(axis=1)
    std = np.sqrt(var)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    energy = np.sum(x * x, axis=1)
    return {
        "mean": mean, "std": std, "var": var, "min": mn, "max": mx,
        "amp": mx - mn, "rms": np.sqrt(energy / x.shape[1]), "energy": energy,
        "skew": stats.skew(x, axis=1), "kurtosis": stats.kurtosis(x, axis=1),
        "line_length": np.abs(np.diff(x, axis=1)).sum(axis=1),
    }


def _spectral_features(
    x: np.ndarray, fs: float, bands: dict[str, tuple[float, float]]
) -> dict[str, np.ndarray]:
    """Puissances de bandes, rapports, fréquence dominante et entropies."""
    n = x.shape[1]
    win = np.hanning(n)[None, :, None]
    spec = np.fft.rfft(x * win, axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    psd = (np.abs(spec) ** 2) / (fs * np.sum(win ** 2))
    psd = psd.astype(np.float64)
    psd[:, 1:-1, :] *= 2.0                                  # spectre unilatéral
    df = freqs[1] - freqs[0]

    masks = {b: (freqs >= lo) & (freqs < hi) for b, (lo, hi) in bands.items()}
    out: dict[str, np.ndarray] = {}
    total = np.zeros(x.shape[0:1] + x.shape[2:])
    for b, m in masks.items():
        p = psd[:, m, :].sum(axis=1) * df
        out[f"pow_{b}"] = p
        total = total + p
    for b in bands:
        out[f"rel_{b}"] = out[f"pow_{b}"] / np.where(total == 0, np.nan, total)
    out["theta_alpha"] = out["pow_Theta"] / np.where(out["pow_Alpha"] == 0, np.nan, out["pow_Alpha"])
    out["delta_alpha"] = out["pow_Delta"] / np.where(out["pow_Alpha"] == 0, np.nan, out["pow_Alpha"])

    band_mask = (freqs >= 0.5) & (freqs <= 30.0)
    idx = np.where(band_mask)[0]
    pf = psd[:, idx, :]
    fr = freqs[idx][None, :, None]
    out["dom_freq"] = freqs[idx][np.argmax(pf, axis=1)]
    ssum = pf.sum(axis=1) + 1e-12
    pnorm = pf / ssum[:, None, :]
    out["spec_entropy"] = -(pnorm * np.log(pnorm + 1e-12)).sum(axis=1) / np.log(len(idx))
    out["spec_centroid"] = (pf * fr).sum(axis=1) / ssum
    return out


def _hjorth_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Paramètres de Hjorth : activité, mobilité, complexité."""
    var0 = x.var(axis=1)
    d1 = np.diff(x, axis=1)
    var1 = d1.var(axis=1)
    activity = var0
    mobility = np.sqrt(var1 / np.where(var0 == 0, np.nan, var0))
    d2 = np.diff(d1, axis=1)
    mob_d1 = np.sqrt(d2.var(axis=1) / np.where(var1 == 0, np.nan, var1))
    complexity = mob_d1 / np.where(mobility == 0, np.nan, mobility)
    return {"hjorth_activity": activity, "hjorth_mobility": mobility,
            "hjorth_complexity": complexity}


def _perm_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> np.ndarray:
    """Entropie de permutation (normalisée) par fenêtre et par canal."""
    nc, n, nch = x.shape
    m = n - (order - 1) * delay
    out = np.zeros((nc, nch), dtype=np.float64)
    if m <= 1:
        return out
    log_fact = np.log(math.factorial(order))
    emb = np.stack([x[:, i:i + m, :] for i in range(order)], axis=2)  # (nc, m, order, nch)
    ranks = np.argsort(emb, axis=2, kind="stable")
    for c in range(nch):
        codes = np.ravel_multi_index(
            [ranks[:, :, k, c] for k in range(order)], (order,) * order
        )                                                    # (nc, m)
        for w in range(nc):
            _, counts = np.unique(codes[w], return_counts=True)
            p = counts / counts.sum()
            out[w, c] = -np.sum(p * np.log(p)) / log_fact
    return out


def _wavelet_energy(x: np.ndarray) -> dict[str, np.ndarray]:
    """Énergie relative par niveau d'ondelette (db4, 4 niveaux)."""
    nc, n, nch = x.shape
    out = {f"wav_e{i}": np.zeros((nc, nch)) for i in range(1, 5)}
    for w in range(nc):
        for c in range(nch):
            coeffs = pywt.wavedec(x[w, :, c], "db4", level=4)
            energies = [float(np.sum(ci * ci)) for ci in coeffs]
            tot = sum(energies) + 1e-12
            for i, e in enumerate(energies[:4], start=1):
                out[f"wav_e{i}"][w, c] = e / tot
    return out


def _synchrony_features(
    x: np.ndarray,
    spec: np.ndarray,
    freqs: np.ndarray,
    bands: dict[str, tuple[float, float]],
    eeg_idx: list[int],
) -> dict[str, np.ndarray]:
    """Synchronie inter-canaux : corrélation moyenne et cohérence moyenne (EEG)."""
    nc = x.shape[0]
    res = {k: np.zeros(nc) for k in GLOBAL_FEATS}
    if len(eeg_idx) < 2:
        return res
    xs = x[:, :, eeg_idx]
    centered = xs - xs.mean(axis=1, keepdims=True)
    norm = np.sqrt((centered ** 2).sum(axis=1))[:, None, :]
    denom = norm * np.transpose(norm, (0, 2, 1))
    csm = np.einsum("wti,wtj->wij", centered, centered)
    corr = csm / np.where(denom == 0, np.nan, denom)
    iu = np.triu_indices(len(eeg_idx), k=1)
    res["corr_mean"] = np.nanmean(np.abs(corr[:, iu[0], iu[1]]), axis=1)

    eeg_spec = spec[:, :, eeg_idx]
    coh_band = {}
    for b, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        Xb = eeg_spec[:, m, :]
        cs = np.einsum("wfc,wfd->wcd", Xb, np.conj(Xb))
        pw = np.real(np.einsum("wfc,wfc->wc", Xb, np.conj(Xb)))
        den = pw[:, :, None] * pw[:, None, :]
        coh = (np.abs(cs) ** 2) / np.where(den == 0, np.nan, den)
        coh_band[b] = np.nanmean(coh[:, iu[0], iu[1]], axis=1)
    res["coh_delta"] = coh_band.get("Delta", np.zeros(nc))
    res["coh_theta"] = coh_band.get("Theta", np.zeros(nc))
    res["coh_mean"] = np.nanmean(
        np.vstack([coh_band[b] for b in bands]), axis=0
    )
    return res


# ---------------------------------------------------------------------------
# Extraction complète pour un patient
# ---------------------------------------------------------------------------
def extract_features(
    sig: np.ndarray,
    fs: float,
    names: list[str],
    win_len: int,
    step: int,
    bands: dict[str, tuple[float, float]] | None = None,
    chunk: int = 256,
    include_perm: bool = True,
    include_wavelet: bool = True,
    eeg_only: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Calcule les features de toutes les fenêtres d'un patient.

    Args:
        sig: signal (n_échantillons, n_canaux), float32.
        fs: fréquence d'échantillonnage.
        names: noms des canaux (normalisés en interne).
        win_len, step: longueur de fenêtre et pas (en échantillons).
        bands: bandes de fréquences (défaut : config.BANDS).
        chunk: nombre de fenêtres traitées simultanément (borne la mémoire).
        include_perm, include_wavelet: activer les features coûteuses.
        eeg_only: si True (défaut), ne calcule les features que sur les canaux
            EEG (les canaux EMG/ECG/SLI sont des sources de bruit physiologique
            et d'artefacts, exclus du jeu de features principal).

    Returns:
        (DataFrame des features — 1 ligne par fenêtre — et indices de début).
    """
    bands = bands or config.BANDS
    names = [config.canonical_channel(n) for n in names]
    if eeg_only:
        keep = [i for i, n in enumerate(names) if config.is_eeg_channel(n)]
        names = [names[i] for i in keep]
        sig = sig[:, keep]
    eeg_idx = [i for i, n in enumerate(names) if config.is_eeg_channel(n)]
    starts = window_starts(sig.shape[0], win_len, step)
    n_win = len(starts)

    # accumulateurs : on empile des tableaux (n_fenêtres, n_canaux) par feature
    acc: dict[str, list[np.ndarray]] = {f: [] for f in CHANNEL_FEATS}
    gacc: dict[str, list[np.ndarray]] = {f: [] for f in GLOBAL_FEATS}
    n = sig.shape[1]

    for b0 in range(0, n_win, chunk):
        idx = starts[b0:b0 + chunk]
        x = np.stack([sig[s:s + win_len, :] for s in idx]).astype(np.float64)

        tfeats = _time_features(x)
        sfeats = _spectral_features(x, fs, bands)
        hfeats = _hjorth_features(x)
        for k, v in {**tfeats, **sfeats, **hfeats}.items():
            acc[k].append(v)
        acc["perm_entropy"].append(_perm_entropy(x) if include_perm else np.zeros((len(idx), n)))
        if include_wavelet:
            wf = _wavelet_energy(x)
            for k in wf:
                acc[k].append(wf[k])
        else:
            for i in range(1, 5):
                acc[f"wav_e{i}"].append(np.zeros((len(idx), n)))

        # spectre recalculé pour la synchronie (bandes limitées)
        win = np.hanning(win_len)[None, :, None]
        spec = np.fft.rfft(x * win, axis=1)
        freqs = np.fft.rfftfreq(win_len, 1.0 / fs)
        syn = _synchrony_features(x, spec, freqs, bands, eeg_idx)
        for k in GLOBAL_FEATS:
            gacc[k].append(syn[k])

    data: dict[str, np.ndarray] = {}
    for f in CHANNEL_FEATS:
        arr = np.concatenate(acc[f], axis=0) if acc[f] else np.zeros((n_win, n))
        for c, ch in enumerate(names):
            data[f"{ch}__{f}"] = arr[:, c].astype(np.float32)
    for f in GLOBAL_FEATS:
        data[f"global__{f}"] = np.concatenate(gacc[f], axis=0).astype(np.float32)

    df = pd.DataFrame(data)
    df.insert(0, "t_start", starts / fs)
    return df, starts


# ---------------------------------------------------------------------------
# Construction du dataset global « une ligne = une fenêtre »
# ---------------------------------------------------------------------------
def patient_window_dataset(
    mat_path,
    seizures: pd.DataFrame,
    win_len: int,
    step: int,
    fs: float = config.FS,
    **kwargs,
) -> pd.DataFrame:
    """Dataset de fenêtres d'un patient : features + colonnes patient/t_start/label.

    Les enregistrements sans crise annotée produisent des fenêtres toutes de
    classe 0 (« normal pur »).
    """
    pid = patient_id_from_stem(mat_path.stem)
    rec = load_patient_signal(mat_path)
    df, starts = extract_features(rec.sig, rec.fs, rec.names, win_len, step, **kwargs)
    y = sample_labels(rec.sig.shape[0], rec.fs, patient_seizures(seizures, pid))
    df.insert(0, "patient", pid)
    df.insert(1, "label", window_labels_from_samples(y, starts, win_len))
    return df


def build_feature_dataset(
    seizures: pd.DataFrame | None = None,
    cache_path=config.FEATURES_PARQUET,
    force: bool = False,
    window_sec: float = config.WINDOW_SEC,
    overlap: float = config.OVERLAP,
    **kwargs,
) -> pd.DataFrame:
    """Construit (ou recharge depuis le cache) le dataset de features complet.

    Args:
        seizures: annotations (défaut : rechargées depuis le fichier Excel).
        cache_path: fichier parquet de cache.
        force: recalculer même si le cache existe.
        window_sec, overlap: paramètres de fenêtrage.
        **kwargs: transmis à :func:`extract_features` (include_perm, ...).

    Returns:
        DataFrame avec ``patient, label, t_start`` et toutes les features.
    """
    from pathlib import Path
    from .io import load_annotations

    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    if seizures is None:
        seizures, _ = load_annotations()
    win_len, step = window_params(config.FS, window_sec, overlap)

    frames = []
    for p in list_mat_files():
        df = patient_window_dataset(p, seizures, win_len, step, **kwargs)
        frames.append(df)
        print(f"  features {patient_id_from_stem(p.stem)} : {df.shape}")
    data = pd.concat(frames, ignore_index=True)
    config.ensure_dirs()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cache_path, index=False)
    return data
