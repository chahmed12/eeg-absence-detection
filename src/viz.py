"""Visualisations EEG : multi-canaux (vue clinique), heatmap, spectrogramme,
et visualisations de cohorte (multi-patients).

Toutes les figures peuvent être sauvegardées dans ``report/figures`` via
:func:`save_fig` afin d'alimenter le rapport.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # rendu sans interface (notebooks, scripts, serveur)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from . import config


# ---------------------------------------------------------------------------
# Style « rapport » (opt-in)
# ---------------------------------------------------------------------------
# Les figures sont dessinées large par défaut (usage interactif / dashboard).
# Pour le rapport, ``set_report_style()`` les redimensionne à la largeur d'une
# page et augmente les polices, faute de quoi tout est réduit (~x0,4) à
# l'insertion dans le PDF et devient illisible.
_REPORT: dict[str, float | bool] = {"on": False, "w": 6.2, "hmax": 5.8}


def set_report_style(fig_width: float = 6.2, height_max: float = 5.8) -> None:
    """Bascule les figures en style « impression » (rapport).

    À appeler au début des scripts qui alimentent le rapport : figures
    dessinées à la largeur d'une page (donc non réduites à l'insertion) et
    polices dimensionnées pour une lecture confortable.
    """
    _REPORT.update(on=True, w=fig_width, hmax=height_max)
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "axes.grid": True,
        "grid.alpha": 0.4,
    })


def _fw(default: float) -> float:
    """Largeur de figure : ``default`` (interactif) ou largeur page (rapport)."""
    return min(default, float(_REPORT["w"])) if _REPORT["on"] else default


def _fh(default: float) -> float:
    """Hauteur de figure : ``default`` (interactif) ou plafonnée (rapport)."""
    return min(default, float(_REPORT["hmax"])) if _REPORT["on"] else default


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def save_fig(fig: plt.Figure, name: str, dpi: int = 200, ext: str = "png") -> Path:
    """Sauvegarde une figure dans ``report/figures/<name>.<ext>`` et renvoie le chemin.

    ``ext="pdf"`` produit une figure **vectorielle** (netteté parfaite, sans
    pixels) : utile pour les spectrogrammes/heatmaps insérés en grand dans le
    rapport et que l'on veut lire sans crénelage même en zoomant.
    """
    config.ensure_dirs()
    path = config.FIGURES_DIR / f"{name}.{ext}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _t_axis(n_samples: int, fs: float) -> np.ndarray:
    """Axe temporel en secondes pour ``n_samples`` échantillons à ``fs`` Hz."""
    return np.arange(n_samples) / fs


def _crop(
    sig: np.ndarray,
    fs: float,
    t_min: float | None,
    t_max: float | None,
) -> tuple[np.ndarray, float]:
    """Découpe ``sig`` sur l'intervalle ``[t_min, t_max]`` (secondes).

    Renvoie le signal rogné et l'instant de départ (en s) pour recaler les axes.
    Si aucun bornage n'est demandé, renvoie le signal tel quel avec un décalage nul.
    """
    n = sig.shape[0]
    if t_min is None and t_max is None:
        return sig, 0.0
    lo = int(round((t_min if t_min is not None else 0.0) * fs))
    hi = int(round((t_max if t_max is not None else n / fs) * fs))
    lo, hi = max(0, lo), min(n, hi)
    if hi <= lo:
        return sig, 0.0
    return sig[lo:hi], lo / fs


def _filter_band(x: np.ndarray, fs: float, low: float, high: float,
                 order: int = 4, axis: int = -1) -> np.ndarray:
    """Filtre passe-bande Butterworth en phase nulle (``filtfilt``)."""
    sos = sp_signal.butter(order, [low, high], "bandpass", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, x, axis=axis)


def _shade_seizures(
    ax: plt.Axes,
    seizures: pd.DataFrame,
    color: str = "red",
    t_min: float | None = None,
    t_max: float | None = None,
) -> None:
    """Surligne les intervalles de crises sur un axe (alpha translucide).

    Si ``t_min``/``t_max`` sont fournis, seules les crises **chevauchant** cette
    fenêtre sont tracées (sinon ``axvspan`` étire l'axe bien au-delà du signal
    réellement affiché).
    """
    first = True
    for _, r in seizures.iterrows():
        s, e = float(r["Debut_Sec"]), float(r["Fin_Sec"])
        if t_max is not None and s > t_max:
            continue
        if t_min is not None and e < t_min:
            continue
        ax.axvspan(s, e, color=color, alpha=0.25,
                   label="Crise annotée" if first else None)
        first = False


# ---------------------------------------------------------------------------
# Vue « EEG clinique » multi-canaux
# ---------------------------------------------------------------------------
def plot_multichannel_stack(
    t: np.ndarray,
    sig: np.ndarray,
    names: list[str],
    seizures: pd.DataFrame | None = None,
    channels: list[str] | None = None,
    fs: float = config.FS,
    t_min: float | None = None,
    t_max: float | None = None,
    scale_uv: float | None = None,
    normalize: bool = False,
    title: str = "Vue EEG multi-canaux",
) -> plt.Figure:
    """Trace les canaux empilés avec décalage vertical (présentation clinique).

    Args:
        t: axe temporel (s) de longueur ``sig.shape[0]``.
        sig: signal (n_échantillons, n_canaux).
        names: noms de tous les canaux de ``sig``.
        seizures: crises du patient à surligner.
        channels: sous-ensemble de canaux à afficher (défaut : tous).
        fs: fréquence d'échantillonnage.
        t_min, t_max: fenêtre temporelle affichée (s).
        scale_uv: amplitude verticale par canal (µV) ; auto si None.
        normalize: si True, ramène chaque canal à une amplitude unitaire
            (99e percentile) — fait ressortir la synchronie inter-canaux quels
            que soient les écarts d'amplitude entre électrodes.
    """
    channels = channels or names
    idxs = [names.index(c) for c in channels]
    i0 = 0 if t_min is None else max(0, int(round(t_min * fs)))
    i1 = len(t) if t_max is None else min(len(t), int(round(t_max * fs)))
    seg = sig[i0:i1, :][:, idxs].astype(np.float64)
    tt = t[i0:i1]
    if seg.size == 0:
        raise ValueError("Fenêtre temporelle vide (t_min/t_max hors signal).")

    # centrage par canal : retire l'offset DC résiduel et aligne les tracés
    seg = seg - np.median(seg, axis=0, keepdims=True)

    if normalize:
        amp = np.percentile(np.abs(seg), 99, axis=0)
        amp[amp <= 0] = 1.0
        seg = seg / amp
        scale_uv = 1.0
        unit_label = "amplitude normalisée (u.a.)"
    else:
        if scale_uv is None:
            # amplitude robuste (99e percentile) : un canal de grande amplitude
            # n'écrase plus tous les autres
            scale_uv = float(np.median(np.percentile(np.abs(seg), 99, axis=0))) or 1.0
        unit_label = f"{scale_uv:.0f} µV"

    off = 2.5 * scale_uv
    fig, ax = plt.subplots(figsize=(_fw(15), _fh(max(4, 0.6 * len(channels)))))
    for k, ch in enumerate(channels):
        ax.plot(tt, seg[:, k] + k * off, lw=0.6, color="black")
        ax.text(tt[0], k * off + off * 0.15, ch, ha="left", va="bottom",
                clip_on=True)
    if seizures is not None and len(seizures):
        _shade_seizures(ax, seizures, t_min=tt[0], t_max=tt[-1])
    ax.set_xlim(tt[0], tt[-1])
    ax.set_ylim(-off, len(channels) * off)
    # barre d'échelle verticale (amplitude réelle en µV)
    xb = tt[0] + 0.01 * (tt[-1] - tt[0])
    ax.plot([xb, xb], [0, scale_uv], color="black", lw=3, solid_capstyle="butt")
    ax.text(xb + 0.01 * (tt[-1] - tt[0]), scale_uv / 2, unit_label,
            va="center", ha="left")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Canaux décalés (amplitude relative)")
    ax.set_title(title)
    ax.set_yticks([])
    ax.grid(True, axis="x", ls="--", alpha=0.4)
    if seizures is not None and len(seizures):
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_signal_overview(
    t: np.ndarray,
    sig: np.ndarray,
    names: list[str],
    channel: str,
    seizures: pd.DataFrame | None = None,
    fs: float = config.FS,
    title: str | None = None,
) -> plt.Figure:
    """Signal d'un canal sur toute la durée, crises surlignées."""
    c = names.index(channel)
    x = sig[:, c].astype(np.float64)
    x = x - np.median(x)                       # retire l'offset DC résiduel
    fig, ax = plt.subplots(figsize=(_fw(15), _fh(4)))
    ax.plot(t, x, lw=0.4, color="steelblue", label=channel)
    lim = float(np.percentile(np.abs(x), 99.5)) or 1.0
    ax.set_ylim(-3 * lim, 3 * lim)             # échelle robuste (pics isolés)
    if seizures is not None and len(seizures):
        _shade_seizures(ax, seizures, t_min=float(t[0]), t_max=float(t[-1]))
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title or f"Signal {channel} et crises annotées")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Heatmap canaux x temps (RMS par petite fenêtre)
# ---------------------------------------------------------------------------
def compute_rms_matrix(
    sig: np.ndarray,
    fs: float,
    bin_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """RMS par canal et par fenêtre de ``bin_sec`` secondes.

    Returns:
        (rms[canaux, temps], centres_de_temps_en_s).
    """
    step = int(round(bin_sec * fs))
    n_bins = sig.shape[0] // step
    trimmed = sig[: n_bins * step]
    # reshape en (n_bins, step, n_canaux) puis RMS sur l'axe temps
    blocks = trimmed.reshape(n_bins, step, sig.shape[1])
    rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1)).T  # (canaux, bins)
    centers = (np.arange(n_bins) + 0.5) * bin_sec
    return rms, centers


def plot_channel_heatmap(
    sig: np.ndarray,
    names: list[str],
    fs: float = config.FS,
    seizures: pd.DataFrame | None = None,
    bin_sec: float = 1.0,
    t_min: float | None = None,
    t_max: float | None = None,
    normalize: str = "none",
    title: str = "Heatmap canaux x temps (RMS)",
) -> plt.Figure:
    """Heatmap canaux x temps du RMS, avec crises superposées.

    ``t_min``/``t_max`` restreignent l'affichage à une fenêtre temporelle (s).
    ``normalize="zscore"`` centre et réduit chaque canal sur la fenêtre, ce qui
    fait ressortir la synchronie inter-canaux quels que soient les écarts
    d'amplitude entre électrodes.
    """
    sig, t_off = _crop(sig, fs, t_min, t_max)
    rms, centers = compute_rms_matrix(sig, fs, bin_sec)
    centers = centers + t_off

    if normalize == "zscore":
        mu = rms.mean(axis=1, keepdims=True)
        sd = rms.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        data = (rms - mu) / sd
        vmin, vmax, cmap, cbar = -3.0, 3.0, "RdBu_r", "z-score (par canal)"
    else:
        data = rms
        vmin, vmax, cmap, cbar = 0.0, float(np.percentile(rms, 99)), "viridis", "RMS (µV)"

    fig, ax = plt.subplots(figsize=(_fw(15), _fh(max(4, 0.35 * len(names)))))
    im = ax.imshow(
        data, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[centers[0], centers[-1], len(names), 0], interpolation="nearest",
    )
    if seizures is not None and len(seizures):
        _shade_seizures(ax, seizures, t_min=float(centers[0]), t_max=float(centers[-1]))
    ax.set_yticks(np.arange(len(names)) + 0.5)
    ax.set_yticklabels(names)
    ax.set_xlabel("Temps (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectrogramme
# ---------------------------------------------------------------------------
def plot_spectrogram(
    sig: np.ndarray,
    names: list[str],
    channel: str,
    fs: float = config.FS,
    fmax: float = 30.0,
    seizures: pd.DataFrame | None = None,
    nperseg_sec: float = 2.0,
    t_min: float | None = None,
    t_max: float | None = None,
    ref_hz: float | None = 3.0,
    whiten: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """Spectrogramme d'un canal, focalisé sur les basses fréquences (≤ ``fmax``).

    La décharge pointe-onde de l'absence se situe vers 3 Hz : ``fmax`` par
    défaut à 30 Hz suffit à la mettre en évidence. ``t_min``/``t_max``
    restreignent l'analyse à une fenêtre (s) ; ``ref_hz`` trace une ligne de
    référence pointillée. ``whiten=True`` soustrait, fréquence par fréquence,
    la puissance médiane de la fenêtre (blanchiment spectral) : le fond 1/f est
    aplati et la décharge apparaît comme une bande horizontale nette.
    """
    c = names.index(channel)
    x, t_off = _crop(sig[:, c].astype(np.float64), fs, t_min, t_max)
    nperseg = int(nperseg_sec * fs)
    nperseg = max(256, min(nperseg, x.size))
    f, tt, Sxx = sp_signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    tt = tt + t_off
    mask = f <= fmax
    Sdb = 10 * np.log10(Sxx[mask] + 1e-12)
    if whiten:
        Sdb = Sdb - np.median(Sdb, axis=1, keepdims=True)
        vmin, vmax = -6.0, 6.0
        cbar_label = "Puissance relative (dB)"
    else:
        vmin = float(np.percentile(Sdb, 5))
        vmax = float(np.percentile(Sdb, 99.5))
        cbar_label = "Puissance (dB)"
    fig, ax = plt.subplots(figsize=(_fw(15), _fh(4)))
    im = ax.pcolormesh(tt, f[mask], Sdb, shading="auto", cmap="magma",
                       vmin=vmin, vmax=vmax)
    if ref_hz is not None and ref_hz <= fmax:
        ax.axhline(ref_hz, color="white", lw=0.9, ls="--", alpha=0.9)
        ax.text(tt[0], ref_hz + 0.4, f"{ref_hz:g} Hz", color="white",
                va="bottom")
    if seizures is not None and len(seizures):
        _shade_seizures(ax, seizures, color="cyan", t_min=float(tt[0]), t_max=float(tt[-1]))
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence (Hz)")
    ax.set_title(title or f"Spectrogramme {channel}")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    return fig


def plot_seizure_spectrogram(
    sig: np.ndarray,
    names: list[str],
    channel: str,
    fs: float = config.FS,
    seizures: pd.DataFrame | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    fmax: float = 8.0,
    nperseg_sec: float = 2.0,
    noverlap_frac: float = 0.75,
    band: tuple[float, float] = (0.5, 20.0),
    ref_hz: float | None = 3.0,
    title: str | None = None,
) -> plt.Figure:
    """Figure composite « décharge d'absence » : signal + spectrogramme.

    Panneau supérieur : trace EEG filtrée (bande ``band``) montrant la décharge
    rythmique. Panneau inférieur : spectrogramme (≤ ``fmax``) **normalisé par
    rapport au fond** (blanchiment), où la décharge ressort comme une bande
    horizontale soutenue autour de ``ref_hz``.
    """
    c = names.index(channel)
    x, t_off = _crop(sig[:, c].astype(np.float64), fs, t_min, t_max)
    tt = np.arange(x.size) / fs + t_off
    xf = _filter_band(x, fs, band[0], band[1])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(_fw(15), _fh(5)), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.7], "hspace": 0.08},
    )

    ax1.plot(tt, xf, lw=0.6, color="black")
    lim = float(np.percentile(np.abs(xf), 99.5)) or 1.0
    ax1.set_ylim(-3 * lim, 3 * lim)
    ax1.set_ylabel("Amplitude (µV)")
    ax1.set_title(title or f"Signal {channel} et spectrogramme")

    nperseg = max(256, min(int(nperseg_sec * fs), xf.size))
    noverlap = int(round(nperseg * noverlap_frac))
    f, ts, Sxx = sp_signal.spectrogram(xf, fs=fs, nperseg=nperseg,
                                       noverlap=noverlap)
    ts = ts + t_off
    mask = f <= fmax
    Sdb = 10 * np.log10(Sxx[mask] + 1e-12)
    Sdb = Sdb - np.median(Sdb, axis=1, keepdims=True)   # blanchiment spectral
    im = ax2.pcolormesh(ts, f[mask], Sdb, shading="auto", cmap="turbo",
                        vmin=-6.0, vmax=6.0)
    if ref_hz is not None and ref_hz <= fmax:
        ax2.axhline(ref_hz, color="white", lw=0.9, ls="--", alpha=0.9)
        ax2.text(ts[0], ref_hz + 0.25, f"{ref_hz:g} Hz", color="white", va="bottom")
    if seizures is not None and len(seizures):
        _shade_seizures(ax1, seizures, color="red",
                        t_min=float(tt[0]), t_max=float(tt[-1]))
        _shade_seizures(ax2, seizures, color="cyan",
                        t_min=float(ts[0]), t_max=float(ts[-1]))
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Fréquence (Hz)")
    ax2.set_xlim(tt[0], tt[-1])
    fig.colorbar(im, ax=ax2, label="Puissance relative (dB)")
    fig.tight_layout()
    return fig


def plot_band_power_heatmap(
    sig: np.ndarray,
    names: list[str],
    fs: float = config.FS,
    seizures: pd.DataFrame | None = None,
    band: tuple[float, float] = (2.5, 3.5),
    bin_sec: float = 0.25,
    t_min: float | None = None,
    t_max: float | None = None,
    channels: list[str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Heatmap canaux x temps de la puissance dans la bande ``band``.

    On isole la bande de la décharge (par défaut 2,5–3,5 Hz), on en calcule le
    RMS par fenêtre de ``bin_sec`` puis on l'exprime en dB **par rapport au fond
    propre à chaque canal** (médiane temporelle). La crise ressort alors comme
    une bande verticale franche : c'est la signature de la synchronie
    inter-canaux, bien plus lisible que le RMS large-bande.
    """
    channels = channels or [c for c in names if config.is_eeg_channel(c)]
    idxs = [names.index(c) for c in channels]
    seg, t_off = _crop(sig[:, idxs].astype(np.float64), fs, t_min, t_max)
    segf = _filter_band(seg, fs, band[0], band[1], axis=0)
    rms, centers = compute_rms_matrix(segf, fs, bin_sec)
    centers = centers + t_off
    ref = np.median(rms, axis=1, keepdims=True)
    ref[ref <= 0] = 1e-12
    data = 10 * np.log10(rms / ref)
    lim = float(np.percentile(np.abs(data), 98)) or 1.0

    # pcolormesh (et non imshow) : chaque cellule devient un rectangle
    # **vectoriel** -> figure nette à tout zoom (pas de pixels), adaptée au PDF.
    n_bins = data.shape[1]
    t_edges = t_off + np.arange(n_bins + 1) * bin_sec
    y_edges = np.arange(len(channels) + 1)
    fig, ax = plt.subplots(figsize=(_fw(15), _fh(max(4, 0.28 * len(channels)))))
    mesh = ax.pcolormesh(
        t_edges, y_edges, data, cmap="RdBu_r", vmin=-lim, vmax=lim,
        shading="flat", edgecolors="none", linewidth=0,
    )
    ax.set_ylim(len(channels), 0)          # canal 0 en haut (comme imshow upper)
    ax.set_xlim(t_edges[0], t_edges[-1])
    if seizures is not None and len(seizures):
        _shade_seizures(ax, seizures, t_min=float(t_edges[0]), t_max=float(t_edges[-1]))
    ax.set_yticks(np.arange(len(channels)) + 0.5)
    ax.set_yticklabels(channels)
    ax.set_xlabel("Temps (s)")
    ax.set_title(title or f"Puissance {band[0]:g}–{band[1]:g} Hz par canal (dB re fond)")
    fig.colorbar(mesh, ax=ax, label="dB re fond")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Visualisations de cohorte (multi-patients)
# ---------------------------------------------------------------------------
def plot_cohort_timeline(
    durations: dict[str, float],
    seizures: pd.DataFrame,
    title: str = "Frise chronologique de la cohorte",
) -> plt.Figure:
    """Frise : un patient par ligne, du début à la fin du tracé, crises marquées.

    Args:
        durations: dict {patient: durée du tracé en s}.
        seizures: tableau global des crises.
    """
    patients = list(durations.keys())
    fig, ax = plt.subplots(figsize=(_fw(14), _fh(max(4, 0.4 * len(patients)))))
    for k, pid in enumerate(patients):
        dur = durations[pid]
        ax.hlines(k, 0, dur, color="lightgray", lw=6)
        sub = seizures[seizures["Patient"].astype(str) == str(pid)]
        for _, r in sub.iterrows():
            ax.hlines(k, r["Debut_Sec"], r["Fin_Sec"], color="crimson", lw=6)
    ax.set_yticks(range(len(patients)))
    ax.set_yticklabels(patients)
    ax.set_xlabel("Temps (s)")
    ax.set_title(title)
    ax.invert_yaxis()
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], color="crimson", lw=6)],
              ["Crise"], loc="upper right")
    ax.grid(True, axis="x", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_duration_violin(
    seizures: pd.DataFrame,
    title: str = "Durée des crises par patient",
) -> plt.Figure:
    """Violin plot + points des durées de crises, par patient."""
    patients = sorted(seizures["Patient"].astype(str).unique())
    data = [seizures.loc[seizures["Patient"].astype(str) == p, "Duree_Sec"].values for p in patients]
    fig, ax = plt.subplots(figsize=(_fw(14), _fh(5)))
    ax.violinplot(data, showmeans=False, showextrema=False)
    for k, d in enumerate(data):
        ax.scatter(np.full(len(d), k + 1), d, s=12, color="crimson", zorder=3, alpha=0.7)
    ax.set_xticks(range(1, len(patients) + 1))
    ax.set_xticklabels(patients, rotation=90)
    ax.set_ylabel("Durée (s)")
    ax.set_title(title)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_interval_hist(
    intervals_s: np.ndarray,
    title: str = "Distribution des intervalles inter-crises",
) -> plt.Figure:
    """Histogramme des intervalles entre fins et débuts de crises successives."""
    fig, ax = plt.subplots(figsize=(_fw(9), _fh(4)))
    if len(intervals_s):
        ax.hist(intervals_s, bins=min(30, max(5, len(intervals_s) // 2)),
                color="teal", edgecolor="white")
        ax.axvline(np.mean(intervals_s), color="crimson", ls="--",
                   label=f"moyenne {np.mean(intervals_s):.0f} s")
        ax.axvline(np.median(intervals_s), color="black", ls=":",
                   label=f"médiane {np.median(intervals_s):.0f} s")
        ax.legend()
    ax.set_xlabel("Intervalle (s)")
    ax.set_ylabel("Nombre d'intervalles")
    ax.set_title(title)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def inter_crisis_intervals(seizures: pd.DataFrame) -> np.ndarray:
    """Intervalles (s) entre la fin d'une crise et le début de la suivante.

    Calcul effectué patient par patient ; les chevauchements (intervalles
    négatifs, anomalies de saisie) sont ignorés.
    """
    vals: list[float] = []
    for _, g in seizures.groupby("Patient"):
        g = g.sort_values("Debut_Sec")
        if len(g) > 1:
            gaps = g["Debut_Sec"].iloc[1:].to_numpy() - g["Fin_Sec"].iloc[:-1].to_numpy()
            vals.extend(gaps[gaps >= 0].tolist())
    return np.asarray(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Détection : réel vs prédit
# ---------------------------------------------------------------------------
def plot_detection(
    t: np.ndarray,
    sig: np.ndarray,
    names: list[str],
    channel: str,
    true_events: list[tuple[float, float]],
    detected_events: list[tuple[float, float]],
    title: str = "Détection : crises réelles vs prédites",
) -> plt.Figure:
    """Signal d'un canal + crises réelles (vert) + détections (contour rouge)."""
    c = names.index(channel)
    fig, ax = plt.subplots(figsize=(_fw(15), _fh(4)))
    ax.plot(t, sig[:, c], lw=0.4, color="steelblue")
    for i, (s, e) in enumerate(true_events):
        ax.axvspan(s, e, color="green", alpha=0.25,
                   label="Crise réelle" if i == 0 else None)
    for i, (s, e) in enumerate(detected_events):
        ax.axvspan(s, e, facecolor="none", edgecolor="red", lw=1.4,
                   label="Détection" if i == 0 else None)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title)
    if true_events or detected_events:
        ax.legend(loc="upper right")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_probability(
    t: np.ndarray,
    score: np.ndarray,
    threshold: float = 0.5,
    true_events: list[tuple[float, float]] | None = None,
    title: str = "Probabilité de crise au cours du temps",
) -> plt.Figure:
    """Courbe de probabilité de crise (post-traitée) et seuil de décision."""
    fig, ax = plt.subplots(figsize=(_fw(15), _fh(4)))
    if true_events:
        for i, (s, e) in enumerate(true_events):
            ax.axvspan(s, e, color="green", alpha=0.2,
                       label="Crise réelle" if i == 0 else None)
    ax.plot(t, score, lw=0.8, color="darkorange", label="Probabilité de crise")
    ax.axhline(threshold, color="red", ls="--", lw=1, label=f"Seuil = {threshold:.2f}")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Probabilité")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Explicabilité
# ---------------------------------------------------------------------------
_ELECTRODE_MAP = {
    "EEGFp1": "Fp1", "EEGFp2": "Fp2", "EEGF7": "F7", "EEGF3": "F3", "EEGFz": "Fz",
    "EEGF4": "F4", "EEGF8": "F8", "EEGT3": "T3", "EEGC3": "C3", "EEGCz": "Cz",
    "EEGC4": "C4", "EEGT4": "T4", "EEGT5": "T5", "EEGP3": "P3", "EEGPz": "Pz",
    "EEGP4": "P4", "EEGT6": "T6", "EEGO1": "O1", "EEGO2": "O2",
}


def plot_channel_topomap(
    channel_importance: dict[str, float],
    title: str = "Importance par canal (carte topographique)",
    cmap: str = "Reds",
) -> plt.Figure:
    """Carte topographique du scalp (montage 10-20) de l'importance par canal.

    Seuls les canaux EEG standard sont représentés ; leur importance est
    normalisée entre 0 et 1 pour la lisibilité.
    """
    import mne

    noms = [c for c in channel_importance if c in _ELECTRODE_MAP]
    if len(noms) < 3:
        raise ValueError("Pas assez de canaux EEG standard pour une carte topographique.")
    elec = [_ELECTRODE_MAP[c] for c in noms]
    vals = np.array([channel_importance[c] for c in noms], dtype=float)
    vals = (vals - vals.min()) / (np.ptp(vals) + 1e-12)

    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(elec, sfreq=config.FS, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")

    fig, ax = plt.subplots(figsize=(_fw(6), _fh(5)))
    im, _ = mne.viz.plot_topomap(vals, info, axes=ax, cmap=cmap, show=False,
                                 sensors=True, contours=4)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Importance normalisée")
    fig.tight_layout()
    return fig


def plot_top_features(
    importances: pd.Series, top_n: int = 30,
    title: str = "Features les plus importantes",
) -> plt.Figure:
    """Diagramme en barres des ``top_n`` features les plus importantes."""
    top = importances.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(_fw(10), _fh(max(4, 0.28 * top_n))))
    ax.barh(top.index, top.values, color="teal")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, axis="x", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_band_importance(
    bands: dict[str, float],
    title: str = "Poids des bandes de fréquences",
    xlabel: str = "Information mutuelle cumulée",
    color: str = "teal",
) -> plt.Figure:
    """Barres du poids cumulé par bande (Delta/Theta/Alpha/Beta)."""
    order = [b for b in ("Delta", "Theta", "Alpha", "Beta") if b in bands]
    order += [b for b in bands if b not in order]
    vals = [bands[b] for b in order]
    fig, ax = plt.subplots(figsize=(_fw(7), _fh(4)))
    ax.bar(order, vals, color=color)
    ax.set_ylabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig
