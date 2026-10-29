"""Tests unitaires de l'extraction de features (src/features.py).

On utilise un signal synthétique contrôlé (sinus + bruit) pour tester
sans dépendre des données réelles (.mat).  Les tests vérifient :
- la forme et les colonnes du DataFrame de sortie ;
- les features temporelles sur des signaux analytiquement connus ;
- les features spectrales (bandes, entropie) ;
- les features de Hjorth ;
- la synchronie inter-canaux (corrélation) ;
- le comportement avec include_perm=False / include_wavelet=False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    extract_features,
    CHANNEL_FEATS,
    GLOBAL_FEATS,
    TIME_FEATS,
    SPEC_FEATS,
    HJORTH_FEATS,
)
from src import config


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------
FS = 256.0
WIN_SEC = 2.0
WIN_LEN = int(WIN_SEC * FS)   # 512 échantillons
STEP = WIN_LEN // 2            # 50 % recouvrement = 256

# Canaux EEG synthétiques : on préfixe "EEG" pour que is_eeg_channel() les accepte
CHAN_NAMES = ["EEGFp1", "EEGFp2", "EEGCz"]
N_CHANS = len(CHAN_NAMES)
N_SAMPLES = WIN_LEN * 4        # 4 fenêtres pleines environ


def _make_signal(freq: float = 3.0, n: int = N_SAMPLES,
                 fs: float = FS, n_chans: int = N_CHANS,
                 seed: int = 0) -> np.ndarray:
    """Signal synthétique : sinus + bruit blanc gaussien, (n, n_chans)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    base = np.sin(2 * np.pi * freq * t)
    sig = np.column_stack([base + 0.05 * rng.standard_normal(n)
                           for _ in range(n_chans)])
    return sig.astype(np.float32)


@pytest.fixture
def simple_signal():
    return _make_signal()


@pytest.fixture
def simple_extract(simple_signal):
    df, starts = extract_features(
        simple_signal, FS, CHAN_NAMES, WIN_LEN, STEP,
        include_perm=False, include_wavelet=False,
    )
    return df, starts


# ---------------------------------------------------------------------------
# 1. Forme et colonnes du DataFrame
# ---------------------------------------------------------------------------
def test_output_is_dataframe(simple_extract):
    df, _ = simple_extract
    assert isinstance(df, pd.DataFrame)


def test_output_has_t_start_column(simple_extract):
    df, _ = simple_extract
    assert "t_start" in df.columns


def test_output_number_of_windows(simple_extract):
    df, starts = simple_extract
    assert len(df) == len(starts)
    assert len(df) > 0


def test_output_has_channel_features(simple_extract):
    df, _ = simple_extract
    # Chaque canal doit avoir au moins les features temporelles
    for chan in CHAN_NAMES:
        for feat in TIME_FEATS:
            col = f"{chan}__{feat}"
            assert col in df.columns, f"Colonne manquante : {col}"


def test_output_has_global_features(simple_extract):
    df, _ = simple_extract
    for feat in GLOBAL_FEATS:
        assert f"global__{feat}" in df.columns


def test_output_dtype_float32(simple_extract):
    df, _ = simple_extract
    feat_cols = [c for c in df.columns if c != "t_start"]
    for col in feat_cols:
        assert df[col].dtype == np.float32, f"dtype incorrect pour {col}"


def test_t_start_monotone(simple_extract):
    df, _ = simple_extract
    assert (df["t_start"].diff().dropna() > 0).all()


# ---------------------------------------------------------------------------
# 2. Features temporelles — valeurs analytiques
# ---------------------------------------------------------------------------
def test_mean_near_zero_for_sine():
    """Un sinus centré a une moyenne proche de 0."""
    sig = _make_signal(freq=3.0, n=N_SAMPLES * 4)  # beaucoup d'échantillons
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        assert df[f"{chan}__mean"].abs().mean() < 0.2, \
            f"Moyenne trop élevée pour {chan}"


def test_std_positive():
    """L'écart-type doit être positif pour un signal non constant."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        assert (df[f"{chan}__std"] > 0).all()


def test_amplitude_geq_std():
    """L'amplitude (max-min) est toujours >= écart-type."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        assert (df[f"{chan}__amp"] >= df[f"{chan}__std"] - 1e-4).all()


def test_rms_positive():
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        assert (df[f"{chan}__rms"] > 0).all()


# ---------------------------------------------------------------------------
# 3. Features spectrales
# ---------------------------------------------------------------------------
def test_spectral_power_bands_positive():
    """Les puissances de bandes doivent être non-négatives."""
    sig = _make_signal(freq=3.0)
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        for band in config.BANDS:
            col = f"{chan}__pow_{band}"
            assert (df[col] >= 0).all(), f"Puissance négative dans {col}"


def test_relative_power_sums_to_one():
    """La somme des puissances relatives par canal ≈ 1."""
    sig = _make_signal(freq=3.0)
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        rel_cols = [f"{chan}__rel_{b}" for b in config.BANDS]
        total = df[rel_cols].sum(axis=1)
        # Tolérance pour les NaN éventuels et erreurs numériques
        valid = total.dropna()
        np.testing.assert_allclose(valid, 1.0, atol=1e-4,
                                   err_msg=f"Puissances relatives ne somment pas à 1 pour {chan}")


def test_delta_dominant_for_3hz_sine():
    """Un sinus à 3 Hz (bande Delta 0,5-4 Hz) doit dominer en puissance Delta."""
    sig = _make_signal(freq=3.0, n=N_SAMPLES * 4)
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    chan = CHAN_NAMES[0]
    assert df[f"{chan}__pow_Delta"].mean() > df[f"{chan}__pow_Alpha"].mean()
    assert df[f"{chan}__pow_Delta"].mean() > df[f"{chan}__pow_Beta"].mean()


def test_spec_entropy_in_range():
    """L'entropie spectrale est normalisée entre 0 et 1."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        col = f"{chan}__spec_entropy"
        assert (df[col].dropna() >= 0).all()
        assert (df[col].dropna() <= 1.0 + 1e-4).all()


# ---------------------------------------------------------------------------
# 4. Features de Hjorth
# ---------------------------------------------------------------------------
def test_hjorth_activity_equals_variance():
    """L'activité de Hjorth est la variance du signal."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    chan = CHAN_NAMES[0]
    # Activité positive
    assert (df[f"{chan}__hjorth_activity"] >= 0).all()


def test_hjorth_mobility_positive():
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        assert (df[f"{chan}__hjorth_mobility"].dropna() >= 0).all()


# ---------------------------------------------------------------------------
# 5. Synchronie inter-canaux
# ---------------------------------------------------------------------------
def test_corr_mean_in_range():
    """La corrélation moyenne inter-canaux est entre 0 et 1."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    col = df["global__corr_mean"].dropna()
    assert (col >= 0).all()
    assert (col <= 1.0 + 1e-4).all()


def test_corr_high_for_identical_channels():
    """Canaux identiques → corrélation proche de 1."""
    base = _make_signal(freq=3.0)[:, 0]
    sig = np.column_stack([base, base, base]).astype(np.float32)
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    assert df["global__corr_mean"].mean() > 0.95


# ---------------------------------------------------------------------------
# 6. Paramètres optionnels
# ---------------------------------------------------------------------------
def test_include_perm_false_produces_zeros():
    """Avec include_perm=False, l'entropie de permutation vaut 0."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        assert (df[f"{chan}__perm_entropy"] == 0).all()


def test_include_wavelet_false_produces_zeros():
    """Avec include_wavelet=False, les énergies ondelettes valent 0."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    for chan in CHAN_NAMES:
        for i in range(1, 5):
            assert (df[f"{chan}__wav_e{i}"] == 0).all()


def test_include_perm_true_nonzero():
    """Avec include_perm=True (défaut), l'entropie de permutation est non nulle."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=True, include_wavelet=False)
    for chan in CHAN_NAMES:
        assert (df[f"{chan}__perm_entropy"] > 0).any()


def test_include_wavelet_true_nonzero():
    """Avec include_wavelet=True, les énergies ondelettes sont non nulles."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=True)
    for chan in CHAN_NAMES:
        assert (df[f"{chan}__wav_e1"] > 0).any()


# ---------------------------------------------------------------------------
# 7. Cas limites
# ---------------------------------------------------------------------------
def test_single_window():
    """Un signal juste assez long pour une seule fenêtre."""
    sig = _make_signal(n=WIN_LEN)
    df, starts = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                                   include_perm=False, include_wavelet=False)
    assert len(df) == 1
    assert starts[0] == 0


def test_no_nan_in_output_for_clean_signal():
    """Un signal propre ne doit pas produire de NaN dans les features de base."""
    sig = _make_signal()
    df, _ = extract_features(sig, FS, CHAN_NAMES, WIN_LEN, STEP,
                              include_perm=False, include_wavelet=False)
    time_cols = [f"{c}__{f}" for c in CHAN_NAMES for f in TIME_FEATS]
    assert not df[time_cols].isnull().any().any()
