"""Fixtures partagées entre tous les fichiers de tests.

Centralise la création de signaux synthétiques et de DataFrames
pour éviter la duplication dans test_features.py, test_models.py, etc.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garantit que la racine du projet est dans sys.path quel que soit l'environnement ou la méthode d'appel
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Constantes communes
# ---------------------------------------------------------------------------
FS = 256.0
WIN_SEC = 2.0
WIN_LEN = int(WIN_SEC * FS)   # 512 échantillons
STEP = WIN_LEN // 2            # recouvrement 50 % = 256 échantillons
CHAN_NAMES = ["EEGFp1", "EEGFp2", "EEGCz"]
N_CHANS = len(CHAN_NAMES)
N_SAMPLES = WIN_LEN * 6        # ~6 fenêtres


# ---------------------------------------------------------------------------
# Helpers réutilisables (fonctions, pas fixtures, pour plus de flexibilité)
# ---------------------------------------------------------------------------
def make_sine_signal(
    freq: float = 3.0,
    n: int = N_SAMPLES,
    fs: float = FS,
    n_chans: int = N_CHANS,
    amp: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Signal synthétique (n, n_chans) : sinus + bruit blanc gaussien."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    base = amp * np.sin(2 * np.pi * freq * t)
    cols = [base + 0.05 * rng.standard_normal(n) for _ in range(n_chans)]
    return np.column_stack(cols).astype(np.float32)


def make_feature_df(
    n: int = 200,
    n_feats: int = 10,
    n_patients: int = 5,
    crisis_rate: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """DataFrame (patient, label, t_start, feat_*) entièrement synthétique."""
    rng = np.random.default_rng(seed)
    patients = [f"P{i:02d}" for i in range(n_patients)]
    data: dict = {
        "patient": np.tile(patients, n // n_patients + 1)[:n],
        "label": (rng.random(n) > (1 - crisis_rate)).astype(int),
        "t_start": np.arange(n) * 1.0,
    }
    for i in range(n_feats):
        data[f"feat_{i:02d}"] = rng.standard_normal(n).astype(np.float32)
    return pd.DataFrame(data)


def make_seizures_df(patients: list[str] | None = None) -> pd.DataFrame:
    """DataFrame d'annotations minimal (Patient, Debut_Sec, Fin_Sec, Duree_Sec)."""
    patients = patients or ["P00", "P01", "P02"]
    rows = []
    for i, pid in enumerate(patients):
        rows.append({
            "Patient": pid,
            "Debut_Sec": float(10 + i * 50),
            "Fin_Sec": float(20 + i * 50),
            "Duree_Sec": 10.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures pytest
# ---------------------------------------------------------------------------
@pytest.fixture
def sine_signal():
    """Signal EEG synthétique (N_SAMPLES, N_CHANS) à 3 Hz."""
    return make_sine_signal()


@pytest.fixture
def feature_df():
    """DataFrame synthétique prêt pour les modèles."""
    return make_feature_df()


@pytest.fixture
def seizures_df():
    """Annotations synthétiques pour 3 patients."""
    return make_seizures_df()


@pytest.fixture
def extracted_features(sine_signal):
    """Résultat de extract_features sur le signal synthétique (sans perm/wavelet)."""
    from src.features import extract_features
    df, starts = extract_features(
        sine_signal, FS, CHAN_NAMES, WIN_LEN, STEP,
        include_perm=False, include_wavelet=False,
    )
    return df, starts
