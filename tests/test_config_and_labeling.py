"""Tests unitaires pour src/config.py et src/labeling.py.

Vérifie les fonctions de normalisation de canaux, de détection de canaux EEG,
de filtrage de crises par patient, et de calcul de la cible d'échantillons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.labeling import (
    LABEL_NORMAL,
    LABEL_SEIZURE,
    class_balance,
    patient_seizures,
    sample_labels,
)


# ---------------------------------------------------------------------------
# src/config.py tests
# ---------------------------------------------------------------------------
def test_canonical_channel_normalization():
    assert config.canonical_channel("EMG_1_") == "EMG1"
    assert config.canonical_channel("EMG_2") == "EMG2"
    assert config.canonical_channel(" EEGFp1 ") == "EEGFp1"
    assert config.canonical_channel("EEGCz") == "EEGCz"


def test_is_eeg_channel():
    assert config.is_eeg_channel("EEGFp1") is True
    assert config.is_eeg_channel("eegCz") is True
    assert config.is_eeg_channel("EMG1") is False
    assert config.is_eeg_channel("SLI") is False
    assert config.is_eeg_channel("ECG") is False


def test_split_channels():
    chans = ["EEGFp1", "EMG1", "EEGCz", "SLI", "EEGO1"]
    eeg, non_eeg = config.split_channels(chans)
    assert eeg == ["EEGFp1", "EEGCz", "EEGO1"]
    assert non_eeg == ["EMG1", "SLI"]


def test_ensure_dirs(tmp_path, monkeypatch):
    test_cache = tmp_path / "test_cache"
    test_report = tmp_path / "test_report"
    monkeypatch.setattr(config, "CACHE_DIR", test_cache)
    monkeypatch.setattr(config, "FIGURES_DIR", test_report / "figures")

    config.ensure_dirs()
    assert test_cache.exists()
    assert (test_report / "figures").exists()


# ---------------------------------------------------------------------------
# src/labeling.py tests
# ---------------------------------------------------------------------------
def test_patient_seizures_filter():
    df = pd.DataFrame({
        "Patient": ["190304A-E", "190304A-E", "230515B-G"],
        "Debut_Sec": [10.0, 50.0, 100.0],
        "Fin_Sec": [20.0, 60.0, 110.0],
    })
    res = patient_seizures(df, "190304A-E")
    assert len(res) == 2
    assert (res["Patient"] == "190304A-E").all()

    res_empty = patient_seizures(df, "UNKNOWN_PATIENT")
    assert len(res_empty) == 0


def test_sample_labels_basic():
    fs = 10.0
    n_samples = 100  # 10 secondes (indices 0 à 99)
    # Crise de t=2.0s à t=4.0s -> échantillons 20 à 40 inclus
    seizures = pd.DataFrame([{"Debut_Sec": 2.0, "Fin_Sec": 4.0}])
    labels = sample_labels(n_samples, fs, seizures)

    assert labels.dtype == np.uint8
    assert len(labels) == n_samples
    assert (labels[:20] == LABEL_NORMAL).all()
    assert (labels[20:41] == LABEL_SEIZURE).all()
    assert (labels[41:] == LABEL_NORMAL).all()


def test_sample_labels_clipping():
    fs = 10.0
    n_samples = 50
    # Crise qui dépasse les bornes temporelles
    seizures = pd.DataFrame([{"Debut_Sec": -5.0, "Fin_Sec": 100.0}])
    labels = sample_labels(n_samples, fs, seizures)
    assert (labels == LABEL_SEIZURE).all()


def test_sample_labels_empty_seizures():
    fs = 256.0
    n_samples = 1000
    labels = sample_labels(n_samples, fs, pd.DataFrame(columns=["Debut_Sec", "Fin_Sec"]))
    assert len(labels) == n_samples
    assert (labels == LABEL_NORMAL).all()


def test_class_balance():
    labels = np.array([0, 0, 0, 1], dtype=np.uint8)
    bal = class_balance(labels)
    assert bal["n0"] == 3
    assert bal["n1"] == 1
    assert bal["pct0"] == 75.0
    assert bal["pct1"] == 25.0
