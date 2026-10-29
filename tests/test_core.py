"""Tests unitaires des briques de base : fenêtrage, étiquetage, post-traitement,
correspondance patient et appariement d'événements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.io import patient_id_from_stem
from src.labeling import class_balance, sample_labels
from src.windowing import (
    count_windows, window_labels_from_samples, window_params, window_starts,
)
from src.postprocess import binary_to_intervals, merge_gaps, smooth_scores
from src.evaluation import match_events


# --- Fenêtrage -------------------------------------------------------------
def test_window_params_2s_50pct():
    w, s = window_params(fs=256, window_sec=2.0, overlap=0.5)
    assert w == 512 and s == 256


def test_count_windows_formula():
    # floor((N - W) / S) + 1
    assert count_windows(403200, 512, 256) == (403200 - 512) // 256 + 1
    assert count_windows(512, 512, 256) == 1
    assert count_windows(511, 512, 256) == 0


def test_window_starts_values():
    starts = window_starts(2000, 512, 256)
    assert starts[0] == 0 and starts[-1] == 1280
    assert np.all(np.diff(starts) == 256)


def test_window_labels_threshold():
    y = np.zeros(1024, dtype=np.uint8)
    y[0:300] = 1                              # 300/512 >= 0.5 -> fenêtre 0 = crise
    starts = window_starts(1024, 512, 256)    # [0, 256, 512]
    lab = window_labels_from_samples(y, starts, 512)
    assert lab[0] == 1                        # 300 positifs
    assert lab[1] == 0                        # 44 positifs (256..511 -> 44)
    assert lab[2] == 0


# --- Étiquetage par échantillon -------------------------------------------
def test_sample_labels_inclusive():
    sez = pd.DataFrame({"Debut_Sec": [1.0], "Fin_Sec": [2.0]})
    y = sample_labels(n_samples=10 * 256, fs=256, patient_seizures_df=sez)
    assert y[256] == 1 and y[512] == 1        # bornes incluses
    assert y[255] == 0 and y[513] == 0
    bal = class_balance(y)
    assert bal["n1"] == 257                   # indices 256..512 inclus


# --- Correspondance fichier / patient -------------------------------------
def test_patient_id_from_stem():
    assert patient_id_from_stem("190304A_E_0000d") == "190304A-E"
    assert patient_id_from_stem("230515B_G_0000d") == "230515B-G"


# --- Post-traitement -------------------------------------------------------
def test_merge_gaps_and_intervals():
    t = np.arange(10, dtype=float)
    b = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
    merged = merge_gaps(t, b, min_gap_s=2.0)   # comble le trou de 1 échantillon
    assert merged[3] == 1
    iv = binary_to_intervals(t, merged, min_dur_s=0.0)
    assert len(iv) == 1 and iv[0][0] == 1.0 and iv[0][1] == 7.0


def test_smooth_scores_median():
    x = np.array([0, 0, 1, 0, 0], dtype=float)
    assert smooth_scores(x, 3)[2] == 0.0       # pic isolé supprimé


# --- Appariement d'événements ---------------------------------------------
def test_match_events_tolerance():
    true = [(10.0, 20.0), (100.0, 110.0)]
    detected = [(10.5, 19.0), (200.0, 205.0)]  # 1 TP, 1 FP, 1 FN
    m = match_events(true, detected, tolerance_s=2.0)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
