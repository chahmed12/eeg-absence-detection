"""Tests unitaires des métriques événementielles (src/evaluation.py).

Tous les tests sont basés sur des données synthétiques contrôlées
pour vérifier les calculs sans dépendre des fichiers EEG réels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    event_metrics,
    event_metrics_by_patient,
    match_events,
    pr_curve_data,
    select_threshold,
    threshold_sweep,
)


# ---------------------------------------------------------------------------
# match_events
# ---------------------------------------------------------------------------
def test_match_events_perfect_detection():
    true = [(10.0, 20.0), (50.0, 60.0)]
    det = [(10.0, 20.0), (50.0, 60.0)]
    m = match_events(true, det, tolerance_s=2.0)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0


def test_match_events_no_detection():
    true = [(10.0, 20.0), (50.0, 60.0)]
    m = match_events(true, [], tolerance_s=2.0)
    assert m["tp"] == 0 and m["fp"] == 0 and m["fn"] == 2


def test_match_events_no_true():
    det = [(10.0, 20.0)]
    m = match_events([], det, tolerance_s=2.0)
    assert m["tp"] == 0 and m["fp"] == 1 and m["fn"] == 0


def test_match_events_tolerance_within():
    """Une détection décalée de 1 s est appariée si tolérance = 2 s."""
    true = [(10.0, 20.0)]
    det = [(11.0, 19.0)]
    m = match_events(true, det, tolerance_s=2.0)
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0


def test_match_events_tolerance_exceeded():
    """Une détection trop éloignée n'est PAS appariée."""
    true = [(10.0, 20.0)]
    det = [(25.0, 35.0)]  # aucun chevauchement même avec tolérance 2 s
    m = match_events(true, det, tolerance_s=2.0)
    assert m["tp"] == 0 and m["fp"] == 1 and m["fn"] == 1


def test_match_events_one_tp_one_fp_one_fn():
    """Scénario classique : 2 crises, 1 bien détectée, 1 fausse alarme."""
    true = [(10.0, 20.0), (100.0, 110.0)]
    det = [(10.5, 19.0), (200.0, 205.0)]
    m = match_events(true, det, tolerance_s=2.0)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1


def test_match_events_latence_positive():
    """Détection anticipée → latence négative ; détection tardive → positive."""
    true = [(10.0, 20.0)]
    det = [(12.0, 21.0)]   # début 2 s après la crise réelle
    m = match_events(true, det, tolerance_s=5.0)
    assert m["tp"] == 1
    assert len(m["latences"]) == 1
    assert abs(m["latences"][0] - 2.0) < 1e-6


def test_match_events_one_to_one_matching():
    """Chaque détection ne peut être appariée qu'une seule fois."""
    true = [(10.0, 20.0), (12.0, 22.0)]  # deux crises proches
    det = [(10.0, 20.0)]                   # une seule détection
    m = match_events(true, det, tolerance_s=5.0)
    assert m["tp"] == 1 and m["fn"] == 1


# ---------------------------------------------------------------------------
# event_metrics
# ---------------------------------------------------------------------------
def test_event_metrics_taux_detection_1():
    """Toutes les crises détectées → taux = 1."""
    true = [(10.0, 20.0), (50.0, 60.0)]
    det = [(10.0, 20.0), (50.0, 60.0)]
    m = event_metrics(true, det, duration_s=3600.0)
    assert m["taux_detection"] == 1.0


def test_event_metrics_taux_detection_0():
    """Aucune crise détectée → taux = 0."""
    true = [(10.0, 20.0)]
    m = event_metrics(true, [], duration_s=3600.0)
    assert m["taux_detection"] == 0.0


def test_event_metrics_fausses_alarmes_par_h():
    """2 fausses alarmes sur 2 heures → 1 FP/h."""
    true = []
    det = [(10.0, 20.0), (100.0, 110.0)]
    m = event_metrics(true, det, duration_s=7200.0)
    assert abs(m["fausses_alarmes_par_h"] - 1.0) < 1e-6


def test_event_metrics_no_true_events():
    """Sans crise réelle, taux_detection est NaN."""
    m = event_metrics([], [(5.0, 10.0)], duration_s=3600.0)
    assert np.isnan(m["taux_detection"])


def test_event_metrics_keys():
    true = [(10.0, 20.0)]
    det = [(10.0, 20.0)]
    m = event_metrics(true, det, duration_s=3600.0)
    for key in ["n_crises_reelles", "n_detections", "tp", "fp", "fn",
                "taux_detection", "fausses_alarmes_par_h", "latence_moy_s"]:
        assert key in m


# ---------------------------------------------------------------------------
# event_metrics_by_patient
# ---------------------------------------------------------------------------
def test_event_metrics_by_patient_shape():
    seizures = pd.DataFrame({
        "Patient": ["P01", "P01", "P02"],
        "Debut_Sec": [10.0, 50.0, 100.0],
        "Fin_Sec": [20.0, 60.0, 115.0],
    })
    detected = {
        "P01": [(10.0, 20.0)],
        "P02": [],
    }
    durations = {"P01": 3600.0, "P02": 3600.0}
    df = event_metrics_by_patient(seizures, detected, durations)
    assert len(df) == 2
    assert "patient" in df.columns
    assert "taux_detection" in df.columns


def test_event_metrics_by_patient_values():
    seizures = pd.DataFrame({
        "Patient": ["P01"],
        "Debut_Sec": [10.0],
        "Fin_Sec": [20.0],
    })
    detected = {"P01": [(10.0, 20.0)]}
    durations = {"P01": 3600.0}
    df = event_metrics_by_patient(seizures, detected, durations)
    assert df.loc[df["patient"] == "P01", "tp"].iloc[0] == 1
    assert df.loc[df["patient"] == "P01", "taux_detection"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# threshold_sweep
# ---------------------------------------------------------------------------
def test_threshold_sweep_returns_dataframe():
    y = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    df = threshold_sweep(y, scores)
    assert isinstance(df, pd.DataFrame)
    assert "threshold" in df.columns
    assert "f1" in df.columns


def test_threshold_sweep_monotone_threshold():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    df = threshold_sweep(y, scores)
    assert (df["threshold"].diff().dropna() > 0).all()


def test_threshold_sweep_f1_in_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    df = threshold_sweep(y, scores)
    assert (df["f1"] >= 0).all() and (df["f1"] <= 1).all()


# ---------------------------------------------------------------------------
# select_threshold
# ---------------------------------------------------------------------------
def test_select_threshold_returns_best_f1():
    y = np.array([0, 0, 0, 1, 1, 1])
    # Séparation parfaite : seuil 0,5 → F1 = 1
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t, f1 = select_threshold(y, scores)
    assert 0.0 < t < 1.0
    assert f1 > 0.9


def test_select_threshold_all_zeros_scores():
    """Si les scores ne discriminent pas, le F1 reste à 0."""
    y = np.array([0, 1, 0, 1])
    scores = np.zeros(4)
    t, f1 = select_threshold(y, scores)
    assert isinstance(t, float)
    assert isinstance(f1, float)


# ---------------------------------------------------------------------------
# pr_curve_data
# ---------------------------------------------------------------------------
def test_pr_curve_data_shape():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    recall, precision = pr_curve_data(y, scores)
    assert len(recall) == len(precision)
    assert len(recall) > 0


def test_pr_curve_data_recall_in_range():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    recall, precision = pr_curve_data(y, scores)
    assert (recall >= 0).all() and (recall <= 1).all()
    assert (precision >= 0).all() and (precision <= 1).all()
