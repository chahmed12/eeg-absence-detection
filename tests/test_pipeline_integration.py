"""Test d'intégration bout-en-bout (pipeline complet).

Vérifie l'enchaînement de toutes les étapes du pipeline sans nécessiter
les fichiers .mat réels :
1. Génération d'un signal EEG synthétique avec des crises simulées (décharges ~3 Hz).
2. Pré-traitement (bandpass + notch).
3. Fenêtrage et extraction de caractéristiques (temporelles, spectrales, Hjorth, synchronie).
4. Étiquetage des fenêtres.
5. Entraînement et prédiction par un classifieur (RandomForest / DecisionTree).
6. Post-traitement temporel (lissage médian, fusion des trous, filtrage par durée).
7. Évaluation par événement (TP, FP, FN, latence, métriques cliniques).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocess import preprocess_signal
from src.windowing import window_params, window_starts, window_labels_from_samples
from src.labeling import sample_labels
from src.features import extract_features
from src.models import clean_feature_matrix, get_models, fit_evaluate, random_stratified_split
from src.postprocess import postprocess
from src.evaluation import match_events, event_metrics, select_threshold


def test_full_pipeline_end_to_end():
    fs = 256.0
    duration_sec = 60.0
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs

    # 1. Génération signal multicanal : bruit de fond + 1 crise simulée (15s à 25s) à 3 Hz
    rng = np.random.default_rng(42)
    sig = rng.standard_normal((n_samples, 3)).astype(np.float32) * 5.0

    # Injection d'une oscillation forte 3 Hz entre 15s et 25s
    mask_seizure = (t >= 15.0) & (t <= 25.0)
    for c in range(3):
        sig[mask_seizure, c] += 50.0 * np.sin(2 * np.pi * 3.0 * t[mask_seizure]).astype(np.float32)

    # Crise réelle annotée
    true_events = [(15.0, 25.0)]
    seizures_df = pd.DataFrame([{"Debut_Sec": 15.0, "Fin_Sec": 25.0}])

    # 2. Pré-traitement
    sig_clean = preprocess_signal(sig, fs=fs)
    assert sig_clean.shape == sig.shape

    # 3. Fenêtrage & extraction de caractéristiques
    win_len, step = window_params(fs, window_sec=2.0, overlap=0.5)
    chan_names = ["EEGFp1", "EEGFp2", "EEGCz"]
    df_feats, starts = extract_features(
        sig_clean, fs=fs, names=chan_names, win_len=win_len, step=step,
        include_perm=False, include_wavelet=False,
    )
    assert len(df_feats) > 0
    assert "t_start" in df_feats.columns

    # 4. Étiquetage
    y_samples = sample_labels(n_samples, fs, seizures_df)
    y_windows = window_labels_from_samples(y_samples, starts, win_len)
    df_feats["label"] = y_windows
    df_feats["patient"] = "P_SYNTH"

    # Vérification que la crise a bien généré des fenêtres positives
    assert (df_feats["label"] == 1).sum() > 0

    # 5. Modélisation (RandomForest)
    X, feat_cols = clean_feature_matrix(df_feats)
    y = df_feats["label"].to_numpy()
    model = get_models(seed=42)["RandomForest"]

    # Split train/test (moitié-moitié chronologique pour ce test unitaire synthétique)
    split_idx = len(df_feats) // 2
    train_idx = np.arange(split_idx)
    test_idx = np.arange(split_idx, len(df_feats))

    metrics, scores, preds = fit_evaluate(model, X, y, train_idx, test_idx)
    assert "accuracy" in metrics
    assert len(scores) == len(test_idx)

    # 6. Post-traitement temporel
    t_test = df_feats.iloc[test_idx]["t_start"].to_numpy()
    # Recherche du seuil optimal sur le train pour le postprocess
    model_fitted = model.fit(X.iloc[train_idx], y[train_idx])
    train_scores = model_fitted.predict_proba(X.iloc[train_idx])[:, 1]
    best_thresh, _ = select_threshold(y[train_idx], train_scores)

    pred_smooth, detected_events = postprocess(
        t_test, scores, threshold=best_thresh, smooth_size=3, min_gap_s=1.0, min_dur_s=1.0
    )

    # 7. Évaluation par événement
    ev_res = match_events(true_events, detected_events, tolerance_s=3.0)
    assert isinstance(ev_res["tp"], int)
    assert isinstance(ev_res["fp"], int)
    assert isinstance(ev_res["fn"], int)

    ev_metrics = event_metrics(true_events, detected_events, duration_s=duration_sec, tolerance_s=3.0)
    assert "taux_detection" in ev_metrics
    assert "fausses_alarmes_par_h" in ev_metrics
