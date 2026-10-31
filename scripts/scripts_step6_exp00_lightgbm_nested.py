"""Expérience EXP-00 (P0) — Baseline LightGBM sous validation imbriquée stricte.

Protocole d'évaluation :
- 8 canaux EEG communs (260 features calculées par fenêtre de 2 s).
- Exactement les mêmes 5 plis de GroupKFold par patient que pour EEGNet (EXP-01 à EXP-04).
- Sélection strictement imbriquée du seuil tau* :
  * Pour chaque pli externe k, les patients du train externe sont séparés en
    train interne / validation interne (25 % des patients annotés).
  * Le seuil tau*_k est choisi sur les prédictions de validation interne d'un
    modèle entraîné uniquement sur le train interne.
  * Un modèle final est réentraîné sur tout le train externe, puis tau*_k est
    appliqué au pli de test k, jamais utilisé pour la sélection.
- Sauvegarde des prédictions de test et de validation interne dans
  cache/experiments/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

from src import config, models
from src.evaluation import event_metrics_pooled
from src.io import load_annotations
from src.postprocess import postprocess


def _event_metrics_for_threshold(
    df: pd.DataFrame,
    seizures: pd.DataFrame,
    threshold: float,
    smooth_size: int = 5,
    min_gap_s: float = 2.0,
    min_dur_s: float = 2.0,
    causal: bool = False,
) -> dict:
    """Calcule détection, FA/h et latence pour un seuil donné."""
    durations = {}
    detected_by_patient = {}
    for pid, g in df.groupby("patient"):
        g = g.sort_values("t_start")
        _, intervals = postprocess(
            g["t_start"].to_numpy(),
            g["score"].to_numpy(),
            threshold=threshold,
            smooth_size=smooth_size,
            min_gap_s=min_gap_s,
            min_dur_s=min_dur_s,
            causal=causal,
            window_sec=config.WINDOW_SEC,
        )
        detected_by_patient[pid] = intervals
        durations[pid] = float(g["t_start"].max()) + config.WINDOW_SEC

    ev = event_metrics_pooled(seizures, detected_by_patient, durations)
    return {
        "taux_detection": float(ev["taux_detection"]) * 100.0,
        "fausses_alarmes_par_h": float(ev["fausses_alarmes_par_h"]),
        "latence_moy_s": float(ev["latence_moy_s"]),
        "latence_std_s": float(ev["latence_std_s"]),
        "n_tp": int(ev["tp"]),
        "n_fp": int(ev["fp"]),
        "n_fn": int(ev["fn"]),
        "n_detections": int(ev["n_detections"]),
        "duree_totale_h": float(ev["duree_totale_h"]),
    }


def _best_threshold_on_df(
    df: pd.DataFrame,
    seizures: pd.DataFrame,
    thresholds: list[float],
    min_det: float = 85.0,
    smooth_size: int = 5,
    min_gap_s: float = 2.0,
    min_dur_s: float = 2.0,
) -> float:
    """Sélectionne le seuil qui minimise FA/h parmi ceux atteignant min_det % de détection."""
    candidates = []
    for th in thresholds:
        th = round(float(th), 2)
        ev = _event_metrics_for_threshold(df, seizures, th, smooth_size, min_gap_s, min_dur_s, causal=True)
        if ev["taux_detection"] >= min_det:
            candidates.append((th, ev["fausses_alarmes_par_h"]))
    if candidates:
        return min(candidates, key=lambda x: x[1])[0]

    # Fallback : seuil qui maximise le F1 fenêtre
    best_f1, best_th = -1.0, 0.20
    y_true = df["label"].to_numpy()
    for th in thresholds:
        th = round(float(th), 2)
        preds = (df["score"].to_numpy() >= th).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    return best_th


def main():
    print("=" * 80)
    print("[EXP-00 REVISEE] LightGBM — Validation Imbriquée Stricte (Zéro Fuite)")
    print("=" * 80)

    seizures, _ = load_annotations()
    df = pd.read_parquet(config.FEATURES_PARQUET)
    X_all, cols = models.clean_feature_matrix(df)
    y = df["label"].to_numpy()

    common_8 = set(config.COMMON_EEG_CHANNELS)
    sub_cols = [c for c in cols if c.startswith("global__") or c.split("__")[0] in common_8]
    print(f"Features retenues : {len(sub_cols)} features (8 canaux EEG communs)")
    X = X_all[sub_cols]

    # Mêmes 5 plis GroupKFold
    folds = models.group_kfold_splits(df, n_splits=5, seed=config.RANDOM_SEED)

    oof_preds = np.zeros(len(df), dtype=np.float32)
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    fold_results = []
    fold_taus = []
    inner_val_frames = []

    print("\n[1/3] Sélection imbriquée du seuil et inférence test sur les 5 plis...")
    for k, f in enumerate(folds):
        t0 = time.time()

        # Validation interne par patient, disjointe du train interne et du test externe
        inner_train_idx, inner_val_idx = models.patient_validation_split(
            df, f.train_idx, val_fraction=0.25, seed=config.RANDOM_SEED + 100 + k
        )
        clf_inner = LGBMClassifier(
            random_state=config.RANDOM_SEED,
            n_estimators=100,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1,
        )
        clf_inner.fit(X.iloc[inner_train_idx], y[inner_train_idx])
        inner_scores = clf_inner.predict_proba(X.iloc[inner_val_idx])[:, 1]

        df_val = pd.DataFrame({
            "patient": df.iloc[inner_val_idx]["patient"].to_numpy(),
            "label": y[inner_val_idx],
            "t_start": df.iloc[inner_val_idx]["t_start"].to_numpy(),
            "score": inner_scores,
        })
        tau_star = _best_threshold_on_df(df_val, seizures, thresholds, min_det=85.0)
        fold_taus.append(tau_star)

        # Modèle final sur tout le train externe, test externe jamais vu
        clf = LGBMClassifier(
            random_state=config.RANDOM_SEED,
            n_estimators=100,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1,
        )
        clf.fit(X.iloc[f.train_idx], y[f.train_idx])
        probs = clf.predict_proba(X.iloc[f.test_idx])[:, 1]
        oof_preds[f.test_idx] = probs

        inner_val_frames.append(df_val.assign(fold=k + 1))
        print(f"  Pli {k + 1}/5 ({len(f.test_patients)} patients) | "
              f"val interne={len(df_val['patient'].unique())} patients | "
              f"tau*={tau_star:.2f} | {time.time() - t0:.1f}s")

    # Sauvegarde des prédictions OOF et de validation interne
    out_dir = config.CACHE_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_oof = pd.DataFrame({
        "patient": df["patient"],
        "label": y,
        "t_start": df["t_start"],
        "score": oof_preds,
        "pred": (oof_preds >= 0.5).astype(int),
    })
    oof_path = out_dir / "predictions_lightgbm_oof.parquet"
    df_oof.to_parquet(oof_path, index=False)
    df_inner_val = pd.concat(inner_val_frames, ignore_index=True)
    inner_val_path = out_dir / "predictions_lightgbm_inner_val.parquet"
    df_inner_val.to_parquet(inner_val_path, index=False)
    print(f"\n[2/3] Prédictions OOF : {oof_path}")
    print(f"      Validation interne : {inner_val_path}")

    print(f"\n[3/3] Évaluation aveugle des seuils tau* sur les plis de test...")
    for k, f in enumerate(folds):
        df_test = df_oof.iloc[f.test_idx]
        tau_star = fold_taus[k]

        y_true = df_test["label"].to_numpy()
        y_bin = (df_test["score"].to_numpy() >= tau_star).astype(int)
        f1 = f1_score(y_true, y_bin, zero_division=0)
        prec = precision_score(y_true, y_bin, zero_division=0)
        rec = recall_score(y_true, y_bin, zero_division=0)
        prauc = average_precision_score(y_true, df_test["score"].to_numpy())

        ev_off = _event_metrics_for_threshold(df_test, seizures, tau_star, causal=False)
        ev_caus = _event_metrics_for_threshold(df_test, seizures, tau_star, causal=True)

        print(f"  Fold {k + 1} ({len(f.test_patients)} patients) | tau*={tau_star:.2f} | F1={f1:.3f} | "
              f"Préc={prec:.3f} | Rappel={rec:.3f} | Détection={ev_off['taux_detection']:.1f}% | "
              f"FA/h={ev_off['fausses_alarmes_par_h']:.2f} | Lat. Causale={ev_caus['latence_moy_s']:.2f}s")

        fold_results.append({
            "fold": k + 1,
            "test_patients": f.test_patients,
            "tau_star": tau_star,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "pr_auc": prauc,
            "taux_detection": ev_off["taux_detection"],
            "fa_per_h": ev_off["fausses_alarmes_par_h"],
            "latence_hors_ligne": ev_off["latence_moy_s"],
            "latence_causale": ev_caus["latence_moy_s"],
            "n_tp": ev_off["n_tp"],
            "n_fp": ev_off["n_fp"],
            "n_fn": ev_off["n_fn"],
        })

    res_df = pd.DataFrame(fold_results)
    res_path = out_dir / "EXP-00_nested_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("SYNTHÈSE LIGHTGBM SOUS VALIDATION IMBRIQUÉE (Moyenne ± Écart-type)")
    print("=" * 80)
    print(f"  Seuil moyen tau*        : {res_df['tau_star'].mean():.2f} ± {res_df['tau_star'].std():.2f}")
    print(f"  F1 fenêtre               : {res_df['f1'].mean():.3f} ± {res_df['f1'].std():.3f}")
    print(f"  Précision fenêtre        : {res_df['precision'].mean():.3f} ± {res_df['precision'].std():.3f}")
    print(f"  Rappel fenêtre           : {res_df['recall'].mean():.3f} ± {res_df['recall'].std():.3f}")
    print(f"  PR-AUC                   : {res_df['pr_auc'].mean():.3f} ± {res_df['pr_auc'].std():.3f}")
    print(f"  Taux Détection Crises    : {res_df['taux_detection'].mean():.1f}% ± {res_df['taux_detection'].std():.1f}%")
    print(f"  Fausses Alarmes / h      : {res_df['fa_per_h'].mean():.2f} ± {res_df['fa_per_h'].std():.2f} FA/h")
    print(f"  Latence causale estimée  : {res_df['latence_causale'].mean():.2f} ± {res_df['latence_causale'].std():.2f} s")
    print(f"  Latence hors-ligne       : {res_df['latence_hors_ligne'].mean():.2f} ± {res_df['latence_hors_ligne'].std():.2f} s")
    lat_caus_ew = float(np.average(res_df["latence_causale"], weights=res_df["n_tp"]))
    lat_off_ew = float(np.average(res_df["latence_hors_ligne"], weights=res_df["n_tp"]))
    print(f"  Latence causale (ev.)    : {lat_caus_ew:.2f} s (ponderee par TP)")
    print(f"  Latence hors-ligne (ev.) : {lat_off_ew:.2f} s (ponderee par TP)")
    print("=" * 80)


if __name__ == "__main__":
    main()
