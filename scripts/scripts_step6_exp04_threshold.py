"""Expérience EXP-04 — Calibration du Seuil par Validation Interne (strictement imbriquée).

Hypothèse testée (Hypothèse 4) :
L'utilisation d'une perte pondérée (pos_weight ~ 27) décale les scores vers le haut.
Un balayage du seuil sur l'ensemble OOF réuni introduit un biais de sélection.
La procédure correcte est :
  - Grille étendue de seuils : [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96].
  - Pour chaque fold k, sélectionner tau*_k sur les prédictions de validation
    interne du modèle entraîné sans le fold de test k.
  - Évaluer tau*_k sur le fold k (test externe jamais utilisé pour la sélection).
  - Calculer à la fois :
      1. Les métriques hors-ligne (mode rétrospectif standard).
      2. Les métriques causales temps réel (filtre rétrograde + alarme à t_start + 2s).
  - Agréger les métriques (moyenne +/- écart-type) et afficher les détails par fold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src import config
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
    """Sélectionne le seuil qui minimise FA/h parmi ceux qui atteignent min_det % de détection."""
    candidates = []
    for th in thresholds:
        th = round(float(th), 2)
        ev = _event_metrics_for_threshold(df, seizures, th, smooth_size, min_gap_s, min_dur_s, causal=True)
        if ev["taux_detection"] >= min_det:
            candidates.append((th, ev["fausses_alarmes_par_h"]))
    if candidates:
        return min(candidates, key=lambda x: x[1])[0]

    # Fallback : seuil qui maximise le F1 fenêtre
    best_f1, best_th = -1.0, 0.5
    y_true = df["label"].to_numpy()
    for th in thresholds:
        th = round(float(th), 2)
        preds = (df["score"].to_numpy() >= th).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    return best_th


def run_nested_calibration(
    df_test_all: pd.DataFrame,
    df_inner_val_all: pd.DataFrame,
    seizures: pd.DataFrame,
    thresholds: list[float],
    smooth_size: int = 5,
    min_gap_s: float = 2.0,
    min_dur_s: float = 2.0,
    min_det: float = 85.0,
) -> dict:
    """Calibration strictement imbriquée : seuil sur validation interne, test aveugle."""
    print("\n[NESTED] Calibration strictement imbriquée — seuil sur validation interne")
    print("=" * 85)

    if "fold" not in df_test_all.columns or "fold" not in df_inner_val_all.columns:
        raise ValueError("Les prédictions EXP-03 doivent contenir une colonne 'fold'.")

    fold_ids = sorted(df_test_all["fold"].unique().tolist())
    fold_results = []

    for fold_id in fold_ids:
        df_val = df_inner_val_all[df_inner_val_all["fold"] == fold_id]
        df_test = df_test_all[df_test_all["fold"] == fold_id]
        if df_val.empty or df_test.empty:
            raise ValueError(f"Fold {fold_id} absent des prédictions de validation ou de test.")
        test_patients = sorted(df_test["patient"].unique().tolist())

        tau_star = _best_threshold_on_df(df_val, seizures, thresholds, min_det, smooth_size, min_gap_s, min_dur_s)

        y_true = df_test["label"].to_numpy()
        y_bin = (df_test["score"].to_numpy() >= tau_star).astype(int)
        f1 = f1_score(y_true, y_bin, zero_division=0)
        prec = precision_score(y_true, y_bin, zero_division=0)
        rec = recall_score(y_true, y_bin, zero_division=0)

        ev_off = _event_metrics_for_threshold(df_test, seizures, tau_star, smooth_size, min_gap_s, min_dur_s, causal=False)
        ev_caus = _event_metrics_for_threshold(df_test, seizures, tau_star, smooth_size, min_gap_s, min_dur_s, causal=True)

        print(f"  Fold {fold_id} ({len(test_patients)} patients) | tau*={tau_star:.2f} | F1={f1:.3f} | "
              f"Détection={ev_off['taux_detection']:.1f}% | FA/h={ev_off['fausses_alarmes_par_h']:.2f} | "
              f"Lat. Hors-Ligne={ev_off['latence_moy_s']:.2f}s | Lat. Causale={ev_caus['latence_moy_s']:.2f}s")

        fold_results.append({
            "fold": int(fold_id),
            "patients": test_patients,
            "inner_val_patients": sorted(df_val["patient"].unique().tolist()),
            "tau_star": tau_star,
            "f1": round(f1, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "taux_detection": round(ev_off["taux_detection"], 1),
            "fausses_alarmes_par_h": round(ev_off["fausses_alarmes_par_h"], 2),
            "latence_hors_ligne_s": round(ev_off["latence_moy_s"], 2),
            "latence_causale_s": round(ev_caus["latence_moy_s"], 2),
            "causal_detection": round(ev_caus["taux_detection"], 1),
            "causal_fa_per_h": round(ev_caus["fausses_alarmes_par_h"], 2),
            "n_tp": int(ev_off["n_tp"]),
        })

    tau_mean = float(np.mean([r["tau_star"] for r in fold_results]))
    tau_std = float(np.std([r["tau_star"] for r in fold_results]))
    f1_mean = float(np.mean([r["f1"] for r in fold_results]))
    f1_std = float(np.std([r["f1"] for r in fold_results]))
    prec_mean = float(np.mean([r["precision"] for r in fold_results]))
    prec_std = float(np.std([r["precision"] for r in fold_results]))
    rec_mean = float(np.mean([r["recall"] for r in fold_results]))
    rec_std = float(np.std([r["recall"] for r in fold_results]))
    det_mean = float(np.mean([r["taux_detection"] for r in fold_results]))
    det_std = float(np.std([r["taux_detection"] for r in fold_results]))
    fa_mean = float(np.mean([r["fausses_alarmes_par_h"] for r in fold_results]))
    fa_std = float(np.std([r["fausses_alarmes_par_h"] for r in fold_results]))
    weights = [r["n_tp"] for r in fold_results]
    lat_off_mean = float(np.average([r["latence_hors_ligne_s"] for r in fold_results], weights=weights))
    lat_caus_mean = float(np.average([r["latence_causale_s"] for r in fold_results], weights=weights))

    print("-" * 85)
    print(f"  => Seuil moyen        : tau* = {tau_mean:.2f} +/- {tau_std:.2f}")
    print(f"  => F1 fenêtre         : {f1_mean:.3f} +/- {f1_std:.3f}")
    print(f"  => Précision fenêtre  : {prec_mean:.3f} +/- {prec_std:.3f}")
    print(f"  => Rappel fenêtre     : {rec_mean:.3f} +/- {rec_std:.3f}")
    print(f"  => Taux détection     : {det_mean:.1f} % +/- {det_std:.1f} % (min={min(r['taux_detection'] for r in fold_results)}%, max={max(r['taux_detection'] for r in fold_results)}%)")
    print(f"  => Fausses alarmes/h  : {fa_mean:.2f} +/- {fa_std:.2f} FA/h")
    print(f"  => Latence hors-ligne : {lat_off_mean:.2f} s  (artefact horodatage bord gauche)")
    print(f"  => Latence causale    : {lat_caus_mean:.2f} s  (temps réel : buffer 2s + confirmation)")

    return {
        "tau_mean": round(tau_mean, 2),
        "tau_std": round(tau_std, 2),
        "f1_mean": round(f1_mean, 3),
        "f1_std": round(f1_std, 3),
        "precision_mean": round(prec_mean, 3),
        "precision_std": round(prec_std, 3),
        "recall_mean": round(rec_mean, 3),
        "recall_std": round(rec_std, 3),
        "taux_detection_mean": round(det_mean, 1),
        "taux_detection_std": round(det_std, 1),
        "fausses_alarmes_par_h_mean": round(fa_mean, 2),
        "fausses_alarmes_par_h_std": round(fa_std, 2),
        "latence_hors_ligne_s_mean": round(lat_off_mean, 2),
        "latence_causale_s_mean": round(lat_caus_mean, 2),
        "folds": fold_results,
    }


def run_indicative_sweep(
    df_oof: pd.DataFrame,
    seizures: pd.DataFrame,
    thresholds: list[float],
    smooth_size: int = 5,
    min_gap_s: float = 2.0,
    min_dur_s: float = 2.0,
) -> list[dict]:
    """Balayage indicatif (OOF réuni — biais de sélection résiduel)."""
    print("\n[INDICATIF] Balayage étendu sur prédictions OOF réunies (exploration)")
    print("  /!\\ Ces métriques sont optimistes car calculées sur le pool global.")
    print(f"{'Seuil':>6} | {'F1':>6} | {'Préc':>6} | {'Rappel':>6} | {'Détection %':>11} | {'FA/h':>8} | {'Lat. Causale':>12}")
    print("-" * 75)

    results = []
    y_true = df_oof["label"].to_numpy()
    for th in thresholds:
        th = round(float(th), 2)
        y_bin = (df_oof["score"].to_numpy() >= th).astype(int)
        f1 = f1_score(y_true, y_bin, zero_division=0)
        prec = precision_score(y_true, y_bin, zero_division=0)
        rec = recall_score(y_true, y_bin, zero_division=0)
        ev = _event_metrics_for_threshold(df_oof, seizures, th, smooth_size, min_gap_s, min_dur_s, causal=False)
        ev_c = _event_metrics_for_threshold(df_oof, seizures, th, smooth_size, min_gap_s, min_dur_s, causal=True)
        print(f"  {th:>4.2f} | {f1:>6.3f} | {prec:>6.3f} | {rec:>6.3f} | "
              f"{ev['taux_detection']:>10.1f}% | {ev['fausses_alarmes_par_h']:>7.2f} | {ev_c['latence_moy_s']:>10.2f}s")
        results.append({
            "seuil": th,
            "f1": round(f1, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "taux_detection": round(ev["taux_detection"], 1),
            "fausses_alarmes_par_h": round(ev["fausses_alarmes_par_h"], 2),
            "latence_hors_ligne_s": round(ev["latence_moy_s"], 2),
            "latence_causale_s": round(ev_c["latence_moy_s"], 2),
        })
    return results


def run_exp04(smooth_size: int = 5, min_gap_s: float = 2.0, min_dur_s: float = 2.0, min_det: float = 85.0) -> dict:
    print("=" * 85)
    print("[EXP-04] Calibration Clinique du Seuil — Grille Étendue & Validation Interne")
    print("=" * 85)

    oof_path = config.CACHE_DIR / "experiments" / "EXP-03_oof_predictions.pkl"
    inner_val_path = config.CACHE_DIR / "experiments" / "EXP-03_inner_val_predictions.pkl"
    if not oof_path.exists() or not inner_val_path.exists():
        raise FileNotFoundError(
            f"Prédictions EXP-03 introuvables ({oof_path.name}, {inner_val_path.name}). "
            "Veuillez d'abord exécuter EXP-03."
        )

    print("\n[1/3] Chargement des prédictions de test et de validation interne d'EEGNet")
    df = pd.read_pickle(oof_path).reset_index(drop=True)
    df_inner_val = pd.read_pickle(inner_val_path).reset_index(drop=True)
    print(f"  Fenêtres test : {len(df)} | validation interne : {len(df_inner_val)}")

    seizures, _ = load_annotations()
    # Grille étendue pour éliminer tout effet de bord à 0.90
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]
    t_start = time.time()

    # 1. Balayage indicatif étendu
    indicative = run_indicative_sweep(df, seizures, thresholds, smooth_size, min_gap_s, min_dur_s)

    # 2. Calibration strictement imbriquée (validation interne, test aveugle)
    nested = run_nested_calibration(df, df_inner_val, seizures, thresholds, smooth_size, min_gap_s, min_dur_s, min_det=min_det)

    out_dir = config.CACHE_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experience": "EXP-04-THRESHOLD-CALIBRATION",
        "nested": nested,
        "indicative_sweep": indicative,
        "duree_totale_s": round(time.time() - t_start, 1),
    }
    with open(out_dir / "EXP-04_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nRésultats sauvegardés dans {out_dir / 'EXP-04_metrics.json'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--min-gap", type=float, default=2.0)
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--min-det", type=float, default=85.0)
    args = parser.parse_args()
    run_exp04(smooth_size=args.smooth, min_gap_s=args.min_gap, min_dur_s=args.min_dur, min_det=args.min_det)
