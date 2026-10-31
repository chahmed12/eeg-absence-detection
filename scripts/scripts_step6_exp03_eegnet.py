"""Expérience EXP-03 — Évaluation de l'architecture EEGNet (Spatio-Temporel Complet).

Hypothèse testée (Hypothèse 3) :
L'architecture standard EEGNet (Lawhern et al., 2018) sépare explicitement le filtrage
temporel (oscillations pointe-onde) et spatial (scalp), puis applique des convolutions
séparables en profondeur avec pooling régulier. Cette structure permet-elle d'obtenir
un meilleur compromis sensibilité / fausses alarmes et de surpasser la baseline ?

Protocole :
- Entrée brute : (B, 1, 8 canaux communs, 512 pas de temps à 256 Hz).
- Normalisation z-score calculée exclusivement sur le train de chaque fold (zéro fuite).
- Validation GroupKFold 5 patients (identique à EXP-00, EXP-01, EXP-02).
- Sauvegarde des prédictions hors-fold pour l'analyse de calibration (EXP-04).
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
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src import config
from src.deep_learning import (
    EEGNet,
    build_raw_windows_dataset,
    train_fold_nested,
)
from src.evaluation import event_metrics_pooled
from src.io import load_annotations
from src.models import group_kfold_splits
from src.postprocess import postprocess


def run_exp03(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.25,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    print("=" * 70)
    print(f"[EXP-03] EEGNet Complet (Spatio-temporel séparable, dropout={dropout})")
    print("=" * 70)

    # 1. Chargement des fenêtres EEG brutes
    print("\n[1/5] Chargement du cache des fenêtres EEG brutes...")
    raw = build_raw_windows_dataset()
    print(f"  Tenseurs bruts : {raw.X.shape} (N={len(raw.X)}, 8 canaux, 512 pas)")
    print(f"  Taux de crise : {100 * raw.y.mean():.2f} % ({int(raw.y.sum())} fenêtres positives)")

    # 2. Folds GroupKFold par patient (zéro fuite)
    meta_df = pd.DataFrame({"patient": raw.patient, "label": raw.y, "t_start": raw.t_start})
    folds = group_kfold_splits(meta_df, n_splits=5, seed=seed)
    print(f"\n[2/5] Découpage en {len(folds)} GroupKFold par patient...")

    fold_results = []
    all_test_preds = []
    all_inner_val_preds = []
    seizures, _ = load_annotations()

    t_start_total = time.time()

    # 3. Boucle sur les folds : validation interne par patient, test externe aveugle
    for k, f in enumerate(folds):
        print(f"\n--- Fold {k + 1}/{len(folds)} : Test sur {f.test_patients} ---")
        tr_idx, te_idx = f.train_idx, f.test_idx

        def make_model():
            return EEGNet(
                n_channels=8,
                n_samples=512,
                F1=8,
                D=2,
                F2=16,
                kernel_len=64,
                dropout=dropout,
            )

        res = train_fold_nested(
            make_model,
            raw.X,
            raw.y,
            meta_df,
            tr_idx,
            te_idx,
            val_fraction=0.25,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + 100 + k,
            device=device,
        )
        y_te = res["test_y"]
        y_prob = res["test_probs"]
        preds_bin = (y_prob >= 0.5).astype(int)
        f1 = f1_score(y_te, preds_bin, zero_division=0)
        prec = precision_score(y_te, preds_bin, zero_division=0)
        rec = recall_score(y_te, preds_bin, zero_division=0)
        pr_auc = average_precision_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else np.nan

        print(f"  Fold {k + 1} -> best_epoch={res['best_epoch']} | F1={f1:.3f} | "
              f"Recall={rec:.3f} | Precision={prec:.3f} | PR-AUC={pr_auc:.3f}")
        fold_results.append({
            "fold": k + 1,
            "best_epoch": res["best_epoch"],
            "f1": f1,
            "recall": rec,
            "precision": prec,
            "pr_auc": pr_auc,
        })

        sub = meta_df.iloc[te_idx].copy()
        sub["score"] = y_prob
        sub["pred"] = preds_bin
        sub["fold"] = k + 1
        all_test_preds.append(sub)

        val_sub = meta_df.iloc[res["inner_val_idx"]].copy()
        val_sub["score"] = res["inner_val_probs"]
        val_sub["pred"] = (res["inner_val_probs"] >= 0.5).astype(int)
        val_sub["fold"] = k + 1
        all_inner_val_preds.append(val_sub)

    # 4. Post-traitement temporel et métriques événementielles cliniques
    print("\n[4/5] Post-traitement temporel (seuil=0.5, lissage médian, fusion)...")
    preds_df = pd.concat(all_test_preds, ignore_index=True)
    detected_by_patient = {}
    durations = {}

    for pid, g in preds_df.groupby("patient"):
        g = g.sort_values("t_start")
        _, intervals = postprocess(
            g["t_start"].to_numpy(),
            g["score"].to_numpy(),
            threshold=0.5,
            smooth_size=5,
            min_gap_s=2.0,
            min_dur_s=2.0,
            causal=True,
            window_sec=config.WINDOW_SEC,
        )
        detected_by_patient[pid] = intervals
        durations[pid] = float(g["t_start"].max()) + config.WINDOW_SEC

    ev = event_metrics_pooled(seizures, detected_by_patient, durations)

    # 5. Synthèse comparative
    res_f1 = np.mean([r["f1"] for r in fold_results])
    res_f1_std = np.std([r["f1"] for r in fold_results])
    res_rec = np.mean([r["recall"] for r in fold_results])
    res_prec = np.mean([r["precision"] for r in fold_results])
    res_pr_auc = np.mean([r["pr_auc"] for r in fold_results])

    detection_rate = float(ev["taux_detection"])
    fa_per_h = float(ev["fausses_alarmes_par_h"])
    avg_latency = float(ev["latence_moy_s"])

    summary = {
        "experience": "EXP-03-EEGNET",
        "f1_mean": round(res_f1, 3),
        "f1_std": round(res_f1_std, 3),
        "recall_mean": round(res_rec, 3),
        "precision_mean": round(res_prec, 3),
        "pr_auc_mean": round(res_pr_auc, 3),
        "taux_detection": round(detection_rate * 100, 1),
        "fausses_alarmes_par_h": round(fa_per_h, 2),
        "latence_moy_s": round(avg_latency, 2),
        "latence_std_s": round(float(ev["latence_std_s"]), 2),
        "n_tp": int(ev["tp"]),
        "n_fp": int(ev["fp"]),
        "n_fn": int(ev["fn"]),
        "duree_totale_h": round(float(ev["duree_totale_h"]), 2),
        "duree_totale_s": round(time.time() - t_start_total, 1),
        "dropout": dropout,
    }

    print("\n" + "=" * 70)
    print("RESULTATS COMPARATIFS : EXP-02 vs EXP-03 (EEGNet Complet)")
    print("=" * 70)
    print(f"F1 Fenêtre (GroupKFold) : {summary['f1_mean']} ± {summary['f1_std']}  (EXP-02: 0.553 ± 0.088 | Baseline: 0.689)")
    print(f"Précision Fenêtre       : {summary['precision_mean']}  (EXP-02: 0.427)")
    print(f"Rappel Fenêtre          : {summary['recall_mean']}  (EXP-02: 0.849)")
    print(f"PR-AUC                  : {summary['pr_auc_mean']}  (EXP-02: 0.638 | Baseline: 0.704)")
    print(f"Taux de Détection (Év.) : {summary['taux_detection']} %  (EXP-02: 93.9 %)")
    print(f"Latence Moyenne (Év.)   : {summary['latence_moy_s']} s  (EXP-02: -0.16 s)")
    print(f"Fausses Alarmes / heure : {summary['fausses_alarmes_par_h']} FA/h  (EXP-02: 15.6 FA/h)")
    print("=" * 70)

    # Sauvegarde des métriques et des prédictions hors-fold pour EXP-04
    out_dir = config.CACHE_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "EXP-03_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Sauvegarde des prédictions de test et de validation interne (calibration EXP-04)
    preds_df.to_pickle(out_dir / "EXP-03_oof_predictions.pkl")
    inner_val_df = pd.concat(all_inner_val_preds, ignore_index=True)
    inner_val_df.to_pickle(out_dir / "EXP-03_inner_val_predictions.pkl")
    print(f"Prédictions de test sauvegardées dans {out_dir / 'EXP-03_oof_predictions.pkl'}")
    print(f"Validation interne sauvegardée dans {out_dir / 'EXP-03_inner_val_predictions.pkl'}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques par fold")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()
    run_exp03(epochs=args.epochs, batch_size=args.batch_size, dropout=args.dropout)
