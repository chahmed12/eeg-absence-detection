"""Étape 4 — détection temporelle : post-traitement + métriques par événement.

Réentraîne RandomForest en LOPO pour obtenir, pour chaque patient, une
probabilité de crise par fenêtre. Les probabilités sont ensuite lissées,
seuillées, fusionnées, puis converties en intervalles de crises comparés aux
annotations (tolérance ± 2 s).
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, models, postprocess as pp, viz
from src.evaluation import event_metrics_by_patient, threshold_sweep
from src.io import load_annotations, list_mat_files, load_patient_signal, patient_id_from_stem
from src.labeling import patient_seizures

pd.set_option("display.width", 220)


def _true_events(seizures: pd.DataFrame, pid: str) -> list[tuple[float, float]]:
    """Liste (début, fin) des crises réelles d'un patient."""
    s = patient_seizures(seizures, pid)
    return list(map(tuple, s[["Debut_Sec", "Fin_Sec"]].to_numpy())) if len(s) else []


def build_lopo_predictions(force: bool = False) -> pd.DataFrame:
    """Calcule (ou recharge) les probabilités LOPO du RandomForest."""
    path = config.PREDICTIONS_PARQUET
    if path.exists() and not force:
        return pd.read_parquet(path)

    df = pd.read_parquet(config.FEATURES_PARQUET)
    X, _ = models.clean_feature_matrix(df)
    y = df["label"].to_numpy()
    seizures, _ = load_annotations()
    folds = models.lopo_folds(df, models.annotated_patients(seizures))
    model = models.get_models()["RandomForest"]

    parts = []
    for f in folds:
        m, score, pred = models.fit_evaluate(model, X, y, f.train_idx, f.test_idx)
        sub = df.iloc[f.test_idx][["patient", "t_start", "label"]].copy()
        sub["score"] = score
        sub["pred"] = pred
        parts.append(sub)
        print(f"  {f.name:22s} F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f}")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def main(force: bool = False) -> None:
    seizures, _ = load_annotations()
    viz.set_report_style()
    preds = build_lopo_predictions(force=force)
    print(f"\nPrédictions : {preds.shape} | patients : {preds.patient.nunique()}")

    threshold = 0.5
    # post-traitement patient par patient -> intervalles + scores lissés
    detected_by_patient: dict[str, list[tuple[float, float]]] = {}
    preds = preds.sort_values(["patient", "t_start"]).reset_index(drop=True)
    smooth = np.zeros(len(preds))
    for pid, g in preds.groupby("patient"):
        sm, intervals = pp.postprocess(g["t_start"].to_numpy(), g["score"].to_numpy(),
                                       threshold=threshold, smooth_size=5,
                                       min_gap_s=2.0, min_dur_s=2.0)
        smooth[g.index.to_numpy()] = sm
        detected_by_patient[pid] = intervals
    preds["score_smooth"] = smooth
    preds.to_parquet(config.PREDICTIONS_PARQUET, index=False)

    durations = {pid: float(g["t_start"].max()) + config.WINDOW_SEC
                 for pid, g in preds.groupby("patient")}
    ev = event_metrics_by_patient(seizures, detected_by_patient, durations)
    ev.to_parquet(config.CACHE_DIR / "event_metrics.parquet", index=False)

    print("\n=== MÉTRIQUES PAR ÉVÉNEMENT (LOPO, seuil=%.2f) ===" % threshold)
    print(ev.to_string(index=False))
    total_h = sum(durations.values()) / 3600.0
    agg = {
        "crises_reelles": int(ev.n_crises_reelles.sum()),
        "detections": int(ev.n_detections.sum()),
        "TP": int(ev.tp.sum()), "FP": int(ev.fp.sum()), "FN": int(ev.fn.sum()),
        "taux_detection_global": float(ev.tp.sum() / ev.n_crises_reelles.sum()),
        "fausses_alarmes_par_h": float(ev.fp.sum() / total_h),
        "latence_moy_s": float(np.nanmean(ev.latence_moy_s)),
        "err_debut_moy_s": float(np.nanmean(ev.err_debut_moy_s)),
        "err_fin_moy_s": float(np.nanmean(ev.err_fin_moy_s)),
    }
    print("\nGlobal :", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()})

    def patient_intervals(pid: str) -> list[tuple[float, float]]:
        return detected_by_patient.get(pid, [])

    # --- figures --------------------------------------------------------
    demo = "210914B-A"
    gdemo = preds[preds.patient == demo].sort_values("t_start")
    fig = viz.plot_probability(gdemo["t_start"].to_numpy(), gdemo["score_smooth"].to_numpy(),
                               threshold, true_events=_true_events(seizures, demo),
                               title=f"Patient {demo} — probabilité de crise (LOPO)")
    viz.save_fig(fig, "04_probability_210914B-A")
    plt.close(fig)

    for pid in [demo, "230102B-F"]:
        p = next((q for q in list_mat_files() if patient_id_from_stem(q.stem) == pid), None)
        if p is None:
            continue
        rec = load_patient_signal(p)
        t = np.arange(rec.sig.shape[0]) / rec.fs
        fig = viz.plot_detection(t, rec.sig, rec.names, "EEGFp1",
                                 _true_events(seizures, pid), patient_intervals(pid),
                                 title=f"Patient {pid} — détection (seuil {threshold})")
        viz.save_fig(fig, f"04_detection_{pid}")
        plt.close(fig)

    # --- courbe métrique vs seuil --------------------------------------
    sweep = threshold_sweep(preds["label"].to_numpy(), preds["score"].to_numpy())
    sweep.to_parquet(config.CACHE_DIR / "threshold_sweep.parquet", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(viz._fw(14), viz._fh(4)))
    ax[0].plot(sweep.threshold, sweep.f1, label="F1")
    ax[0].plot(sweep.threshold, sweep.precision, label="Précision")
    ax[0].plot(sweep.threshold, sweep.recall, label="Rappel")
    ax[0].set_xlabel("Seuil"); ax[0].legend(); ax[0].grid(alpha=.4)
    ax[1].plot(sweep.recall, sweep.precision)
    ax[1].set_xlabel("Rappel"); ax[1].set_ylabel("Précision")
    ax[1].set_title("Courbe Precision-Recall"); ax[1].grid(alpha=.4)
    viz.save_fig(fig, "04_metric_vs_seuil")
    plt.close(fig)
    print("\nFigures : report/figures/04_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recalculer les prédictions LOPO")
    main(force=ap.parse_args().force)
