import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src import config
from src.evaluation import event_metrics_pooled
from src.io import load_annotations
from src.postprocess import postprocess

EXP = ROOT / "cache" / "experiments"
TAU_LGB = [0.30, 0.50, 0.70, 0.90, 0.75]
TAU_EEG = [0.80, 0.50, 0.75, 0.96, 0.85]

seizures, _ = load_annotations()
annotated = set(seizures["Patient"].astype(str))

eeg = pd.read_pickle(EXP / "EXP-03_oof_predictions.pkl")
lgb = pd.read_parquet(EXP / "predictions_lightgbm_oof.parquet")
fold_of = eeg.groupby("patient")["fold"].first().to_dict()
assert set(lgb["patient"]) == set(fold_of), "Patients differents entre LightGBM et EEGNet"
lgb["fold"] = lgb["patient"].map(fold_of)


def evaluate(name, df, taus):
    df = df.copy()
    folds = sorted(df["fold"].unique())
    assert len(folds) == len(taus)
    df["tau"] = df["fold"].map(dict(zip(folds, taus)))
    df["pred"] = (df["score"] >= df["tau"]).astype(int)

    detected, durations, per_patient = {}, {}, []
    for pid, g in df.groupby("patient"):
        g = g.sort_values("t_start")
        _, iv = postprocess(
            g["t_start"].to_numpy(), g["score"].to_numpy(),
            threshold=float(g["tau"].iloc[0]), smooth_size=5,
            min_gap_s=2.0, min_dur_s=2.0, causal=True, window_sec=config.WINDOW_SEC,
        )
        detected[str(pid)] = [tuple(x) for x in iv]
        durations[str(pid)] = float(g["t_start"].max() + config.WINDOW_SEC)
        m = event_metrics_pooled(seizures, {str(pid): detected[str(pid)]}, {str(pid): durations[str(pid)]})
        per_patient.append({
            "patient": str(pid), "fold": int(g["fold"].iloc[0]), "n_crises": m["n_crises_reelles"],
            f"tp_{name}": m["tp"], f"fp_{name}": m["fp"], "heures": m["duree_totale_h"],
        })

    all_ev = event_metrics_pooled(seizures, detected, durations)
    ann_d = {p: v for p, v in detected.items() if p in annotated}
    ann_ev = event_metrics_pooled(seizures, ann_d, {p: durations[p] for p in ann_d})

    fold_f1 = [f1_score(g["label"], g["pred"], zero_division=0) for _, g in df.groupby("fold")]
    summary = {
        "modele": name,
        "F1_pool": f1_score(df["label"], df["pred"], zero_division=0),
        "F1_plis_moy": np.mean(fold_f1), "F1_plis_std": np.std(fold_f1),
        "Prec": precision_score(df["label"], df["pred"], zero_division=0),
        "Rappel": recall_score(df["label"], df["pred"], zero_division=0),
        "PR_AUC": average_precision_score(df["label"], df["score"]),
        "Detection_%": 100 * all_ev["taux_detection"],
        "TP": all_ev["tp"], "FN": all_ev["fn"], "FP": all_ev["fp"],
        "Heures_22": all_ev["duree_totale_h"], "FA_h_22": all_ev["fausses_alarmes_par_h"],
        "Heures_19": ann_ev["duree_totale_h"], "FA_h_19": ann_ev["fausses_alarmes_par_h"],
        "Lat_ev_s": all_ev["latence_moy_s"], "Lat_std_s": all_ev["latence_std_s"],
    }
    return summary, pd.DataFrame(per_patient)


s_lgb, p_lgb = evaluate("lgb", lgb, TAU_LGB)
s_eeg, p_eeg = evaluate("eeg", eeg, TAU_EEG)

summ = pd.DataFrame([s_lgb, s_eeg]).set_index("modele").T
pd.set_option("display.width", 200)
print("\n=== METRIQUES CAUSALES POOLEES (tau* par pli, validation interne) ===")
print(summ.to_string(float_format=lambda x: f"{x:.3f}"))

pp = p_lgb.merge(p_eeg, on=["patient", "fold", "n_crises", "heures"]).sort_values(["fold", "patient"])
pp = pp[pp["n_crises"] > 0].copy()
pp["manque_lgb"] = pp["n_crises"] - pp["tp_lgb"]
pp["manque_eeg"] = pp["n_crises"] - pp["tp_eeg"]
print("\n=== PAR PATIENT ANNOTE (crises, detectees, fausses alarmes) ===")
print(pp[["fold", "patient", "n_crises", "tp_lgb", "tp_eeg", "fp_lgb", "fp_eeg"]].to_string(index=False))

summ.to_csv(EXP / "comparaison_causale_poolee.csv")
pp.to_csv(EXP / "comparaison_par_patient.csv", index=False)
print("\nCSV sauvegardes dans", EXP)
