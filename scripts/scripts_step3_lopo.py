"""Étape 3b — validation par patient (Leave-One-Patient-Out).

Le test porte sur un patient jamais vu : c'est l'évaluation honnête en clinique.
On rapporte la moyenne ± écart-type sur les patients et on compare au split
aléatoire (qui, lui, fuit à cause du recouvrement des fenêtres).

Un sous-ensemble de modèles est évalué (SVM coûteux -> option --svm).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src import config, models

pd.set_option("display.width", 220)


def main(svm: bool = False) -> None:
    df = pd.read_parquet(config.FEATURES_PARQUET)
    X, cols = models.clean_feature_matrix(df)
    y = df["label"].to_numpy()
    pids = models.annotated_patients(_load_seizures())
    folds = models.lopo_folds(df, pids)
    print(f"LOPO : {len(folds)} folds (patients annotés) | features : {len(cols)}")

    tous = models.get_models()
    noms = ["DecisionTree", "RandomForest", "KNN", "LogReg"] + (["SVM"] if svm else [])
    resultats = []
    for name in noms:
        t0 = time.time()
        res = models.run_protocol(df, tous[name], folds, model_name=name, X=X, y=y)
        res["duree_s"] = time.time() - t0
        resultats.append(res)
        print(f"  {name:14s} F1={res.f1.mean():.3f}±{res.f1.std():.3f} "
              f"PR-AUC={res.pr_auc.mean():.3f} recall={res.recall.mean():.3f} "
              f"({res.duree_s.iloc[0]:.0f}s)")

    tab = pd.concat(resultats, ignore_index=True)
    tab.to_parquet(config.CACHE_DIR / "results_lopo.parquet", index=False)
    resume = tab.groupby("model")[["accuracy", "balanced_accuracy", "precision",
                                   "recall", "f1", "mcc", "specificity", "roc_auc", "pr_auc"]].agg(
        ["mean", "std"]).round(3)
    print("\nRésumé LOPO (moyenne ± écart-type sur les patients) :")
    print(resume.to_string())


def _load_seizures() -> pd.DataFrame:
    from src.io import load_annotations

    s, _ = load_annotations()
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--svm", action="store_true", help="inclure SVM (coûteux)")
    args = ap.parse_args()
    main(svm=args.svm)
