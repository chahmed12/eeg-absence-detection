"""Étape 3a — comparaison des modèles sur le split aléatoire stratifié 80/20.

ATTENTION : ce split est celui exigé par le cahier des charges, mais il fuit
(recouvrement 50 % des fenêtres). Il sert de point de comparaison avec la
validation par patient (LOPO) de l'étape suivante.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import config, models

pd.set_option("display.width", 220)


def main() -> None:
    df = pd.read_parquet(config.FEATURES_PARQUET)
    X, cols = models.clean_feature_matrix(df)
    y = df["label"].to_numpy()
    print(f"Dataset : {df.shape} | features retenues : {len(cols)} | "
          f"classe 1 : {100*y.mean():.2f}%")

    fold = models.random_stratified_split(df, test_size=0.2)
    print(f"Split aléatoire : train={len(fold.train_idx)} test={len(fold.test_idx)} "
          f"| patients test={len(fold.test_patients)}")

    resultats = []
    for name, mdl in models.get_models().items():
        t0 = time.time()
        res = models.run_protocol(df, mdl, [fold], model_name=name, X=X, y=y)
        res["duree_s"] = time.time() - t0
        resultats.append(res)
        m = res.iloc[0]
        print(f"  {name:14s} acc={m.accuracy:.3f} prec={m.precision:.3f} "
              f"rec={m.recall:.3f} f1={m.f1:.3f} mcc={m.mcc:.3f} "
              f"PR-AUC={m.pr_auc:.3f} ({m.duree_s:.1f}s)")

    tab = pd.concat(resultats, ignore_index=True)
    tab.to_parquet(config.CACHE_DIR / "results_random_split.parquet", index=False)
    print("\nTableau comparatif (split aléatoire) :")
    print(tab[["model", "accuracy", "balanced_accuracy", "precision", "recall",
               "f1", "mcc", "specificity", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
