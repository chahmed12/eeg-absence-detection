"""Étape 5a — modèles étendus (XGBoost, LightGBM) et explicabilité.

* comparaison des modèles sur split aléatoire et sur GroupKFold(5) par patient ;
* importance des features (forêt aléatoire) globale, par type et par canal ;
* carte topographique de l'importance par canal (montage 10-20).
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, models, viz

pd.set_option("display.width", 240)


def main() -> None:
    viz.set_report_style()
    df = pd.read_parquet(config.FEATURES_PARQUET)
    X, cols = models.clean_feature_matrix(df)
    y = df["label"].to_numpy()

    # --- 1. Split aléatoire : tous les modèles ---------------------------
    fold = models.random_stratified_split(df, test_size=0.2)
    toutes = models.get_models()
    res = []
    for name, mdl in toutes.items():
        t0 = time.time()
        r = models.run_protocol(df, mdl, [fold], model_name=name, X=X, y=y)
        r["duree_s"] = time.time() - t0
        res.append(r)
        m = r.iloc[0]
        print(f"  [aléatoire] {name:14s} F1={m.f1:.3f} PR-AUC={m.pr_auc:.3f} MCC={m.mcc:.3f} ({m.duree_s:.0f}s)")
    tab_rand = pd.concat(res, ignore_index=True)
    tab_rand.to_parquet(config.CACHE_DIR / "results_random_extended.parquet", index=False)

    # --- 2. GroupKFold(5) par patient : les modèles phares ----------------
    folds = models.group_kfold_splits(df, n_splits=5)
    cles = ["RandomForest", "XGBoost", "LightGBM"]
    res = []
    for name in cles:
        t0 = time.time()
        r = models.run_protocol(df, toutes[name], folds, model_name=name, X=X, y=y)
        r["duree_s"] = time.time() - t0
        res.append(r)
        print(f"  [GroupKFold] {name:14s} F1={r.f1.mean():.3f}±{r.f1.std():.3f} "
              f"PR-AUC={r.pr_auc.mean():.3f} ({r.duree_s.iloc[0]:.0f}s)")
    tab_gkf = pd.concat(res, ignore_index=True)
    tab_gkf.to_parquet(config.CACHE_DIR / "results_groupkfold.parquet", index=False)

    # --- 3. Explicabilité : importance des features (RF) ------------------
    from sklearn.base import clone
    mdl = clone(toutes["RandomForest"])
    mdl.fit(X.iloc[fold.train_idx], y[fold.train_idx])
    rf = mdl.named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    imp.to_csv(config.CACHE_DIR / "feature_importance.csv", header=["importance"])

    fig = viz.plot_top_features(imp, top_n=30,
                                title="RandomForest — 30 features les plus importantes")
    viz.save_fig(fig, "05_top_features")
    plt.close(fig)

    # importance par canal (somme des features du canal)
    par_canal: dict[str, float] = {}
    for c, v in imp.items():
        ch = c.split("__")[0]
        par_canal[ch] = par_canal.get(ch, 0.0) + float(v)
    pd.Series(par_canal).sort_values(ascending=False).to_csv(config.CACHE_DIR / "channel_importance.csv")
    print("\nTop 10 canaux :", pd.Series(par_canal).sort_values(ascending=False).head(10).round(4).to_dict())

    fig = viz.plot_channel_topomap(par_canal, title="Importance par canal (RandomForest)")
    viz.save_fig(fig, "05_channel_topomap")
    plt.close(fig)

    # importance par type de feature
    par_type: dict[str, float] = {}
    for c, v in imp.items():
        if "__" not in c:
            par_type[c] = par_type.get(c, 0.0) + float(v)
            continue
        feat = c.split("__", 1)[1]
        par_type[feat] = par_type.get(feat, 0.0) + float(v)
    print("Top 10 types de features :",
          pd.Series(par_type).sort_values(ascending=False).head(10).round(4).to_dict())

    print("\nRésumé GroupKFold :")
    print(tab_gkf.groupby("model")[["f1", "pr_auc", "recall", "mcc"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
