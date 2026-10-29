"""Étape 5c — sensibilité à la taille de fenêtre et au recouvrement.

Grille évaluée : fenêtre {1, 2, 4} s (recouvrement 50 %) et recouvrement
{0, 50, 75} % (fenêtre 2 s). Pour rester comparable, tous les datasets de cette
étude utilisent le même jeu de features (sans entropie de permutation ni
ondelettes, coûteuses) et le même modèle (LightGBM, GroupKFold 5).
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, models

pd.set_option("display.width", 220)


def evaluate(df: pd.DataFrame, model) -> dict:
    X, cols = models.clean_feature_matrix(df)
    y = df["label"].to_numpy()
    folds = models.group_kfold_splits(df, n_splits=5)
    res = []
    for f in folds:
        m, _, _ = models.fit_evaluate(model, X, y, f.train_idx, f.test_idx)
        res.append(m)
    r = pd.DataFrame(res)
    return {"n_fenetres": len(df), "n_features": len(cols),
            "taux_crise": float(100 * y.mean()),
            "f1": r.f1.mean(), "f1_std": r.f1.std(), "pr_auc": r.pr_auc.mean(),
            "recall": r.recall.mean(), "precision": r.precision.mean()}


def main() -> None:
    from src import viz
    viz.set_report_style()
    lgbm = models.get_models()["LightGBM"]
    kwargs = dict(include_perm=False, include_wavelet=False)
    grille = [(1.0, 0.5), (2.0, 0.5), (4.0, 0.5), (2.0, 0.0), (2.0, 0.75)]
    lignes = []
    for win, ov in grille:
        cache = config.CACHE_DIR / f"features_w{win}_o{ov}.parquet"
        t0 = time.time()
        from src.features import build_feature_dataset

        df = build_feature_dataset(cache_path=cache, force=False,
                                   window_sec=win, overlap=ov, **kwargs)
        r = evaluate(df, lgbm)
        r.update({"fenetre_s": win, "recouvrement": ov, "duree_extraction_s": time.time() - t0})
        lignes.append(r)
        print(f"  fenêtre={win}s recouvrement={int(ov*100)}% : n={r['n_fenetres']} "
              f"F1={r['f1']:.3f}±{r['f1_std']:.3f} PR-AUC={r['pr_auc']:.3f} "
              f"({r['duree_extraction_s']:.0f}s)")

    tab = pd.DataFrame(lignes)
    tab.to_csv(config.CACHE_DIR / "window_overlap_sensitivity.csv", index=False)
    print("\nÉtude de sensibilité :")
    print(tab.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(viz._fw(13), viz._fh(4)))
    w = tab[tab.recouvrement == 0.5].sort_values("fenetre_s")
    axes[0].errorbar(w.fenetre_s, w.f1, yerr=w.f1_std, marker="o", capsize=4, label="F1")
    axes[0].plot(w.fenetre_s, w.pr_auc, marker="s", label="PR-AUC")
    axes[0].set_xlabel("Taille de fenêtre (s)"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensibilité à la taille de fenêtre"); axes[0].legend(); axes[0].grid(alpha=.4)
    o = tab[tab.fenetre_s == 2.0].sort_values("recouvrement")
    axes[1].errorbar(o.recouvrement * 100, o.f1, yerr=o.f1_std, marker="o", capsize=4, label="F1")
    axes[1].plot(o.recouvrement * 100, o.pr_auc, marker="s", label="PR-AUC")
    axes[1].set_xlabel("Recouvrement (%)"); axes[1].set_ylabel("Score")
    axes[1].set_title("Sensibilité au recouvrement"); axes[1].legend(); axes[1].grid(alpha=.4)
    from src import viz
    viz.save_fig(fig, "05_window_overlap_sensitivity"); plt.close(fig)


if __name__ == "__main__":
    main()
