"""Étape 5b — améliorations testées (SANS FUITE) et ablation de canaux.

Révision suite à la revue de projet :

* la sélection de features par information mutuelle (top-k) est désormais
  **réapprise dans chaque fold d'entraînement** (via le pipeline) au lieu d'être
  calculée sur l'ensemble du dataset puis évaluée en GroupKFold — ce qui fuyait
  de l'information du test vers le train ;
* le seuil de décision est réglé de façon **nestée** (choisi sur le train,
  appliqué au test) ;
* le dataset principal est **EEG seul** (canaux EMG/ECG/SLI exclus) ; une
  variante « tous canaux » est conservée comme ligne d'ablation.

Protocole : LightGBM en GroupKFold(5) par patient.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src import config, models
from src.evaluation import nested_event_metrics, threshold_sweep
from src.features import build_feature_dataset
from src.io import load_annotations

pd.set_option("display.width", 240)


def cols_for_channels(cols: list[str], channels: set[str]) -> list[str]:
    """Sous-ensemble de colonnes de features restreint à des canaux donnés.

    Les features globales (``global__*``) sont toujours conservées.
    """
    out = []
    for c in cols:
        if c.startswith("global__"):
            out.append(c)
        elif c.split("__")[0] in channels:
            out.append(c)
    return out


def eval_subset(df, cols, model, folds, y, n_features: int | None = None) -> dict:
    """Évalue un modèle sur un sous-ensemble de colonnes via GroupKFold."""
    X = df[cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    res = []
    for f in folds:
        m, _, _ = models.fit_evaluate(model, X, y, f.train_idx, f.test_idx)
        res.append(m)
    r = pd.DataFrame(res)
    return {"f1": r.f1.mean(), "f1_std": r.f1.std(), "pr_auc": r.pr_auc.mean(),
            "recall": r.recall.mean(), "precision": r.precision.mean(),
            "mcc": r.mcc.mean(), "n_features": n_features or len(cols)}


def main() -> None:
    df = pd.read_parquet(config.FEATURES_PARQUET)          # dataset principal : EEG seul
    X, cols = models.clean_feature_matrix(df)
    y = df["label"].to_numpy()
    seizures, _ = load_annotations()
    folds = models.group_kfold_splits(df, n_splits=5)
    lgbm = models.get_models()["LightGBM"]

    resultats = []

    # --- 0. Référence : EEG seul, sans sélection --------------------------
    r = eval_subset(df, cols, lgbm, folds, y)
    r.update({"config": "EEG_seul"})
    resultats.append(r)
    print(f"  EEG_seul      ({len(cols):4d} feats) : F1={r['f1']:.3f}±{r['f1_std']:.3f} PR-AUC={r['pr_auc']:.3f}")

    # --- 1. Sélection MI top-k, réapprise DANS chaque fold ----------------
    t0 = time.time()
    mi = mutual_info_classif(X.iloc[::5].fillna(0), y[::5], random_state=config.RANDOM_SEED)
    pd.Series(mi, index=cols).sort_values(ascending=False).to_csv(
        config.CACHE_DIR / "mutual_info.csv", header=["mi"])   # descriptif (interprétation)
    for k in [30, 100, 250]:
        if k >= len(cols):
            continue
        mdl = models.add_mi_selection(lgbm, k)
        r = eval_subset(df, cols, mdl, folds, y, n_features=k)
        r.update({"config": f"MI_top{k}_infold"})
        resultats.append(r)
        print(f"  MI_top{k:4d}     : F1={r['f1']:.3f}±{r['f1_std']:.3f} PR-AUC={r['pr_auc']:.3f} ({time.time()-t0:.0f}s)")

    # --- 2. SMOTE (déjà appliqué dans les folds d'entraînement) -----------
    t0 = time.time()
    lgbm_smote = models.get_models(use_smote=True)["LightGBM"]
    r = eval_subset(df, cols, lgbm_smote, folds, y)
    r.update({"config": "LightGBM+SMOTE"})
    resultats.append(r)
    print(f"  SMOTE         : F1={r['f1']:.3f}±{r['f1_std']:.3f} PR-AUC={r['pr_auc']:.3f} ({time.time()-t0:.0f}s)")

    # --- 3. Ablation de canaux -------------------------------------------
    eeg_ch = {c.split("__")[0] for c in cols}
    all_path = config.CACHE_DIR / "features_all_channels.parquet"
    df_all = build_feature_dataset(cache_path=all_path, force=False, eeg_only=False)
    _, cols_all = models.clean_feature_matrix(df_all)

    groupes = {
        "ALL_tous_canaux": None,
        "EEG+EMG+ECG": eeg_ch | {"EMG1", "EMG2", "ECG"},
        "EEG_commun_8": set(config.COMMON_EEG_CHANNELS),
        "Frontal_7": {"EEGFp1", "EEGFp2", "EEGF7", "EEGF3", "EEGFz", "EEGF4", "EEGF8"},
        "Frontal_4": {"EEGFp1", "EEGFp2", "EEGF3", "EEGF4"},
    }
    for nom, chans in groupes.items():
        t0 = time.time()
        if chans is None:
            r = eval_subset(df_all, cols_all, lgbm, folds, y)
        else:
            base = df_all if nom == "EEG+EMG+ECG" else df
            base_cols = cols_all if nom == "EEG+EMG+ECG" else cols
            sub = cols_for_channels(base_cols, chans)
            r = eval_subset(base, sub, lgbm, folds, y)
        r.update({"config": f"Ablation_{nom}"})
        resultats.append(r)
        print(f"  {nom:16s} ({r['n_features']:4d} feats) : F1={r['f1']:.3f}±{r['f1_std']:.3f} "
              f"PR-AUC={r['pr_auc']:.3f} ({time.time()-t0:.0f}s)")

    tab = pd.DataFrame(resultats)
    tab.to_csv(config.CACHE_DIR / "improvements.csv", index=False)
    print("\nTableau des améliorations :")
    print(tab.round(3).to_string(index=False))

    # --- 4a. Réglage du seuil NESTÉ (choisi sur le train, appliqué au test) ---
    nested = nested_event_metrics(lgbm, df, X, y, folds, seizures)
    nested.to_csv(config.CACHE_DIR / "threshold_nested_event_metrics.csv", index=False)
    tp, fp, fn = int(nested.tp.sum()), int(nested.fp.sum()), int(nested.fn.sum())
    hours = float(nested.heures.sum())
    print("\n=== Seuil NESTÉ (GroupKFold, LightGBM) ===")
    print(nested.to_string(index=False))
    print(f"Global : taux_detection={tp/(tp+fn):.3f} | fausses_alarmes/h={fp/hours:.2f} "
          f"| TP={tp} FP={fp} FN={fn}")

    # --- 4b. Balayage descriptif du seuil (courbe, sans réglage) ----------
    preds = pd.read_parquet(config.PREDICTIONS_PARQUET)
    sweep = threshold_sweep(preds["label"].to_numpy(), preds["score"].to_numpy())
    sweep.to_parquet(config.CACHE_DIR / "threshold_sweep.parquet", index=False)
    print("\nSeuil au meilleur F1 (fenêtre, descriptif) :",
          round(sweep.loc[sweep.f1.idxmax(), "threshold"], 2),
          "F1 =", round(sweep.f1.max(), 3))

    from src import postprocess as pp
    from src.evaluation import event_metrics_by_patient

    durations = {pid: float(g.t_start.max()) + config.WINDOW_SEC
                 for pid, g in preds.groupby("patient")}
    lignes = []
    for t in np.arange(0.1, 0.95, 0.1):
        det = {}
        for pid, g in preds.groupby("patient"):
            _, iv = pp.postprocess(g["t_start"].to_numpy(), g["score"].to_numpy(),
                                   threshold=t, smooth_size=5, min_gap_s=2.0, min_dur_s=2.0)
            det[pid] = iv
        ev = event_metrics_by_patient(seizures, det, durations)
        total_h = sum(durations.values()) / 3600.0
        lignes.append({"seuil": round(float(t), 2),
                       "taux_detection": ev.tp.sum() / ev.n_crises_reelles.sum(),
                       "fausses_alarmes_par_h": ev.fp.sum() / total_h,
                       "latence_moy_s": np.nanmean(ev.latence_moy_s)})
    ev_tab = pd.DataFrame(lignes)
    ev_tab.to_csv(config.CACHE_DIR / "threshold_event_metrics.csv", index=False)
    print("\nMétriques par événement selon le seuil (descriptif) :")
    print(ev_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
