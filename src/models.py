"""Modèles de classification et protocoles d'évaluation.

Deux protocoles sont comparés explicitement :

* **split aléatoire stratifié 80/20** (exigé par le cahier des charges) — mais
  les fenêtres se recouvrant à 50 %, ce split **fuit** de l'information entre
  train et test (des fenêtres voisines quasi identiques se retrouvent des deux
  côtés), ce qui gonfle artificiellement les scores ;
* **validation par patient** (Leave-One-Patient-Out / GroupKFold) — le test
  porte sur un patient jamais vu, situation réaliste en clinique.

Le déséquilibre (~3,5 % de crises) est géré par ``class_weight`` et/ou SMOTE
appliqué **uniquement dans les folds d'entraînement** (via Pipeline imblearn).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    average_precision_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:  # SMOTE optionnel (imblearn présent dans l'environnement)
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except Exception:  # pragma: no cover
    HAS_IMBLEARN = False

#: Colonnes non-features du dataset.
META_COLS = ("patient", "label", "t_start")


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------
def feature_columns(df: pd.DataFrame) -> list[str]:
    """Liste des colonnes de features (hors métadonnées)."""
    return [c for c in df.columns if c not in META_COLS]


def clean_feature_matrix(
    df: pd.DataFrame, cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Nettoie la matrice de features : inf→NaN, supprime colonnes vides/constantes.

    Returns:
        (DataFrame nettoyé, liste des colonnes finales).
    """
    cols = cols or feature_columns(df)
    X = df[cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    gardees = [c for c in cols if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
    return X[gardees], gardees


def annotated_patients(seizures: pd.DataFrame) -> set[str]:
    """Patients portant au moins une crise annotée (utilisables en test)."""
    return set(seizures["Patient"].astype(str).unique())


# ---------------------------------------------------------------------------
# Protocoles de découpage
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    """Un split train/test : indices de lignes et identifiants de patients testés."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    test_patients: list[str]
    name: str = ""


def random_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = config.RANDOM_SEED,
    epochs_test_only: bool = True,
) -> Fold:
    """Split aléatoire stratifié 80/20 (les enregistrements sans crise restent en train).

    Args:
        epochs_test_only: si True, aucun patient non annoté n'est placé en test.
    """
    from sklearn.model_selection import train_test_split

    y = df["label"].to_numpy()
    if epochs_test_only:
        testable = df["patient"].isin(df.loc[df["label"] == 1, "patient"].unique())
        pool = df[testable]
        other = df[~testable]
        idx_tr, idx_te = train_test_split(
            pool.index.to_numpy(), test_size=test_size, stratify=pool["label"],
            random_state=seed,
        )
        train_idx = np.concatenate([idx_tr, other.index.to_numpy()])
    else:
        idx_tr, idx_te = train_test_split(
            df.index.to_numpy(), test_size=test_size, stratify=y, random_state=seed
        )
        train_idx = idx_tr
    test_patients = sorted(df.loc[idx_te, "patient"].unique().tolist())
    return Fold(np.sort(train_idx), np.sort(idx_te), test_patients, name="split_aleatoire")


def lopo_folds(df: pd.DataFrame, patients: set[str]) -> list[Fold]:
    """Leave-One-Patient-Out : un fold par patient annoté (test = ce patient)."""
    folds = []
    for pid in sorted(patients):
        test_idx = df.index[df["patient"] == pid].to_numpy()
        train_idx = df.index[df["patient"] != pid].to_numpy()
        folds.append(Fold(train_idx, test_idx, [pid], name=f"LOPO[{pid}]"))
    return folds


def group_kfold_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = config.RANDOM_SEED) -> list[Fold]:
    """GroupKFold par patient : équilibre le coût du LOPO (moins de folds)."""
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    groups = df["patient"].to_numpy()
    folds = []
    for k, (tr, te) in enumerate(gkf.split(df, groups=groups)):
        folds.append(Fold(tr, te, sorted(df.iloc[te]["patient"].unique().tolist()),
                          name=f"GroupKFold[{k}]"))
    return folds


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
def _make_pipeline(clf, scale: bool, use_smote: bool) -> Pipeline:
    """Pipeline [imputation] + [SMOTE] + [StandardScaler] + classifieur.

    L'imputation médiane est indispensable : les patients n'ont pas tous les
    mêmes canaux, donc certaines colonnes de features sont manquantes (NaN).
    SMOTE est placé après l'imputation et n'agit que sur le train.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_smote and HAS_IMBLEARN:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_SEED)))
        pipe_cls = ImbPipeline
    else:
        pipe_cls = Pipeline
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return pipe_cls(steps)


def get_models(use_smote: bool = False, seed: int = config.RANDOM_SEED) -> dict[str, Pipeline]:
    """Dictionnaire {nom: pipeline} des classifieurs du cahier des charges."""
    modeles = {
        "DecisionTree": _make_pipeline(
            DecisionTreeClassifier(max_depth=12, class_weight="balanced",
                                   random_state=seed), scale=False, use_smote=use_smote),
        "RandomForest": _make_pipeline(
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=seed), scale=False, use_smote=use_smote),
        "KNN": _make_pipeline(
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1), scale=True, use_smote=use_smote),
        "SVM": _make_pipeline(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                probability=False, random_state=seed), scale=True, use_smote=use_smote),
        "LogReg": _make_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               n_jobs=-1), scale=True, use_smote=use_smote),
    }
    if XGBClassifier is not None:
        modeles["XGBoost"] = _make_pipeline(
            XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          eval_metric="logloss", n_jobs=-1, random_state=seed),
            scale=False, use_smote=use_smote)
    if LGBMClassifier is not None:
        modeles["LightGBM"] = _make_pipeline(
            LGBMClassifier(n_estimators=400, num_leaves=31, class_weight="balanced",
                           n_jobs=-1, random_state=seed, verbose=-1),
            scale=False, use_smote=use_smote)
    return modeles


def _mi_score(X, y) -> np.ndarray:
    """Information mutuelle (déterministe) utilisée par la sélection de features."""
    return mutual_info_classif(X, y, random_state=config.RANDOM_SEED)


def add_mi_selection(model: Pipeline, k: int) -> Pipeline:
    """Insère une sélection top-k (information mutuelle) dans un pipeline.

    La sélection est placée **après l'imputation** (la MI ne gère pas les NaN) et
    **avant le classifieur**. Comme elle vit dans le pipeline, elle est donc
    **réapprise sur le seul train de chaque fold** : contrairement à un classement
    MI calculé sur l'ensemble du dataset, elle ne fuit pas d'information du test.

    Args:
        model: pipeline ``get_models`` (``imputer`` en premier étage).
        k: nombre de features à conserver.
    """
    mdl = clone(model)
    steps = list(mdl.steps)
    imputer = steps[0]
    reste = steps[1:]
    new_steps = [imputer, ("select", SelectKBest(score_func=_mi_score, k=k))] + reste
    return type(mdl)(new_steps)


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """Métriques de classification adaptées au déséquilibre."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "specificity": specificity,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
        m["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        m["roc_auc"] = np.nan
        m["pr_auc"] = np.nan
    return m


def fit_evaluate(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Entraîne un modèle sur train_idx, évalue sur test_idx.

    Returns:
        (métriques, scores de crise sur le test, prédictions binaires au seuil 0.5).
    """
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    mdl = clone(model)
    mdl.fit(X.iloc[train_idx], y[train_idx])
    if hasattr(mdl, "predict_proba"):
        score = mdl.predict_proba(X.iloc[test_idx])[:, 1]
    else:
        score = mdl.decision_function(X.iloc[test_idx])
    pred = (score >= 0.5).astype(int)
    return compute_metrics(y[test_idx], pred, score), score, pred


def run_protocol(
    df: pd.DataFrame,
    model: Pipeline,
    folds: list[Fold],
    model_name: str = "model",
    X: pd.DataFrame | None = None,
    y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Exécute un modèle sur une liste de folds et renvoie un tableau de métriques."""
    X = X if X is not None else clean_feature_matrix(df)[0]
    y = y if y is not None else df["label"].to_numpy()
    lignes = []
    for f in folds:
        m, _, _ = fit_evaluate(model, X, y, f.train_idx, f.test_idx)
        m["model"] = model_name
        m["fold"] = f.name
        m["test_patients"] = ",".join(f.test_patients)
        m["n_test"] = len(f.test_idx)
        lignes.append(m)
    return pd.DataFrame(lignes)


def summarise(results: pd.DataFrame, by: str = "model") -> pd.DataFrame:
    """Moyenne ± écart-type des métriques par modèle."""
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall",
                   "f1", "mcc", "specificity", "roc_auc", "pr_auc"]
    agg = results.groupby(by)[metric_cols].agg(["mean", "std"])
    return agg
