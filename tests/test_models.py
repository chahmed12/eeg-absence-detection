"""Tests unitaires des modèles et protocoles de validation (src/models.py).

On utilise des DataFrames synthétiques pour ne pas dépendre des données EEG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.models import (
    Fold,
    annotated_patients,
    clean_feature_matrix,
    compute_metrics,
    feature_columns,
    fit_evaluate,
    get_models,
    group_kfold_splits,
    lopo_folds,
    patient_validation_split,
    random_stratified_split,
    run_protocol,
    summarise,
    add_mi_selection,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------
def _make_df(n: int = 200, n_feats: int = 10, seed: int = 42) -> pd.DataFrame:
    """DataFrame synthétique (patient, label, t_start, feat_*)."""
    rng = np.random.default_rng(seed)
    patients = [f"P{i:02d}" for i in range(5)]
    data = {
        "patient": np.tile(patients, n // len(patients) + 1)[:n],
        "label": (rng.random(n) > 0.85).astype(int),  # ~15 % de crises
        "t_start": np.arange(n) * 1.0,
    }
    for i in range(n_feats):
        data[f"feat_{i:02d}"] = rng.standard_normal(n).astype(np.float32)
    return pd.DataFrame(data)


@pytest.fixture
def df():
    return _make_df()


@pytest.fixture
def seizures_df():
    return pd.DataFrame({"Patient": ["P00", "P01", "P02"]})


# ---------------------------------------------------------------------------
# feature_columns / clean_feature_matrix
# ---------------------------------------------------------------------------
def test_feature_columns_excludes_meta(df):
    cols = feature_columns(df)
    assert "patient" not in cols
    assert "label" not in cols
    assert "t_start" not in cols
    assert all(c.startswith("feat_") for c in cols)


def test_clean_feature_matrix_shape(df):
    X, cols = clean_feature_matrix(df)
    assert X.shape[0] == len(df)
    assert len(cols) == X.shape[1]


def test_clean_feature_matrix_removes_constant_col(df):
    df2 = df.copy()
    df2["feat_constant"] = 42.0
    X, cols = clean_feature_matrix(df2)
    assert "feat_constant" not in cols


def test_clean_feature_matrix_removes_all_nan_col(df):
    df2 = df.copy()
    df2["feat_nan"] = np.nan
    X, cols = clean_feature_matrix(df2)
    assert "feat_nan" not in cols


def test_clean_feature_matrix_replaces_inf(df):
    df2 = df.copy()
    df2.loc[0, "feat_00"] = np.inf
    X, cols = clean_feature_matrix(df2)
    assert not np.isinf(X.values).any()


# ---------------------------------------------------------------------------
# annotated_patients
# ---------------------------------------------------------------------------
def test_annotated_patients(seizures_df):
    pats = annotated_patients(seizures_df)
    assert pats == {"P00", "P01", "P02"}


# ---------------------------------------------------------------------------
# Protocoles de découpage
# ---------------------------------------------------------------------------
def test_random_stratified_split_sizes(df):
    fold = random_stratified_split(df, test_size=0.2, seed=0)
    assert len(fold.train_idx) + len(fold.test_idx) <= len(df)
    assert len(fold.test_idx) > 0


def test_random_stratified_split_no_overlap(df):
    fold = random_stratified_split(df, test_size=0.2, seed=0)
    common = np.intersect1d(fold.train_idx, fold.test_idx)
    assert len(common) == 0


def test_lopo_folds_count(df, seizures_df):
    patients = annotated_patients(seizures_df)
    folds = lopo_folds(df, patients)
    assert len(folds) == len(patients)


def test_lopo_folds_no_overlap(df, seizures_df):
    patients = annotated_patients(seizures_df)
    folds = lopo_folds(df, patients)
    for fold in folds:
        common = np.intersect1d(fold.train_idx, fold.test_idx)
        assert len(common) == 0


def test_lopo_fold_test_is_single_patient(df, seizures_df):
    patients = annotated_patients(seizures_df)
    folds = lopo_folds(df, patients)
    for fold in folds:
        assert len(fold.test_patients) == 1


def test_group_kfold_count(df):
    folds = group_kfold_splits(df, n_splits=3)
    assert len(folds) == 3


def test_group_kfold_no_overlap(df):
    folds = group_kfold_splits(df, n_splits=3)
    for fold in folds:
        common = np.intersect1d(fold.train_idx, fold.test_idx)
        assert len(common) == 0


def test_patient_validation_split_disjoint_patients(df):
    inner_train, inner_val = patient_validation_split(df, df.index.to_numpy(), seed=0)
    train_patients = set(df.iloc[inner_train]["patient"])
    val_patients = set(df.iloc[inner_val]["patient"])
    assert train_patients.isdisjoint(val_patients)
    assert len(inner_train) + len(inner_val) == len(df)


def test_patient_validation_split_val_is_annotated(df):
    _, inner_val = patient_validation_split(df, df.index.to_numpy(), seed=0)
    assert len(inner_val) > 0
    assert (df.iloc[inner_val]["label"] == 1).any()


def test_patient_validation_split_keeps_positive_in_train(df):
    inner_train, _ = patient_validation_split(df, df.index.to_numpy(), seed=0)
    assert (df.iloc[inner_train]["label"] == 1).any()


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
def test_compute_metrics_perfect_prediction():
    y = np.array([0, 0, 1, 1, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 3 and m["fp"] == 0 and m["fn"] == 0


def test_compute_metrics_all_zeros():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.zeros(4, dtype=int)
    m = compute_metrics(y_true, y_pred)
    assert m["recall"] == 0.0
    assert m["fn"] == 2


def test_compute_metrics_with_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, (scores >= 0.5).astype(int), scores)
    assert "roc_auc" in m
    assert "pr_auc" in m
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0


def test_compute_metrics_keys():
    y = np.array([0, 1])
    m = compute_metrics(y, y)
    for key in ["accuracy", "balanced_accuracy", "precision", "recall",
                "f1", "mcc", "specificity", "tp", "fp", "fn", "tn"]:
        assert key in m


# ---------------------------------------------------------------------------
# fit_evaluate
# ---------------------------------------------------------------------------
def test_fit_evaluate_returns_dict_and_arrays(df):
    X, _ = clean_feature_matrix(df)
    y = df["label"].to_numpy()
    mdl = get_models()["DecisionTree"]
    train = np.arange(0, 150)
    test = np.arange(150, 200)
    metrics, scores, preds = fit_evaluate(mdl, X, y, train, test)
    assert isinstance(metrics, dict)
    assert len(scores) == len(test)
    assert len(preds) == len(test)
    assert set(preds).issubset({0, 1})


def test_fit_evaluate_scores_in_range(df):
    X, _ = clean_feature_matrix(df)
    y = df["label"].to_numpy()
    mdl = get_models()["RandomForest"]
    train = np.arange(0, 150)
    test = np.arange(150, 200)
    _, scores, _ = fit_evaluate(mdl, X, y, train, test)
    assert (scores >= 0).all() and (scores <= 1).all()


# ---------------------------------------------------------------------------
# run_protocol
# ---------------------------------------------------------------------------
def test_run_protocol_returns_dataframe(df):
    X, _ = clean_feature_matrix(df)
    y = df["label"].to_numpy()
    fold = random_stratified_split(df, test_size=0.2)
    mdl = get_models()["DecisionTree"]
    result = run_protocol(df, mdl, [fold], model_name="DT", X=X, y=y)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "f1" in result.columns
    assert result["model"].iloc[0] == "DT"


def test_run_protocol_multiple_folds(df, seizures_df):
    X, _ = clean_feature_matrix(df)
    y = df["label"].to_numpy()
    folds = group_kfold_splits(df, n_splits=3)
    mdl = get_models()["DecisionTree"]
    result = run_protocol(df, mdl, folds, model_name="DT", X=X, y=y)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# get_models
# ---------------------------------------------------------------------------
def test_get_models_returns_dict():
    models = get_models()
    assert isinstance(models, dict)
    assert len(models) >= 4   # DT, RF, KNN, LogReg au minimum


def test_get_models_all_have_fit_predict():
    for name, mdl in get_models().items():
        assert hasattr(mdl, "fit"), f"{name} n'a pas de méthode fit"
        assert hasattr(mdl, "predict"), f"{name} n'a pas de méthode predict"


# ---------------------------------------------------------------------------
# add_mi_selection
# ---------------------------------------------------------------------------
def test_add_mi_selection_inserts_select_step(df):
    mdl = get_models()["RandomForest"]
    mdl_mi = add_mi_selection(mdl, k=5)
    step_names = [s[0] for s in mdl_mi.steps]
    assert "select" in step_names


def test_add_mi_selection_trains_correctly(df):
    X, _ = clean_feature_matrix(df)
    y = df["label"].to_numpy()
    mdl = get_models()["DecisionTree"]
    mdl_mi = add_mi_selection(mdl, k=5)
    # Ne doit pas lever d'exception
    mdl_mi.fit(X.iloc[:150], y[:150])
    preds = mdl_mi.predict(X.iloc[150:])
    assert len(preds) == len(X) - 150


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------
def test_summarise_returns_dataframe(df):
    X, _ = clean_feature_matrix(df)
    y = df["label"].to_numpy()
    fold = random_stratified_split(df, test_size=0.2)
    mdl = get_models()["DecisionTree"]
    result = run_protocol(df, mdl, [fold], model_name="DT", X=X, y=y)
    summary = summarise(result)
    assert isinstance(summary, pd.DataFrame)
    assert "DT" in summary.index
