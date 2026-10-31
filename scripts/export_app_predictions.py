"""Exporte les prédictions P0 utilisées par l'application Streamlit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src import config

EXP_DIR = config.CACHE_DIR / "experiments"


def _fold_metadata(folds: list[dict]) -> tuple[dict[str, int], dict[str, float]]:
    fold_by_patient: dict[str, int] = {}
    tau_by_patient: dict[str, float] = {}
    for row in folds:
        fold = int(row["fold"])
        tau = float(row["tau_star"])
        for patient in row.get("test_patients", row.get("patients", [])):
            fold_by_patient[str(patient)] = fold
            tau_by_patient[str(patient)] = tau
    return fold_by_patient, tau_by_patient


def export_lightgbm() -> pd.DataFrame:
    preds = pd.read_parquet(EXP_DIR / "predictions_lightgbm_oof.parquet")
    folds = json.loads((EXP_DIR / "EXP-00_nested_results.json").read_text(encoding="utf-8"))
    fold_by_patient, tau_by_patient = _fold_metadata(folds)
    preds["fold"] = preds["patient"].map(fold_by_patient)
    preds["tau_star"] = preds["patient"].map(tau_by_patient)
    if preds[["fold", "tau_star"]].isna().any().any():
        raise ValueError("Métadonnées de pli/seuil manquantes pour LightGBM")
    preds["pred"] = (preds["score"] >= preds["tau_star"]).astype(int)
    preds = preds[["patient", "label", "t_start", "score", "pred", "fold", "tau_star"]]
    preds.to_parquet(config.PREDICTIONS_PARQUET, index=False)
    return preds


def export_eegnet() -> pd.DataFrame:
    preds = pd.read_pickle(EXP_DIR / "EXP-03_oof_predictions.pkl")
    exp4 = json.loads((EXP_DIR / "EXP-04_metrics.json").read_text(encoding="utf-8"))
    _, tau_by_patient = _fold_metadata(exp4["nested"]["folds"])
    preds["tau_star"] = preds["patient"].map(tau_by_patient)
    if preds["tau_star"].isna().any():
        raise ValueError("Seuil P0 manquant pour EEGNet")
    preds["pred"] = (preds["score"] >= preds["tau_star"]).astype(int)
    preds = preds[["patient", "label", "t_start", "score", "pred", "fold", "tau_star"]]
    preds.to_parquet(config.CACHE_DIR / "predictions_eegnet.parquet", index=False)
    return preds


def main() -> None:
    light = export_lightgbm()
    eegnet = export_eegnet()
    print(f"LightGBM : {len(light)} fenêtres, {light.patient.nunique()} patients")
    print(f"EEGNet   : {len(eegnet)} fenêtres, {eegnet.patient.nunique()} patients")


if __name__ == "__main__":
    main()
