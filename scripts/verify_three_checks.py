"""Audit P1 des points critiques du rapport EEG.

Ce script remplace les anciennes conclusions « zéro fuite » par un audit
falsifiable. Il vérifie :

1. la traçabilité des chiffres LightGBM historiques ;
2. l'agrégation de la latence (moyenne par patient/pli vs par événement) ;
3. l'étanchéité des plis patient, du FoldScaler et la structure de sélection
   du seuil.

Limite explicite : la vérification 3.4 est structurelle. Elle ne prouve pas
que les scores OOF utilisés pour la validation sont exempts de contamination.
C'est précisément l'objet de la phase P0.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import numpy as np
import pandas as pd

from src import config, models
from src.deep_learning import FoldScaler
from src.evaluation import event_metrics_by_patient, match_events
from src.io import list_mat_files, load_annotations
from src.postprocess import postprocess


def _fold_patients(fold: dict) -> list[str]:
    return [str(p) for p in fold.get("test_patients", fold.get("patients", []))]


def _event_latency_stats(
    preds: pd.DataFrame,
    folds: list[dict],
    seizures: pd.DataFrame,
    causal: bool,
) -> dict:
    pat_fold = {
        pid: k
        for k, fold in enumerate(folds)
        for pid in _fold_patients(fold)
    }
    latencies: list[float] = []
    per_patient: dict[str, float] = {}
    per_fold: dict[int, list[float]] = {}
    tp = fp = fn = 0

    for pid, g in preds.groupby("patient"):
        pid = str(pid)
        k = pat_fold.get(pid)
        if k is None:
            continue
        g = g.sort_values("t_start")
        _, intervals = postprocess(
            g["t_start"].to_numpy(),
            g["score"].to_numpy(),
            threshold=folds[k]["tau_star"],
            smooth_size=5,
            min_gap_s=2.0,
            min_dur_s=2.0,
            causal=causal,
            window_sec=config.WINDOW_SEC,
        )
        true = seizures[seizures["Patient"].astype(str) == pid]
        true_events = list(map(tuple, true[["Debut_Sec", "Fin_Sec"]].to_numpy()))
        matched = match_events(true_events, list(intervals))
        fold_latencies = list(matched["latences"])
        latencies.extend(fold_latencies)
        tp += int(matched["tp"])
        fp += int(matched["fp"])
        fn += int(matched["fn"])
        if fold_latencies:
            per_patient[pid] = float(np.mean(fold_latencies))
            per_fold.setdefault(k, []).extend(fold_latencies)

    lat = np.asarray(latencies, dtype=float)
    return {
        "latencies": lat,
        "per_patient": per_patient,
        "per_fold": per_fold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def check_1_lightgbm_history() -> None:
    print("=" * 80)
    print("VÉRIFICATION 1 : TRAÇABILITÉ DES RÉSULTATS LIGHTGBM")
    print("=" * 80)

    print("\n[1.1] Valeurs historiques annoncées : 86,6 % / 3,35 FA/h / 0,9 s.")
    print("      Protocole annoncé : seuil 0,20 choisi rétrospectivement sur la courbe OOF,")
    print("      19 patients annotés, 8,06 h, latence hors-ligne non causale.")
    print("      La ligne 62,2 % / 1,24 FA/h est un RandomForest LOPO, pas LightGBM.")

    preds_path = config.CACHE_DIR / "predictions.parquet"
    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        seizures, _ = load_annotations()
        durations = {
            str(pid): float(g["t_start"].max()) + config.WINDOW_SEC
            for pid, g in preds.groupby("patient")
        }
        total_h = sum(durations.values()) / 3600.0
        print(f"\n[1.2] Reproduction sur {preds_path.name} :")
        print(f"      {len(durations)} patients, {total_h:.2f} h.")
        for threshold in (0.20, 0.30, 0.50):
            detected = {}
            for pid, g in preds.groupby("patient"):
                g = g.sort_values("t_start")
                _, intervals = postprocess(
                    g["t_start"].to_numpy(),
                    g["score"].to_numpy(),
                    threshold=threshold,
                    smooth_size=5,
                    min_gap_s=2.0,
                    min_dur_s=2.0,
                    causal=False,
                )
                detected[str(pid)] = intervals
            ev = event_metrics_by_patient(seizures, detected, durations)
            detection = 100.0 * ev["tp"].sum() / ev["n_crises_reelles"].sum()
            fa_per_h = ev["fp"].sum() / total_h
            print(
                f"      seuil {threshold:.2f} : détection={detection:.1f} % | "
                f"FA/h={fa_per_h:.2f} | TP={int(ev['tp'].sum())} "
                f"FP={int(ev['fp'].sum())} FN={int(ev['fn'].sum())}"
            )
        print("      -> 86,6 % / 3,35 FA/h n'est pas reproduit : à 0,20 on obtient")
        print("         3,35 FA/h mais 92,7 % de détection ; à 0,30, 3,10 FA/h et 87,8 %.")
    else:
        print(f"\n[1.2] {preds_path} absent : reproduction impossible.")

    res_path = config.CACHE_DIR / "experiments" / "EXP-00_nested_results.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        taus = np.asarray([r["tau_star"] for r in results], dtype=float)
        print("\n[1.3] EXP-00 (GroupKFold, sélection NON strictement imbriquée) :")
        print(
            f"      tau* : moyenne={taus.mean():.2f}, médiane={np.median(taus):.2f}, "
            f"SD population={taus.std(ddof=0):.2f}, SD échantillon={taus.std(ddof=1):.2f}"
        )
        print("      Le seuil est choisi sur les OOF des 4 autres plis, mais ces OOF")
        print("      proviennent de modèles entraînés sur le pli de test k.")
    else:
        print(f"\n[1.3] {res_path} absent.")

    print("\n  -> CONCLUSION CHECK 1 : les chiffres historiques ne sont pas traçables")
    print("     avec les artefacts actuels. EXP-00 n'est pas une référence sans fuite.")


def check_2_latency_calculation() -> None:
    print("\n" + "=" * 80)
    print("VÉRIFICATION 2 : AGRÉGATION ET COMPARABILITÉ DE LA LATENCE")
    print("=" * 80)

    seizures, _ = load_annotations()
    lgbm_path = config.CACHE_DIR / "experiments" / "predictions_lightgbm_oof.parquet"
    eeg_path = config.CACHE_DIR / "experiments" / "EXP-03_oof_predictions.pkl"
    lgbm_results = config.CACHE_DIR / "experiments" / "EXP-00_nested_results.json"
    eeg_results = config.CACHE_DIR / "experiments" / "EXP-04_metrics.json"

    if not all(p.exists() for p in (lgbm_path, eeg_path, lgbm_results, eeg_results)):
        print("  Artefacts manquants : vérification impossible.")
        return

    with open(lgbm_results, "r", encoding="utf-8") as f:
        lgbm_folds = json.load(f)
    with open(eeg_results, "r", encoding="utf-8") as f:
        eeg_folds = json.load(f)["nested"]["folds"]

    datasets = {
        "LightGBM": (pd.read_parquet(lgbm_path), lgbm_folds),
        "EEGNet": (pd.read_pickle(eeg_path), eeg_folds),
    }

    print("\n[2.1] Latence causale de post-traitement (TP uniquement, FN exclus) :")
    for name, (preds, folds) in datasets.items():
        stats = _event_latency_stats(preds, folds, seizures, causal=True)
        lat = stats["latencies"]
        per_patient = np.asarray(list(stats["per_patient"].values()), dtype=float)
        per_fold = np.asarray(
            [np.mean(v) for v in stats["per_fold"].values()], dtype=float
        )
        print(f"\n  {name} : {len(lat)} événements appariés (TP={stats['tp']}, FN={stats['fn']})")
        print(
            f"    moyenne par événement : {lat.mean():.2f} s "
            f"(SD population={lat.std(ddof=0):.2f}, échantillon={lat.std(ddof=1):.2f})"
        )
        print(
            f"    moyenne des patients  : {per_patient.mean():.2f} s "
            f"(SD population={per_patient.std(ddof=0):.2f})"
        )
        print(
            f"    moyenne des plis      : {per_fold.mean():.2f} s "
            f"(SD population={per_fold.std(ddof=0):.2f}, "
            f"échantillon={per_fold.std(ddof=1):.2f})"
        )
        top = sorted(stats["per_patient"].items(), key=lambda item: -item[1])[:3]
        print("    patients les plus lents : " + ", ".join(f"{pid}={value:.2f}s" for pid, value in top))

    print("\n[2.2] Les valeurs du rapport :")
    print("      6,83 s (LightGBM) et 5,26 s (EEGNet) sont des moyennes de moyennes")
    print("      par patient puis par pli. Les moyennes pondérées par événement sont")
    print("      4,96 s (74 événements) et 5,03 s (71 événements).")
    print("      L'écart 6,83 vs 5,26 est largement gonflé par le patient 230515B-G")
    print("      (1 TP à 26 s) côté LightGBM. Le ±0,79 s annoncé pour EEGNet n'est pas")
    print("      reproduit : SD population=0,755 s, SD échantillon=0,844 s.")
    print("\n  -> CONCLUSION CHECK 2 : les latences ne sont pas comparables telles quelles ;")
    print("     l'agrégation par événement doit être la convention principale.")


def _assert_threshold_selection_uses_val(script_path: Path) -> None:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "_best_threshold_on_df":
                calls.append(node)
    assert calls, f"Aucun appel à _best_threshold_on_df dans {script_path.name}"
    for call in calls:
        assert call.args, f"Appel sans argument dans {script_path.name}"
        first = call.args[0]
        assert isinstance(first, ast.Name) and first.id == "df_val", (
            f"{script_path.name}: le seuil n'est pas sélectionné sur df_val"
        )
        assert not any(
            isinstance(arg, ast.Name) and arg.id == "df_test" for arg in call.args
        ), f"{script_path.name}: df_test est passé à la sélection du seuil"


def check_3_data_leakage() -> None:
    print("\n" + "=" * 80)
    print("VÉRIFICATION 3 : ÉTANCHÉITÉ DES PLIS, DU SCALER ET DU SEUIL")
    print("=" * 80)

    df_feats = pd.read_parquet(config.FEATURES_PARQUET)
    folds = models.group_kfold_splits(
        df_feats, n_splits=5, seed=config.RANDOM_SEED
    )
    all_patients = set(df_feats["patient"].astype(str))
    seizures, _ = load_annotations()
    annotated = set(seizures["Patient"].astype(str))
    controls = sorted(all_patients - annotated)

    print(f"[3.1] {len(all_patients)} identifiants uniques, "
          f"{len(annotated)} annotés, {len(controls)} sans crise : {controls}")
    test_counts: dict[str, int] = {}
    for k, fold in enumerate(folds):
        train_pats = set(df_feats.iloc[fold.train_idx]["patient"].astype(str))
        test_pats = set(df_feats.iloc[fold.test_idx]["patient"].astype(str))
        inter = train_pats.intersection(test_pats)
        assert not inter, f"Pli {k + 1} : fuite patient {sorted(inter)}"
        for pid in test_pats:
            test_counts[pid] = test_counts.get(pid, 0) + 1
        print(
            f"  Pli {k + 1} : {len(train_pats)} train, {len(test_pats)} test, "
            f"intersection={sorted(inter)}"
        )
    assert set(test_counts) == all_patients, "Patients absents des plis de test"
    assert all(count == 1 for count in test_counts.values()), (
        "Un patient apparaît dans plusieurs plis de test"
    )

    print("\n[3.2] Vérification des enregistrements multiples :")
    mat_files = list_mat_files()
    assert len(mat_files) == len(all_patients), (
        f"{len(mat_files)} fichiers .mat pour {len(all_patients)} identifiants"
    )
    print(f"  {len(mat_files)} fichiers .mat, {len(all_patients)} identifiants.")
    print("  Chaque identifiant apparaît dans exactement un pli de test : OK.")

    print("\n[3.3] Vérification runtime du FoldScaler (train-only) :")
    rng = np.random.default_rng(42)
    x_train = (rng.normal(5.0, 10.0, size=(40, 8, 64))).astype(np.float32)
    x_test = (rng.normal(1.0, 2.0, size=(10, 8, 64))).astype(np.float32)
    scaler = FoldScaler().fit(x_train)
    train_norm = scaler.transform(x_train)
    test_norm = scaler.transform(x_test)
    assert np.allclose(train_norm.mean(axis=(0, 2)), 0.0, atol=1e-4)
    assert np.allclose(train_norm.std(axis=(0, 2)), 1.0, atol=1e-4)
    assert np.allclose(scaler.mean.ravel(), x_train.mean(axis=(0, 2)), atol=1e-5)
    assert abs(float(test_norm.mean())) > 0.1, (
        "Le test semble avoir été normalisé avec ses propres statistiques"
    )
    print("  mean/std appris sur X_train uniquement ; X_test transformé sans re-fit : OK.")

    print("\n[3.4] Vérification structurelle de la sélection du seuil :")
    for k, fold in enumerate(folds):
        val_idx = df_feats.index[~df_feats.index.isin(fold.test_idx)].to_numpy()
        assert len(np.intersect1d(val_idx, fold.test_idx)) == 0, (
            f"Pli {k + 1} : df_val contient des indices de test"
        )
    print("  df_val est disjoint de df_test pour les 5 plis.")
    for name in ("scripts_step6_exp00_lightgbm_nested.py",
                 "scripts_step6_exp04_threshold.py"):
        _assert_threshold_selection_uses_val(PROJ_ROOT / "scripts" / name)
        print(f"  {name} : seuil sélectionné sur df_val (vérification AST).")
    print("  ATTENTION : cette vérification est structurelle. Les scores OOF de df_val")
    print("  restent contaminés : pour LightGBM, leurs modèles ont vu le pli de test k ;")
    print("  pour le DL, le checkpoint/scheduler a été choisi sur le pli de test.")
    print("\n  -> CONCLUSION CHECK 3 : pas de fuite patient ; fuite de sélection OUI.")
    print("     La correction P0 (split interne par patient) est requise avant de")
    print("     présenter les résultats DL comme une généralisation sans fuite.")


if __name__ == "__main__":
    check_1_lightgbm_history()
    check_2_latency_calculation()
    check_3_data_leakage()
