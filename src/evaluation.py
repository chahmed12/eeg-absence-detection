"""Métriques par ÉVÉNEMENT (et non par fenêtre) pour la détection temporelle.

En clinique, ce qui compte n'est pas la classification de chaque fenêtre mais :
le **taux de crises détectées**, le **nombre de fausses alarmes par heure**, la
**latence** de détection du début, et l'**erreur** sur les bornes début/fin —
avec une tolérance correspondant au bruit d'annotation humaine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import config
from . import postprocess as pp


def match_events(
    true_events: list[tuple[float, float]],
    detected_events: list[tuple[float, float]],
    tolerance_s: float = config.EVENT_TOLERANCE_S,
) -> dict:
    """Apparie détections et crises réelles (1-1, glouton, chevauchement ± tolérance).

    Une crise réelle est « détectée » si au moins une détection la chevauche
    (avec tolérance). Chaque détection ne peut apparier qu'une seule crise.

    Returns:
        dict avec tp, fp, fn et les paires appariées (latence, erreurs de bornes).
    """
    true_events = list(true_events)
    detected_events = list(detected_events)
    used = set()
    latences, err_debut, err_fin, err_duree = [], [], [], []
    tp = 0
    for ts, te in true_events:
        best = None
        for j, (ds, de) in enumerate(detected_events):
            if j in used:
                continue
            if ds <= te + tolerance_s and de >= ts - tolerance_s:
                best = j
                break
        if best is not None:
            used.add(best)
            tp += 1
            ds, de = detected_events[best]
            latences.append(ds - ts)
            err_debut.append(abs(ds - ts))
            err_fin.append(abs(de - te))
            err_duree.append(abs((de - ds) - (te - ts)))
    fp = len(detected_events) - tp
    fn = len(true_events) - tp
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "latences": np.asarray(latences),
        "err_debut": np.asarray(err_debut),
        "err_fin": np.asarray(err_fin),
        "err_duree": np.asarray(err_duree),
    }


def event_metrics(
    true_events: list[tuple[float, float]],
    detected_events: list[tuple[float, float]],
    duration_s: float,
    tolerance_s: float = config.EVENT_TOLERANCE_S,
) -> dict:
    """Indicateurs de détection par événement pour un enregistrement."""
    m = match_events(true_events, detected_events, tolerance_s)
    n_true = len(true_events)
    heures = duration_s / 3600.0 if duration_s > 0 else np.nan
    return {
        "n_crises_reelles": n_true,
        "n_detections": len(detected_events),
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "taux_detection": m["tp"] / n_true if n_true else np.nan,
        "fausses_alarmes_par_h": m["fp"] / heures if heures and heures > 0 else np.nan,
        "latence_moy_s": float(np.mean(m["latences"])) if len(m["latences"]) else np.nan,
        "err_debut_moy_s": float(np.mean(m["err_debut"])) if len(m["err_debut"]) else np.nan,
        "err_fin_moy_s": float(np.mean(m["err_fin"])) if len(m["err_fin"]) else np.nan,
        "err_duree_moy_s": float(np.mean(m["err_duree"])) if len(m["err_duree"]) else np.nan,
    }


def event_metrics_by_patient(
    seizures: pd.DataFrame,
    detected_by_patient: dict[str, list[tuple[float, float]]],
    durations: dict[str, float],
    tolerance_s: float = config.EVENT_TOLERANCE_S,
) -> pd.DataFrame:
    """Calcule les métriques événementielles pour chaque patient.

    Args:
        seizures: annotations globales.
        detected_by_patient: {patient: liste d'intervalles détectés (start, end)}.
        durations: {patient: durée de l'enregistrement en s}.
    """
    lignes = []
    for pid, detected in detected_by_patient.items():
        true = seizures[seizures["Patient"].astype(str) == str(pid)]
        true_events = list(map(tuple, true[["Debut_Sec", "Fin_Sec"]].to_numpy()))
        m = event_metrics(true_events, list(detected), durations.get(pid, 0.0), tolerance_s)
        m["patient"] = pid
        lignes.append(m)
    cols = ["patient", "n_crises_reelles", "n_detections", "tp", "fp", "fn",
            "taux_detection", "fausses_alarmes_par_h", "latence_moy_s",
            "err_debut_moy_s", "err_fin_moy_s", "err_duree_moy_s"]
    return pd.DataFrame(lignes)[cols]


def pr_curve_data(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points (recall, precision) pour une courbe PR."""
    from sklearn.metrics import precision_recall_curve

    p, r, _ = precision_recall_curve(y_true, scores)
    return r, p


def threshold_sweep(
    y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Évalue les métriques pour une grille de seuils de décision."""
    from .models import compute_metrics

    thresholds = thresholds if thresholds is not None else np.linspace(0.02, 0.98, 49)
    lignes = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        m = compute_metrics(y_true, pred, scores)
        m["threshold"] = t
        lignes.append(m)
    return pd.DataFrame(lignes)


def select_threshold(
    y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | None = None
) -> tuple[float, float]:
    """Seuil maximisant le F1 (fenêtre). **À n'utiliser que sur le train.**

    Returns:
        (seuil retenu, F1 correspondant).
    """
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -np.inf
    for t in thresholds:
        f1 = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def nested_event_metrics(
    model,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list,
    seizures: pd.DataFrame,
    thresholds: np.ndarray | None = None,
    smooth_size: int = 5,
    min_gap_s: float = 2.0,
    min_dur_s: float = 2.0,
) -> pd.DataFrame:
    """Métriques par événement avec **réglage NESTÉ** du seuil (sans fuite).

    Pour chaque fold : le modèle est entraîné sur le train, le seuil optimal est
    choisi sur le **train** (F1 fenêtre), puis appliqué au **test** pour calculer
    les métriques événementielles. Le seuil n'est donc jamais choisi sur les
    données servies à l'évaluation. Les folds GroupKFold étant disjoints en
    patients, les totaux sont additionnables.
    """
    from sklearn.base import clone

    thresholds = thresholds if thresholds is not None else np.linspace(0.1, 0.9, 17)
    lignes = []
    for f in folds:
        mdl = clone(model)
        mdl.fit(X.iloc[f.train_idx], y[f.train_idx])
        s_tr = mdl.predict_proba(X.iloc[f.train_idx])[:, 1]
        t, f1_tr = select_threshold(y[f.train_idx], s_tr, thresholds)
        s_te = mdl.predict_proba(X.iloc[f.test_idx])[:, 1]

        test = df.iloc[f.test_idx][["patient", "t_start"]].copy()
        test["score"] = s_te
        det: dict[str, list[tuple[float, float]]] = {}
        durations: dict[str, float] = {}
        for pid, g in test.groupby("patient"):
            g = g.sort_values("t_start")
            _, iv = pp.postprocess(g["t_start"].to_numpy(), g["score"].to_numpy(),
                                   threshold=t, smooth_size=smooth_size,
                                   min_gap_s=min_gap_s, min_dur_s=min_dur_s)
            det[pid] = iv
            durations[pid] = float(g["t_start"].max()) + config.WINDOW_SEC

        ev = event_metrics_by_patient(seizures, det, durations)
        lignes.append({
            "fold": f.name, "seuil_train": round(t, 3), "f1_train": round(f1_tr, 3),
            "n_crises_reelles": int(ev.n_crises_reelles.sum()),
            "n_detections": int(ev.n_detections.sum()),
            "tp": int(ev.tp.sum()), "fp": int(ev.fp.sum()), "fn": int(ev.fn.sum()),
            "heures": sum(durations.values()) / 3600.0,
            "latence_moy_s": float(np.nanmean(ev.latence_moy_s)),
        })
    return pd.DataFrame(lignes)
