"""Construction de la variable cible (classe par échantillon) à partir des
annotations de crises.

Deux niveaux d'étiquetage coexistent dans le pipeline :

* **par échantillon** (ce module) : 0 = hors crise, 1 = crise — utilisé pour
  colorer le signal et pour l'étiquetage des fenêtres ;
* **par fenêtre** (module ``windowing``) : une fenêtre est « crise » si au moins
  ``config.LABEL_THRESHOLD`` de ses échantillons le sont.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

LABEL_NORMAL: int = 0
LABEL_SEIZURE: int = 1


def patient_seizures(seizures: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    """Sélectionne les crises d'un patient (égalité EXACTE d'identifiant)."""
    return seizures[seizures["Patient"].astype(str) == str(patient_id)].reset_index(drop=True)


def sample_labels(
    n_samples: int,
    fs: float,
    patient_seizures_df: pd.DataFrame,
) -> np.ndarray:
    """Construit le vecteur de classes par échantillon (uint8).

    Un échantillon appartient à la classe 1 s'il tombe dans un intervalle de
    crise annotée ``[Debut_Sec, Fin_Sec]`` (bornes incluses, comme dans le
    notebook initial).

    Args:
        n_samples: nombre d'échantillons total de l'enregistrement.
        fs: fréquence d'échantillonnage (Hz).
        patient_seizures_df: lignes d'annotations du patient (colonnes
            ``Debut_Sec`` et ``Fin_Sec``).

    Returns:
        Vecteur ``uint8`` de longueur ``n_samples``.
    """
    labels = np.zeros(n_samples, dtype=np.uint8)
    for _, r in patient_seizures_df.iterrows():
        start = int(round(r["Debut_Sec"] * fs))
        end = int(round(r["Fin_Sec"] * fs))
        start = max(0, start)
        end = min(n_samples - 1, end)
        if end >= start:
            labels[start:end + 1] = LABEL_SEIZURE
    return labels


def class_balance(labels: np.ndarray) -> dict[str, float]:
    """Proportions et effectifs des deux classes.

    Returns:
        dict avec ``n0, n1, pct0, pct1`` (pourcentages en %).
    """
    n = labels.size
    n1 = int(np.count_nonzero(labels == LABEL_SEIZURE))
    n0 = n - n1
    return {
        "n0": n0,
        "n1": n1,
        "pct0": 100.0 * n0 / n if n else 0.0,
        "pct1": 100.0 * n1 / n if n else 0.0,
    }
