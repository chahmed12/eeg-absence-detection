"""Étape 5d — poids des bandes de fréquences (avant / après filtrage).

Répond directement à la question « quelle bande est importante ? ». On agrège
l'information mutuelle (ou l'importance) des features de puissance par bande :

* **avant** : MI calculée sur le signal **brut** (tous canaux) — conservée dans
  ``mutual_info_raw.csv`` avant le recalcul ;
* **après** : MI calculée sur le signal **filtré** (EEG seul).

La comparaison montre si le filtrage fait basculer le poids vers la bande
attendu cliniquement (Delta/Theta ~3 Hz, décharge pointe-onde de l'absence).
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, viz

pd.set_option("display.width", 200)

ORDER = ["Delta", "Theta", "Alpha", "Beta"]


def _band(name: str) -> str | None:
    m = re.search(r"__(pow|rel)_(Delta|Theta|Alpha|Beta)", str(name))
    return m.group(2) if m else None


def _aggregate(path) -> dict[str, float]:
    s = pd.read_csv(path, index_col=0).iloc[:, 0]
    out: dict[str, float] = {}
    for f, v in s.items():
        b = _band(f)
        if b:
            out[b] = out.get(b, 0.0) + float(v)
    return out


def main() -> None:
    viz.set_report_style()
    new_path = config.CACHE_DIR / "mutual_info.csv"
    old_path = config.CACHE_DIR / "mutual_info_raw.csv"

    bands_new = _aggregate(new_path)
    tab = pd.DataFrame({"MI_apres_filtrage": pd.Series(bands_new)}).reindex(ORDER).round(3)
    if old_path.exists():
        bands_old = _aggregate(old_path)
        tab.insert(0, "MI_avant_filtrage", pd.Series(bands_old).reindex(ORDER).round(3))
    tab.to_csv(config.CACHE_DIR / "band_importance.csv")
    print("=== MI cumulée par bande ===")
    print(tab.to_string())

    fig = viz.plot_band_importance(
        bands_new, title="MI cumulée par bande — après filtrage (EEG seul)",
        xlabel="Information mutuelle cumulée")
    viz.save_fig(fig, "06_band_importance")
    plt.close(fig)

    if old_path.exists():
        x = np.arange(len(ORDER))
        w = 0.38
        fig, ax = plt.subplots(figsize=(viz._fw(8), viz._fh(4)))
        ax.bar(x - w / 2, [bands_old.get(b, 0.0) for b in ORDER], w,
               label="Avant (brut, tous canaux)")
        ax.bar(x + w / 2, [bands_new.get(b, 0.0) for b in ORDER], w,
               label="Après (filtré, EEG seul)")
        ax.set_xticks(x); ax.set_xticklabels(ORDER)
        ax.set_ylabel("Information mutuelle cumulée")
        ax.set_title("Poids des bandes de fréquences : avant vs après filtrage")
        ax.legend(); ax.grid(axis="y", alpha=0.4)
        fig.tight_layout()
        viz.save_fig(fig, "06_band_importance_before_after")
        plt.close(fig)
        print("\nFigure : report/figures/06_band_importance_before_after.png")


if __name__ == "__main__":
    main()
