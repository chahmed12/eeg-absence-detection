"""Script d'exploration (étape 1) — exécutable directement.

Charge les 22 enregistrements depuis le cache, produit les statistiques
globales d'exploration, la variable cible par échantillon d'un patient exemple,
et exporte les figures dans report/figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, viz
from src.io import load_annotations, list_mat_files, load_patient_signal, patient_id_from_stem
from src.labeling import class_balance, patient_seizures, sample_labels

pd.set_option("display.width", 200)


def main() -> None:
    seizures, audit = load_annotations()
    viz.set_report_style()

    # --- 1. Inventaire des enregistrements ---------------------------------
    rows = []
    for p in list_mat_files():
        pid = patient_id_from_stem(p.stem)
        r = load_patient_signal(p)
        eeg, non_eeg = config.split_channels(r.names)
        rows.append({
            "Patient": pid, "Fichier": p.name,
            "n_canaux": len(r.names), "n_EEG": len(eeg),
            "fs_mesuree": round(r.fs, 3),
            "duree_s": round(r.sig.shape[0] / r.fs, 1),
            "n_echantillons": r.sig.shape[0],
            "valeurs_manquantes": int(np.isnan(r.sig).sum()),
            "N_crises": int((seizures["Patient"].astype(str) == pid).sum()),
        })
    inv = pd.DataFrame(rows)
    print("=== EXPLORATION : inventaire des enregistrements ===")
    print(inv.to_string(index=False))
    print()
    print("fs mesurée :", sorted(inv["fs_mesuree"].unique()))
    print("Durées (s) : min %.0f | médiane %.0f | max %.0f | total %.0f"
          % (inv.duree_s.min(), inv.duree_s.median(), inv.duree_s.max(), inv.duree_s.sum()))
    print("Montages : ", inv.groupby("n_canaux")["Patient"].apply(list).to_dict())
    print("Valeurs manquantes totales :", int(inv.valeurs_manquantes.sum()))

    # --- 2. Variable cible par échantillon (patient exemple) ---------------
    pid_ex = "210914B-A"
    r = load_patient_signal(next(p for p in list_mat_files()
                                 if patient_id_from_stem(p.stem) == pid_ex))
    t = np.arange(r.sig.shape[0]) / r.fs
    sez = patient_seizures(seizures, pid_ex)
    y = sample_labels(r.sig.shape[0], r.fs, sez)
    bal = class_balance(y)
    print("\n=== VARIABLE CIBLE (patient %s) ===" % pid_ex)
    print(f"Échantillons : {y.size} | classe 0 : {bal['pct0']:.2f}% | "
          f"classe 1 : {bal['pct1']:.2f}%  (déséquilibre -> métriques PR/F1)")
    print(f"Durée totale : {r.sig.shape[0]/r.fs:.0f} s | crises : {len(sez)}")

    # --- 3. Statistiques globales des crises -------------------------------
    print("\n=== STATISTIQUES GLOBALES DES CRISES ===")
    print(f"Patients annotés : {seizures['Patient'].nunique()} | crises : {len(seizures)}")
    print("Durées : moy %.2f s | méd %.1f | min %.1f | max %.1f | CV %.3f"
          % (seizures.Duree_Sec.mean(), seizures.Duree_Sec.median(),
             seizures.Duree_Sec.min(), seizures.Duree_Sec.max(),
             seizures.Duree_Sec.std() / seizures.Duree_Sec.mean()))
    inter = viz.inter_crisis_intervals(seizures)
    print(f"Intervalles inter-crises : {len(inter)} | moy {inter.mean():.1f} s | "
          f"méd {np.median(inter):.1f} s")

    # --- 4. Figures --------------------------------------------------------
    durations = dict(zip(inv.Patient, inv.duree_s))
    chan_ex = [c for c in r.names if c.startswith("EEG")][:8]

    figures = {}
    # (a) vue multi-canaux autour de la 1re crise (zoom ±6 s)
    c0 = sez.iloc[0]
    sedebut, sefin = c0["Debut_Sec"], c0["Fin_Sec"]
    figures["01_multichannel_zoom"] = viz.plot_multichannel_stack(
        t, r.sig, r.names, sez, channels=chan_ex,
        t_min=max(0, sedebut - 6), t_max=sefin + 6, fs=r.fs,
        title=f"Patient {pid_ex} — vue EEG multi-canaux autour d'une crise")
    # (a') même zoom, canaux ramenés à une amplitude unitaire : la synchronie
    # de la décharge ressort quel que soit le gain respectif des électrodes.
    figures["01_multichannel_zoom_corrected"] = viz.plot_multichannel_stack(
        t, r.sig, r.names, sez, channels=chan_ex,
        t_min=max(0, sedebut - 6), t_max=sefin + 6, fs=r.fs, normalize=True,
        title=f"Patient {pid_ex} — zoom multi-canaux normalisé (synchronie)")
    # (b) aperçu signal complet
    figures["01_signal_overview"] = viz.plot_signal_overview(
        t, r.sig, r.names, "EEGFp1", sez, r.fs,
        title=f"Patient {pid_ex} — EEG Fp1 sur tout l'enregistrement")
    # (c,d) heatmap + spectrogramme : illustrés sur un cas à décharge FRANCHE.
    # Le patient exemple 210914B-A a une décharge très subtile sur ces
    # représentations (rapport de puissance 3 Hz crise/fond ~1,7, l'un des plus
    # faibles de la cohorte) : la signature se noie dans le fond 1/f. On utilise
    # donc 210427B-C (rapport ~6,9, crise de 18 s) pour montrer nettement la
    # signature spectrale (bloc 3 Hz) et la synchronie inter-canaux.
    pid_clear = "210427B-C"
    rc = load_patient_signal(next(p for p in list_mat_files()
                                  if patient_id_from_stem(p.stem) == pid_clear))
    sezc = patient_seizures(seizures, pid_clear)
    zc = sezc.assign(_dur=sezc["Fin_Sec"] - sezc["Debut_Sec"]).sort_values("_dur").iloc[-1]
    pad = max(20.0, 0.5 * float(zc["Fin_Sec"] - zc["Debut_Sec"]))
    zc_min = max(0.0, float(zc["Debut_Sec"]) - pad)
    zc_max = float(zc["Fin_Sec"]) + pad
    zc_lbl = "%d–%d s" % (round(zc["Debut_Sec"]), round(zc["Fin_Sec"]))
    figures["01_heatmap"] = viz.plot_band_power_heatmap(
        rc.sig, rc.names, rc.fs, sezc, band=(2.5, 3.5), bin_sec=0.5,
        t_min=zc_min, t_max=zc_max,
        title=f"Patient {pid_clear} — puissance 3 Hz par canal, crise {zc_lbl}")
    figures["01_spectrogram"] = viz.plot_seizure_spectrogram(
        rc.sig, rc.names, "EEGCz", rc.fs, seizures=sezc,
        t_min=zc_min, t_max=zc_max, fmax=8.0, ref_hz=3.0,
        title=f"Patient {pid_clear} — signal Cz et spectrogramme, crise {zc_lbl}")
    # (e) frise cohorte
    figures["01_cohort_timeline"] = viz.plot_cohort_timeline(durations, seizures)
    # (f) durées par patient
    figures["01_duration_violin"] = viz.plot_duration_violin(seizures)
    # (g) histogramme intervalles
    figures["01_interval_hist"] = viz.plot_interval_hist(inter)

    print("\n=== FIGURES EXPORTÉES ===")
    # spectrogramme et heatmap : exportées aussi en PDF vectoriel (netteté
    # parfaite à l'insertion, sans pixels).
    vector_names = {"01_spectrogram", "01_heatmap"}
    import matplotlib.pyplot as _plt
    for name, fig in figures.items():
        path = viz.save_fig(fig, name)
        if name in vector_names:
            viz.save_fig(fig, name, ext="pdf")
        _plt.close(fig)
        print(" -", path)


if __name__ == "__main__":
    main()
