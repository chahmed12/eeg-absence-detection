"""Génère les 6 notebooks pédagogiques du projet (01 → 06) au format .ipynb."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def build(name, title, cells):
    if not any("from src import config" in c.source for c in cells if c.cell_type == "code"):
        cells = [SETUP] + cells
    cells = [md(f"# {title}")] + cells
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, str(NB / name))
    print("écrit", NB / name)


SETUP = code("""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # racine du projet
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import config, io, viz, models, features
from src import postprocess as pp, evaluation as ev
pd.set_option("display.width", 200)
print("Dossier données :", config.DATA_DIR)
""")

# ---------------------------------------------------------------------------
build("01_exploration.ipynb", "01. Chargement, audit et exploration", [
    md("Objectif : charger les annotations, auditer leur qualité et explorer la cohorte."),
    SETUP,
    md("## Annotations et audit"),
    code("""
seizures, audit = io.load_annotations()
print("Patients annotés :", audit.n_annotation_patients)
print("Crises retenues  :", audit.n_seizures_retained)
print("Doublons patients:", audit.duplicated_patients)
print("Conflits de saisie (doublons non identiques):")
print(audit.duplicate_conflicts)
print("Incohérences N°CE:", audit.nombrece_incoherences)
print("Crises suspectes :", audit.suspect_seizures)
seizures.head()
"""),
    md("## Registre patient × fichier"),
    code("""
registry = io.build_patient_registry(seizures)
registry
"""),
    md("## Durées des crises"),
    code("""
print(seizures.Duree_Sec.describe())
print("CV =", round(seizures.Duree_Sec.std()/seizures.Duree_Sec.mean(), 3))
fig = viz.plot_duration_violin(seizures); viz.save_fig(fig, "01_duration_violin"); plt.show()
"""),
    md("## Frise chronologique de la cohorte"),
    code("""
durations = {p.stem: io.load_patient_signal(p).sig.shape[0]/io.load_patient_signal(p).fs
             for p in io.list_mat_files()}
fig = viz.plot_cohort_timeline(durations, seizures); viz.save_fig(fig, "01_cohort_timeline"); plt.show()
"""),
])

build("02_annotations_cible.ipynb", "02. Cible par échantillon et déséquilibre", [
    md("Objectif : transformer les intervalles de crises en étiquettes échantillon/échantillon, puis par fenêtre."),
    code("""
seizures, _ = io.load_annotations()
mat = io.list_mat_files()[1]
rec = io.load_patient_signal(mat)
pid = io.patient_id_from_stem(mat.stem)
y = io.sample_labels(rec.sig.shape[0], rec.fs, seizures[seizures.Patient == pid])
from src.labeling import class_balance
print("Patient", pid, "déséquilibre :", class_balance(y))
"""),
    md("Le signal est très déséquilibré (≈ 1-5 % de crises). C'est pourquoi on privilégie le **PR-AUC** et le **F1**, et non l'accuracy."),
    code("""
fig = viz.plot_signal_overview(rec.t0, rec.names, rec.sig, rec.fs,
                               seizures[seizures.Patient == pid], title=f"{pid} — crises en rouge")
viz.save_fig(fig, "02_signal_overview"); plt.show()
"""),
])

build("03_features.ipynb", "03. Fenêtrage et extraction de features", [
    md("Objectif : découper le signal en fenêtres (2 s, recouvrement 50 %) et extraire des features par canal."),
    code("""
feats = features.build_feature_dataset()  # utilise le cache si présent
print(feats.shape)
print("Taux de fenêtres crises :", round(100*feats.label.mean(), 2), "%")
feats[["patient", "t_start", "label"]].head()
"""),
    md("Features extraites par canal : statistiques (moyenne, écart-type, percentiles), énergie de bandes "
       "(delta/theta/alpha/beta/gamma), ratios d'asymétrie, entropie spectrale, entropie de permutation "
       "et énergie en ondelettes. Les canaux manquants sont marqués par un masque."),
    code("""
import re
familles = pd.Series([re.sub(r'^EEG[A-Za-z0-9]+_', '', c) for c in feats.columns if c.startswith('EEG')])
familles.value_counts().head(20)
"""),
])

build("04_modeles.ipynb", "04. Modélisation et validation par patient", [
    md("Objectif : comparer les modèles. Le **split aléatoire** exigé par le sujet souffre d'une fuite "
       "(fenêtres recouvrantes) ; la **validation par patient** (GroupKFold / LOPO) est la référence honnête."),
    code("""
rand = pd.read_parquet(config.CACHE_DIR / "results_random_split.parquet")
rand[["model","accuracy","precision","recall","f1","roc_auc","pr_auc"]].round(3)
"""),
    code("""
lopo = pd.read_parquet(config.CACHE_DIR / "results_lopo.parquet")
lopo.groupby("model")[["f1","pr_auc","recall","precision"]].mean().round(3)
"""),
    md("Reproduction (coûteux) : exécuter `python scripts_step3_random.py` puis `python scripts_step3_lopo.py`."),
])

build("05_detection_erreurs.ipynb", "05. Détection temporelle et métriques par événement", [
    md("Objectif : transformer les probabilités par fenêtre en **événements** (lissage, seuil, hystérésis), "
       "puis mesurer détection, fausses alarmes/heure et latence."),
    code("""
preds = pd.read_parquet(config.CACHE_DIR / "predictions.parquet")
pid = sorted(preds.patient.unique())[0]
g = preds[preds.patient == pid].sort_values("t_start")
sm, intervals = pp.postprocess(g.t_start.values, g.score.values,
                               threshold=0.4, smooth_size=5, min_gap_s=2, min_dur_s=2)
print("Intervalles détectés :", intervals)
"""),
    code("""
em = pd.read_parquet(config.CACHE_DIR / "event_metrics.parquet")
em.describe().round(2)
"""),
    code("""
sweep = pd.read_parquet(config.CACHE_DIR / "threshold_sweep.parquet")
fig, ax = plt.subplots(figsize=(8,4))
for c in ["precision","recall","f1","mcc"]:
    ax.plot(sweep.threshold, sweep[c], label=c)
ax.set_xlabel("Seuil"); ax.legend(); plt.show()
"""),
    md("Courbe seuils événementiels (étape 5b) : à seuil 0,2 on détecte 86,6 % des crises pour 3,35 FA/h."),
])

build("06_ameliorations.ipynb", "06. Explicabilité et améliorations", [
    md("Objectif : sélection de features, ablation de canaux, importance, carte topographique."),
    code("""
imp = pd.read_csv(config.CACHE_DIR / "feature_importance.csv", index_col=0)["importance"]
fig = viz.plot_top_features(imp, top_n=25); viz.save_fig(fig, "05_top_features"); plt.show()
"""),
    code("""
ci = pd.read_csv(config.CACHE_DIR / "channel_importance.csv", index_col=0).iloc[:,0]
fig = viz.plot_channel_topomap(ci.to_dict()); viz.save_fig(fig, "05_channel_topomap"); plt.show()
"""),
    code("""
am = pd.read_csv(config.CACHE_DIR / "improvements.csv")
am.round(3)
"""),
    md("Conclusions : la sélection MI top-250 améliore le F1 par patient (≈ 0,74) et "
       "8 canaux EEG communs suffisent (≈ 0,69), ce qui plaide pour un dispositif léger."),
])
