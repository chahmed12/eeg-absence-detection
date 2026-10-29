"""Configuration centralisée du projet de détection des crises d'absence.

Tous les paramètres du pipeline sont regroupés ici : chemins, fréquence
d'échantillonnage, fenêtrage, bandes de fréquences, seuils de qualité, graine
aléatoire. Aucun chemin n'est codé en dur ailleurs dans le code : ce module est
le point unique de vérité et peut être surchargé par variables
d'environnement (voir README), ce qui permet de déplacer le projet sur une
autre machine sans rien modifier dans le code.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Chemins du projet
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Dossier contenant les fichiers .mat et le fichier d'annotations.
#: Surchargeable via la variable d'environnement EEG_DATA_DIR.
_DEFAULT_DATA_DIR: str = r"C:\Users\MLAB\CRISES ABSENCES"
DATA_DIR: Path = Path(os.environ.get("EEG_DATA_DIR", _DEFAULT_DATA_DIR))

#: Dossiers de sortie (cache des conversions, figures du rapport).
CACHE_DIR: Path = Path(os.environ.get("EEG_CACHE_DIR", PROJECT_ROOT / "cache"))
REPORT_DIR: Path = Path(os.environ.get("EEG_REPORT_DIR", PROJECT_ROOT / "report"))
FIGURES_DIR: Path = REPORT_DIR / "figures"

#: Fichier d'annotations des crises (colonnes DebutTrace, DebCE{i}/FinCE{i}...).
ANNOTATIONS_FILE: Path = DATA_DIR / "Annotations MAJ.xlsx"

#: Artefacts mis en cache (dataset de features, prédictions, modèles).
FEATURES_PARQUET: Path = CACHE_DIR / "features.parquet"
PREDICTIONS_PARQUET: Path = CACHE_DIR / "predictions.parquet"
MODELS_DIR: Path = CACHE_DIR / "models"

#: Motif des fichiers de signal à charger (les fichiers « ...0000h.mat » ne
#: contiennent pas le signal complet, ils sont ignorés).
MAT_GLOB: str = "*_0000d.mat"


# ---------------------------------------------------------------------------
# 2. Signal et fenêtrage
# ---------------------------------------------------------------------------
#: Fréquence d'échantillonnage attendue (elle est de toute façon re-mesurée).
FS: float = 256.0

#: Taille de la fenêtre d'analyse en secondes (512 échantillons à 256 Hz).
WINDOW_SEC: float = 2.0

#: Recouvrement entre fenêtres consécutives (0.5 = 50 %).
OVERLAP: float = 0.5

#: Une fenêtre est étiquetée « crise » si au moins cette proportion de ses
#: échantillons appartient à une crise annotée.
LABEL_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# 2b. Pré-traitement du signal (filtrage)
# ---------------------------------------------------------------------------
#: Activer le filtrage (passe-bande + réjection secteur) au chargement.
#: Le signal brut contient un offset DC massif (énergie brute ~1e8, soit des
#: amplitudes non physiologiques) qu'il faut retirer AVANT toute extraction.
PREPROCESS_ENABLED: bool = True

#: Bande passante EEG (Hz) : borne basse (retire l'offset DC et la dérive
#: lente) et borne haute (retire le bruit musculaire haute fréquence). La
#: décharge pointe-onde de l'absence (~3 Hz) est entièrement préservée.
EEG_LOWCUT: float = 0.5
EEG_HIGHCUT: float = 40.0

#: Fréquence du secteur (Hz) à rejeter : 50 Hz en Europe, 60 Hz aux États-Unis.
NOTCH_FREQ: float = 50.0
#: Facteur de qualité du filtre coupe-bande (plus grand = bande plus étroite).
NOTCH_Q: float = 30.0
#: Ordre du filtre Butterworth passe-bande (appliqué en phase nulle/filtfilt).
FILTER_ORDER: int = 4

#: Version du cache signal : à incrémenter si le pré-traitement change, pour
#: invalider les ``.npy`` déjà convertis (sinon un ancien cache non filtré
#: serait réutilisé silencieusement).
SIGNAL_CACHE_VERSION: str = "v2_filtered"


# ---------------------------------------------------------------------------
# 3. Bandes de fréquences EEG (Hz)
# ---------------------------------------------------------------------------
#: Bandes classiques du sujet. La décharge pointe-onde typique de l'absence
#: se situe autour de 3 Hz (bande Delta/Theta).
BANDS: dict[str, tuple[float, float]] = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
}

#: Canaux non-EEG présents dans les enregistrements (myogramme, ligne de
#: stimulation, électrocardiogramme). Les noms varient selon les fichiers
#: (« EMG1 »/« EMG_1_ », etc.), d'où les deux orthographes ci-dessous.
NON_EEG_CHANNELS: frozenset[str] = frozenset(
    {"EMG1", "EMG2", "EMG_1_", "EMG_2", "SLI", "ECG"}
)


#: Canaux EEG standard 10-20 présents dans l'union des enregistrements (19).
CANONICAL_EEG_CHANNELS: tuple[str, ...] = (
    "EEGFp1", "EEGFp2", "EEGF7", "EEGF3", "EEGFz", "EEGF4", "EEGF8",
    "EEGT3", "EEGC3", "EEGCz", "EEGC4", "EEGT4", "EEGT5",
    "EEGP3", "EEGPz", "EEGP4", "EEGT6", "EEGO1", "EEGO2",
)

#: Canaux EEG communs à TOUS les enregistrements (montage minimal, 8 canaux).
COMMON_EEG_CHANNELS: tuple[str, ...] = (
    "EEGFp1", "EEGFp2", "EEGC3", "EEGC4", "EEGT3", "EEGT4", "EEGO1", "EEGO2",
)


def canonical_channel(name: str) -> str:
    """Normalise un nom de canal (variantes EMG_1_/EMG_2 → EMG1/EMG2)."""
    n = name.strip()
    return {"EMG_1_": "EMG1", "EMG_2": "EMG2"}.get(n, n)


def is_eeg_channel(name: str) -> bool:
    """Indique si un canal est un vrai canal EEG (préfixe « EEG »)."""
    return name.upper().startswith("EEG")


def split_channels(names: list[str]) -> tuple[list[str], list[str]]:
    """Sépare une liste de canaux en (canaux EEG, canaux non-EEG)."""
    eeg = [c for c in names if is_eeg_channel(c)]
    non_eeg = [c for c in names if not is_eeg_channel(c)]
    return eeg, non_eeg


# ---------------------------------------------------------------------------
# 4. Qualité des annotations
# ---------------------------------------------------------------------------
#: Une crise d'absence dépassant ce seuil (ou de durée <= 0) est jugée
#: incohérente (erreur de saisie probable) : elle est listée dans l'audit et
#: exclue du jeu de crises retenues.
MAX_SEIZURE_DURATION_S: float = 300.0

#: Tolérance (s) considérée comme le bruit d'annotation humaine pour la
#: comparaison des crises détectées aux crises réelles.
EVENT_TOLERANCE_S: float = 2.0


# ---------------------------------------------------------------------------
# 5. Reproductibilité
# ---------------------------------------------------------------------------
#: Graine aléatoire unique pour tous les tirages (splits, modèles, SMOTE...).
RANDOM_SEED: int = 42


def ensure_dirs() -> None:
    """Crée les dossiers de sortie s'ils n'existent pas encore."""
    for dossier in (CACHE_DIR, FIGURES_DIR):
        dossier.mkdir(parents=True, exist_ok=True)
