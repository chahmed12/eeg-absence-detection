"""Entrées/sorties du projet : lecture des signaux .mat (avec cache disque) et
lecture des annotations des crises (avec audit qualité).

Le pipeline manipule 19 patients x ~1575 s x 256 Hz x 23 canaux : il est
impératif de ne convertir chaque .mat qu'une seule fois. `load_patient_signal`
met donc en cache le signal converti (npy float32 + métadonnées JSON) et le
recharge ensuite en quelques millisecondes.
"""

from __future__ import annotations

import io as _io
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

from . import config


# ---------------------------------------------------------------------------
# 1. Lecture brute d'un fichier .mat « timetable » MATLAB (fonction fournie)
# ---------------------------------------------------------------------------
def load_timetable(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Lit un .mat contenant un ``timetable`` MATLAB (objet MCOS) sans MATLAB.

    Chaque ligne du timetable = 1 s ; chaque cellule = 256 échantillons
    (fs = 256 Hz). Cette fonction est reprise telle quelle du notebook initial.

    Returns:
        (temps_par_ligne_en_s, noms_canaux, signal[n_echantillons, n_canaux]).
    """
    m = sio.loadmat(path)
    fw = bytes(m["__function_workspace__"][0])
    raw = b"MATLAB 5.0 MAT-file".ljust(124, b" ") + fw[:4] + fw[8:]
    r = MatFile5Reader(_io.BytesIO(raw), struct_as_record=False, squeeze_me=False)
    r.mat_stream.seek(128)
    r.initialize_read()
    hdr, _ = r.read_var_header()
    root = r.read_var_array(hdr, process=False)
    cells = root[0, 0].MCOS[0]["arr"]
    row_ms = np.ravel(cells[2][0])                  # rowTimes en millisecondes
    tt = cells[4][0][0, 0]                          # objet timetable
    names = [str(n[0]) for n in tt.varNames[0]]     # noms des canaux
    data = tt.data[0]                               # une cellule par canal
    n_rows = len(row_ms)
    sig = np.stack(
        [np.concatenate([np.ravel(data[c][i, 0]) for i in range(n_rows)])
         for c in range(len(data))],
        axis=1,
    )                                               # (n_echantillons, n_canaux)
    return row_ms / 1000.0, names, sig


# ---------------------------------------------------------------------------
# 2. Cache disque des signaux convertis
# ---------------------------------------------------------------------------
class SignalRecord(NamedTuple):
    """Signal d'un patient et ses métadonnées temporelles."""

    t0: np.ndarray          # début de chaque ligne du timetable (s)
    names: list[str]        # noms des canaux
    sig: np.ndarray         # signal (n_echantillons, n_canaux), float32
    fs: float               # fréquence d'échantillonnage mesurée (Hz)


def _cache_paths(stem: str) -> tuple[Path, Path]:
    """Renvoie les chemins (npy, meta.json) du cache pour un fichier donné.

    La version de pré-traitement (``config.SIGNAL_CACHE_VERSION``) est incluse
    dans le nom : changer le filtrage invalide donc automatiquement l'ancien
    cache au lieu de le réutiliser silencieusement.
    """
    ver = config.SIGNAL_CACHE_VERSION
    return (
        config.CACHE_DIR / f"{stem}.{ver}.npy",
        config.CACHE_DIR / f"{stem}.{ver}.meta.json",
    )


def load_patient_signal(mat_path: str | Path, use_cache: bool = True) -> SignalRecord:
    """Charge le signal d'un patient, en utilisant/écrivant le cache disque.

    Le signal est **pré-traité** (passe-bande 0,5–40 Hz + notch secteur) avant
    d'être mis en cache : toutes les étapes en aval (figures, features,
    modèles) travaillent donc sur le même signal filtré.

    Args:
        mat_path: chemin du fichier .mat.
        use_cache: si True, réutilise le cache s'il existe déjà.

    Returns:
        Un :class:`SignalRecord` (signal float32 + métadonnées + fs mesurée).
    """
    mat_path = Path(mat_path)
    stem = mat_path.stem
    npy_path, meta_path = _cache_paths(stem)
    config.ensure_dirs()

    if use_cache and npy_path.exists() and meta_path.exists():
        sig = np.load(npy_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_version") == config.SIGNAL_CACHE_VERSION:
            return SignalRecord(
                t0=np.asarray(meta["t0"], dtype=np.float64),
                names=list(meta["names"]),
                sig=sig,
                fs=float(meta["fs"]),
            )

    t0, names, sig = load_timetable(mat_path)
    sig = np.ascontiguousarray(sig, dtype=np.float32)
    fs = measure_fs(t0, sig)

    from .preprocess import preprocess_signal

    sig = preprocess_signal(sig, fs=fs)

    np.save(npy_path, sig)
    meta_path.write_text(
        json.dumps(
            {
                "names": names,
                "t0": np.asarray(t0, dtype=np.float64).tolist(),
                "fs": fs,
                "cache_version": config.SIGNAL_CACHE_VERSION,
                "preprocess_enabled": config.PREPROCESS_ENABLED,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return SignalRecord(t0=t0, names=names, sig=sig, fs=fs)


def measure_fs(t0: np.ndarray, sig: np.ndarray) -> float:
    """Re-mesure la fréquence d'échantillonnage à partir des données réelles.

    fs = (n_échantillons / n_lignes) / durée_médiane_d_une_ligne.
    """
    n_lignes = len(t0)
    duree_ligne = float(np.median(np.diff(t0)))
    return (sig.shape[0] / n_lignes) / duree_ligne


# ---------------------------------------------------------------------------
# 3. Correspondance fichier .mat  <->  identifiant patient
# ---------------------------------------------------------------------------
def patient_id_from_stem(stem: str) -> str:
    """Convertit un nom de fichier en identifiant d'annotation.

    « 190304A_E_0000d » -> « 190304A-E » (correspondance EXACTE par
    normalisation, jamais par ``str.contains``).
    """
    tokens = stem.split("_")
    if len(tokens) >= 2:
        return f"{tokens[0]}-{tokens[1]}"
    return tokens[0]


def list_mat_files(data_dir: Path | None = None) -> list[Path]:
    """Liste les fichiers de signal disponibles, triés par identifiant patient."""
    data_dir = data_dir or config.DATA_DIR
    files = sorted(data_dir.glob(config.MAT_GLOB))
    return sorted(files, key=lambda p: patient_id_from_stem(p.stem))


# ---------------------------------------------------------------------------
# 4. Lecture des annotations + audit qualité
# ---------------------------------------------------------------------------
class AnnotationAudit(NamedTuple):
    """Résultats de l'audit qualité des annotations."""

    duplicated_patients: list[str]          # identifiants apparaissant >1 fois
    suspect_seizures: pd.DataFrame          # crises de durée aberrante (exclues)
    nombrece_incoherences: list[str]        # NombreCE annoncé != crises lues
    n_annotation_patients: int              # patients présents dans l'Excel
    n_seizures_retained: int                # crises conservées après exclusions
    duplicate_conflicts: pd.DataFrame       # détail des doublons NON identiques


def _find_debut_column(row: pd.Series, i: int) -> str | None:
    """Nom de la colonne de début de la crise i (3 orthographes possibles)."""
    for nom in (f"DebCE{i}", f"DebutCE{i}", f"DebtCE{i}"):
        if nom in row.index:
            return nom
    return None


def _value_eq(a: Any, b: Any) -> bool:
    """Égalité robuste aux valeurs manquantes (deux NaN sont considérés égaux)."""
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(a == b)


def _duplicate_conflicts(ann: pd.DataFrame) -> pd.DataFrame:
    """Décrit les patients présents plusieurs fois et si leurs lignes divergent.

    Un identifiant présent plusieurs fois recouvre deux cas très différents :

    - *duplication stricte* : les lignes sont identiques (vrai doublon) ;
    - *conflit de saisie* : deux annotations de la MÊME session (même
      ``DebutTrace``, temps de crise quasi superposables) dont quelques champs
      diffèrent (bornes de crise, voire l'âge). Ce n'est **pas** un doublon de
      contenu : la règle « on garde la première ligne » masque alors un choix.

    Cette fonction ne modifie rien : elle expose le détail pour l'audit, tandis
    que ``load_annotations`` continue de ne conserver que la première ligne.

    Returns:
        DataFrame (une ligne par patient dupliqué) :
        ``Patient, N_Lignes, Lignes_Identiques, Champs_Divergents, Ecart_Max_S``.
    """
    colonnes = [c for c in ann.columns if c != "Unnamed: 0"]
    colonnes_temps = [c for c in ann.columns if re.match(r"^(Deb|Fin)", str(c))]

    dup_ids = ann.loc[
        ann.duplicated(subset="Unnamed: 0", keep=False), "Unnamed: 0"
    ].unique()

    lignes: list[dict[str, Any]] = []
    for pid in dup_ids:
        sub = ann[ann["Unnamed: 0"] == pid]
        ref = sub.iloc[0]
        divergents: list[str] = []
        ecart_max = 0.0
        for _, autre in sub.iloc[1:].iterrows():
            for col in colonnes:
                if _value_eq(ref[col], autre[col]):
                    continue
                if col not in divergents:
                    divergents.append(col)
                if col in colonnes_temps:
                    ecart_max = max(ecart_max, abs(float(autre[col]) - float(ref[col])) * 86400.0)
        lignes.append(
            {
                "Patient": pid,
                "N_Lignes": int(len(sub)),
                "Lignes_Identiques": len(divergents) == 0,
                "Champs_Divergents": ", ".join(divergents),
                "Ecart_Max_S": round(ecart_max, 3),
            }
        )
    return pd.DataFrame(
        lignes,
        columns=["Patient", "N_Lignes", "Lignes_Identiques", "Champs_Divergents", "Ecart_Max_S"],
    )


def load_annotations(
    path: str | Path | None = None,
    max_duration_s: float | None = None,
    keep_suspects: bool = False,
) -> tuple[pd.DataFrame, AnnotationAudit]:
    """Charge et nettoie les annotations, puis renvoie le tableau des crises.

    Colonnes renvoyées : ``Patient, Num_Crise, Debut_Sec, Fin_Sec, Duree_Sec``
    (secondes relatives au début du tracé). Un audit qualité complet est
    également produit (doublons, crises suspectes, incohérences NombreCE).

    Args:
        path: chemin du fichier Excel (défaut : config.ANNOTATIONS_FILE).
        max_duration_s: seuil de durée au-delà duquel une crise est suspecte.
        keep_suspects: si True, les crises suspectes sont conservées.

    Returns:
        (df_all_seizures, audit).
    """
    path = Path(path or config.ANNOTATIONS_FILE)
    max_duration_s = config.MAX_SEIZURE_DURATION_S if max_duration_s is None else max_duration_s

    ann = pd.read_excel(path)
    ann = ann.dropna(how="all")                       # supprime ~1 M de lignes vides
    ann = ann.dropna(subset=["Unnamed: 0"]).copy()    # garde les lignes avec patient
    ann["Unnamed: 0"] = ann["Unnamed: 0"].astype(str).str.strip()

    # --- Doublons de patients : on conserve la première ligne ---------------
    # Un même identifiant peut apparaître deux fois (re-saisie de la même
    # session) ; les lignes ne sont alors PAS forcément identiques. On garde la
    # première ligne et `_duplicate_conflicts` documente les divergences.
    duplicate_conflicts = _duplicate_conflicts(ann)
    doublons = (
        ann[ann.duplicated(subset="Unnamed: 0", keep="first")]["Unnamed: 0"]
        .astype(str)
        .tolist()
    )
    ann = ann.drop_duplicates(subset="Unnamed: 0", keep="first")

    # --- Parcours des patients et de leurs 12 crises potentielles -----------
    all_seizures: list[dict[str, Any]] = []
    incoherences: list[str] = []

    for _, row in ann.iterrows():
        patient_id = row["Unnamed: 0"]
        debut_trace_jours = row["DebutTrace"]
        if pd.isna(debut_trace_jours):
            continue

        nb_lues = 0
        for i in range(1, 13):
            col_deb = _find_debut_column(row, i)
            col_fin = f"FinCE{i}"
            if col_deb is None or col_fin not in row.index:
                continue
            val_deb, val_fin = row[col_deb], row[col_fin]
            if pd.notna(val_deb) and pd.notna(val_fin):
                deb_sec = (val_deb - debut_trace_jours) * 86400.0
                fin_sec = (val_fin - debut_trace_jours) * 86400.0
                all_seizures.append(
                    {
                        "Patient": patient_id,
                        "Num_Crise": i,
                        "Debut_Sec": round(deb_sec, 2),
                        "Fin_Sec": round(fin_sec, 2),
                        "Duree_Sec": round(fin_sec - deb_sec, 2),
                    }
                )
                nb_lues += 1

        annonce = row.get("NombreCE")
        if pd.notna(annonce) and int(annonce) != nb_lues:
            incoherences.append(
                f"{patient_id} : {int(annonce)} crise(s) annoncée(s), {nb_lues} lue(s)"
            )

    df = pd.DataFrame(all_seizures)

    # --- Crises suspectes (durée > seuil ou <= 0) ---------------------------
    mask_suspect = (df["Duree_Sec"] > max_duration_s) | (df["Duree_Sec"] <= 0)
    suspectes = df[mask_suspect].reset_index(drop=True)
    if not keep_suspects:
        df = df[~mask_suspect].reset_index(drop=True)

    audit = AnnotationAudit(
        duplicated_patients=doublons,
        suspect_seizures=suspectes,
        nombrece_incoherences=incoherences,
        n_annotation_patients=int(ann["Unnamed: 0"].nunique()),
        n_seizures_retained=len(df),
        duplicate_conflicts=duplicate_conflicts,
    )
    return df, audit


# ---------------------------------------------------------------------------
# 5. Registre patient : fichier .mat <-> nombre de crises
# ---------------------------------------------------------------------------
def build_patient_registry(
    seizures: pd.DataFrame,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Construit le tableau récapitulatif patient x nombre de crises x fichier.

    La correspondance fichier/annotation est faite par normalisation exacte de
    l'identifiant (patient_id_from_stem). Les éventuels désappariements sont
    signalés par colonnes booléennes.

    Returns:
        DataFrame trié : Patient, Fichier, N_Crises, Duree_Totale_S, Mat_Ok, Ann_Ok.
    """
    files = list_mat_files(data_dir)
    file_map = {patient_id_from_stem(p.stem): p.name for p in files}
    ann_patients = set(seizures["Patient"].astype(str).unique())

    lignes = []
    for pid, fname in file_map.items():
        sub = seizures[seizures["Patient"].astype(str) == pid]
        lignes.append(
            {
                "Patient": pid,
                "Fichier": fname,
                "N_Crises": int(len(sub)),
                "Duree_Totale_S": float(sub["Duree_Sec"].sum()) if len(sub) else 0.0,
                "Mat_Ok": True,
                "Ann_Ok": pid in ann_patients,
            }
        )
    reg = pd.DataFrame(lignes)

    # Patients annotés mais sans fichier .mat correspondant.
    for pid in sorted(ann_patients - set(file_map)):
        reg.loc[len(reg)] = {
            "Patient": pid,
            "Fichier": None,
            "N_Crises": int((seizures["Patient"].astype(str) == pid).sum()),
            "Duree_Totale_S": float(
                seizures.loc[seizures["Patient"].astype(str) == pid, "Duree_Sec"].sum()
            ),
            "Mat_Ok": False,
            "Ann_Ok": True,
        }
    return reg.sort_values("Patient").reset_index(drop=True)
