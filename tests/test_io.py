"""Tests I/O et annotations.

Les tests sont séparés en deux catégories :
1. Tests unitaires synthétiques (exécutés partout, y compris en CI GitHub Actions sans données).
2. Tests d'intégration sur les données réelles (exécutés uniquement si les fichiers cliniques locaux sont présents).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src import config, io


# Condition pour ignorer gracieusement les tests d'intégration en CI (données médicales non suivies par Git)
HAS_REAL_DATA = config.ANNOTATIONS_FILE.exists()
skip_if_no_real_data = pytest.mark.skipif(
    not HAS_REAL_DATA,
    reason="Fichiers de données cliniques réels absents (normal en environnement CI/GitHub Actions)",
)


# ---------------------------------------------------------------------------
# Tests unitaires synthétiques (CI-friendly)
# ---------------------------------------------------------------------------
def test_load_annotations_synthetic(tmp_path):
    """Vérifie le parsing Excel et l'audit qualité sur un fichier d'annotation synthétique."""
    excel_path = tmp_path / "Annotations_Mock.xlsx"

    # Création d'un mini DataFrame compatible avec le format du classeur
    t0_days = 44000.0  # Timestamp approximatif en jours Excel
    mock_data = pd.DataFrame([
        {
            "Unnamed: 0": "P01-A",
            "DebutTrace": t0_days,
            "NombreCE": 1,
            "DebCE1": t0_days + 10.0 / 86400.0,
            "FinCE1": t0_days + 25.0 / 86400.0,
        },
        {
            "Unnamed: 0": "P02-B",
            "DebutTrace": t0_days,
            "NombreCE": 2,
            "DebCE1": t0_days + 30.0 / 86400.0,
            "FinCE1": t0_days + 45.0 / 86400.0,
            "DebCE2": t0_days + 100.0 / 86400.0,
            "FinCE2": t0_days + 110.0 / 86400.0,
        },
    ])
    mock_data.to_excel(excel_path, index=False)

    seizures, audit = io.load_annotations(path=excel_path)

    assert isinstance(seizures, pd.DataFrame)
    assert len(seizures) == 3
    assert set(seizures.columns) >= {"Patient", "Num_Crise", "Debut_Sec", "Fin_Sec", "Duree_Sec"}
    assert audit.n_annotation_patients == 2
    assert audit.n_seizures_retained == 3

    # Vérification des calculs de durées relatives (secondes)
    p01 = seizures[seizures["Patient"] == "P01-A"].iloc[0]
    assert abs(p01["Debut_Sec"] - 10.0) < 0.1
    assert abs(p01["Fin_Sec"] - 25.0) < 0.1
    assert abs(p01["Duree_Sec"] - 15.0) < 0.1


def test_build_patient_registry_synthetic(tmp_path):
    """Vérifie l'association patient/fichier .mat sans nécessiter les vrais signaux."""
    # Création de fichiers .mat factices
    (tmp_path / "210406A_F_0000d.mat").touch()
    (tmp_path / "210427B_C_0000d.mat").touch()

    seizures_mock = pd.DataFrame([
        {"Patient": "210406A-F", "Duree_Sec": 8.0},
        {"Patient": "210427B-C", "Duree_Sec": 18.0},
        {"Patient": "ORPHAN-PATIENT", "Duree_Sec": 5.0},  # Patient annoté sans .mat
    ])

    reg = io.build_patient_registry(seizures_mock, data_dir=tmp_path)
    assert isinstance(reg, pd.DataFrame)
    assert set(reg.columns) >= {"Patient", "Fichier", "N_Crises", "Duree_Totale_S", "Mat_Ok", "Ann_Ok"}
    assert len(reg) == 3

    p1 = reg[reg["Patient"] == "210406A-F"].iloc[0]
    assert bool(p1["Mat_Ok"]) is True
    assert bool(p1["Ann_Ok"]) is True

    orphan = reg[reg["Patient"] == "ORPHAN-PATIENT"].iloc[0]
    assert bool(orphan["Mat_Ok"]) is False
    assert bool(orphan["Ann_Ok"]) is True


# ---------------------------------------------------------------------------
# Tests d'intégration réels (exécutés si le dossier de données clinique existe)
# ---------------------------------------------------------------------------
@skip_if_no_real_data
def test_annotations_load_and_audit():
    s, audit = io.load_annotations()
    assert s.shape[0] > 0
    assert {"Patient", "Debut_Sec", "Fin_Sec", "Duree_Sec"} <= set(s.columns)
    assert audit.n_seizures_retained == len(s)
    assert audit.n_annotation_patients == s.Patient.nunique()


@skip_if_no_real_data
def test_build_registry_matches_files():
    s, _ = io.load_annotations()
    reg = io.build_patient_registry(s)
    assert len(reg) == len(io.list_mat_files())
    assert {"Patient", "Fichier", "N_Crises", "Ann_Ok"} <= set(reg.columns)


@skip_if_no_real_data
def test_duplicate_patients_are_session_conflicts_not_strict():
    """Les « doublons » sont deux saisies divergentes de la même session."""
    s, audit = io.load_annotations()
    assert set(audit.duplicated_patients) == {"230406B-A", "230515B-G"}

    conf = audit.duplicate_conflicts.set_index("Patient")
    assert set(conf.index) == {"230406B-A", "230515B-G"}
    assert (conf["N_Lignes"] == 2).all()
    assert not conf["Lignes_Identiques"].any()
    assert (conf["Ecart_Max_S"] > 0).all()
    assert "DebCE2" in conf.loc["230515B-G", "Champs_Divergents"]
    assert "Age" in conf.loc["230515B-G", "Champs_Divergents"]

    a = s[s.Patient == "230406B-A"].sort_values("Num_Crise")
    assert list(a.Duree_Sec.round(0)) == [9.0, 11.0, 10.0, 14.0]
    b = s[s.Patient == "230515B-G"].sort_values("Num_Crise")
    assert list(b.Duree_Sec.round(0)) == [14.0, 32.0]
