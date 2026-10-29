"""Tests I/O et annotations (intégration sur les données réelles, mises en cache).

Note : patient_id_from_stem est déjà testé dans test_core.py — pas de doublon ici.
"""
from src import io


def test_annotations_load_and_audit():
    s, audit = io.load_annotations()
    assert s.shape[0] > 0
    assert {"Patient", "Debut_Sec", "Fin_Sec", "Duree_Sec"} <= set(s.columns)
    assert audit.n_seizures_retained == len(s)
    assert audit.n_annotation_patients == s.Patient.nunique()


def test_build_registry_matches_files():
    s, _ = io.load_annotations()
    reg = io.build_patient_registry(s)
    assert len(reg) == len(io.list_mat_files())
    assert {"Patient", "Fichier", "N_Crises", "Ann_Ok"} <= set(reg.columns)


def test_duplicate_patients_are_session_conflicts_not_strict():
    """Les « doublons » sont deux saisies divergentes de la même session.

    Vérifie que le pipeline garde bien la première ligne tout en exposant la
    divergence (durées qui diffèrent), plutôt que de traiter un doublon strict.
    """
    s, audit = io.load_annotations()
    assert set(audit.duplicated_patients) == {"230406B-A", "230515B-G"}

    conf = audit.duplicate_conflicts.set_index("Patient")
    assert set(conf.index) == {"230406B-A", "230515B-G"}
    assert (conf["N_Lignes"] == 2).all()
    # Deux annotations de la même session -> lignes NON identiques.
    assert not conf["Lignes_Identiques"].any()
    assert (conf["Ecart_Max_S"] > 0).all()
    assert "DebCE2" in conf.loc["230515B-G", "Champs_Divergents"]
    assert "Age" in conf.loc["230515B-G", "Champs_Divergents"]

    # Règle conservée : la PREMIÈRE ligne fait foi (durées en secondes entières).
    a = s[s.Patient == "230406B-A"].sort_values("Num_Crise")
    assert list(a.Duree_Sec.round(0)) == [9.0, 11.0, 10.0, 14.0]
    b = s[s.Patient == "230515B-G"].sort_values("Num_Crise")
    assert list(b.Duree_Sec.round(0)) == [14.0, 32.0]
