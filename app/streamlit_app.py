"""Interface de démonstration Streamlit — détection des crises d'absence.

Lancement (depuis la racine du projet) ::

    streamlit run app/streamlit_app.py

Toutes les données lourdes (signaux convertis, features, prédictions) sont
lues depuis le cache et l'affichage est sous-échantillonné (min-max par pixel)
pour rester fluide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, postprocess as pp, viz
from src.evaluation import event_metrics_by_patient
from src.io import (
    list_mat_files, load_annotations, load_patient_signal, patient_id_from_stem,
)
from src.labeling import patient_seizures

st.set_page_config(page_title="Détection des crises d'absence (EEG)", layout="wide")


# ---------------------------------------------------------------------------
# Chargement (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_seizures() -> pd.DataFrame:
    s, _ = load_annotations()
    return s


@st.cache_data(show_spinner=False)
def get_features() -> pd.DataFrame:
    return pd.read_parquet(config.FEATURES_PARQUET)


@st.cache_data(show_spinner=False)
def get_predictions() -> pd.DataFrame:
    return pd.read_parquet(config.PREDICTIONS_PARQUET)


@st.cache_data(show_spinner=False)
def get_signal(mat_path: str):
    rec = load_patient_signal(mat_path)
    return rec.t0, rec.names, rec.sig, rec.fs


@st.cache_data(show_spinner=False)
def get_results(name: str) -> pd.DataFrame:
    p = config.CACHE_DIR / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def patient_files() -> dict[str, str]:
    return {patient_id_from_stem(p.stem): str(p) for p in list_mat_files()}


def downsample(t: np.ndarray, x: np.ndarray, n_pixels: int = 4000):
    """Sous-échantillonne un signal en gardant min/max par pixel (tracé fidèle)."""
    if len(x) <= n_pixels:
        return t, x
    step = len(x) // n_pixels
    n = step * n_pixels
    tt = t[:n].reshape(n_pixels, step).mean(axis=1)
    xx = x[:n].reshape(n_pixels, step)
    return tt, xx.mean(axis=1)


@st.cache_data(show_spinner=False)
def get_error_analysis(threshold: float = 0.5, smooth_size: int = 5,
                       min_dur_s: float = 2.0):
    """Analyse d'erreurs TP/FP/FN/TN — niveau fenêtre ET niveau crise (Q15)."""
    from src.evaluation import match_events

    preds = get_predictions().sort_values(["patient", "t_start"])
    sz = get_seizures()

    # --- niveau fenetre ---
    tp = fp = fn = tn = 0
    det_by_pid: dict[str, list[tuple[float, float]]] = {}
    durations: dict[str, float] = {}
    for pid, g in preds.groupby("patient"):
        t = g["t_start"].to_numpy()
        sc = pp.smooth_scores(g["score"].to_numpy(), smooth_size)
        y = g["label"].to_numpy().astype(int)
        pr = (sc >= threshold).astype(int)
        tp += int(((pr == 1) & (y == 1)).sum())
        fp += int(((pr == 1) & (y == 0)).sum())
        fn += int(((pr == 0) & (y == 1)).sum())
        tn += int(((pr == 0) & (y == 0)).sum())
        _, iv = pp.postprocess(t, g["score"].to_numpy(), threshold=threshold,
                               smooth_size=smooth_size, min_dur_s=min_dur_s)
        det_by_pid[pid] = iv
        durations[pid] = float(t.max()) + config.WINDOW_SEC

    # --- niveau crise : appariement glouton identique a match_events ---
    paires, fn_list, fp_list = [], [], []
    tp_c = fp_c = fn_c = 0
    heures = sum(durations.values()) / 3600.0
    for pid, det in det_by_pid.items():
        true = sz[sz["Patient"].astype(str) == str(pid)]
        true_events = [tuple(x) for x in true[["Debut_Sec", "Fin_Sec"]].to_numpy()]
        m = match_events(true_events, det)
        tp_c += m["tp"]; fp_c += m["fp"]; fn_c += m["fn"]
        # refaire l'appariement pour lister les erreurs
        used = set()
        for ts, te in true_events:
            best = None
            for j, (ds, de) in enumerate(det):
                if j in used:
                    continue
                if ds <= te + config.EVENT_TOLERANCE_S and de >= ts - config.EVENT_TOLERANCE_S:
                    best = j
                    break
            if best is not None:
                used.add(best)
                ds, de = det[best]
                paires.append({"patient": pid, "reel_debut": ts, "reel_fin": te,
                               "duree_reelle": te - ts, "detecte_debut": ds,
                               "detecte_fin": de, "latence_s": ds - ts,
                               "err_debut_s": abs(ds - ts)})
            else:
                fn_list.append({"patient": pid, "debut": ts, "fin": te,
                                "duree_s": te - ts})
        for j, (ds, de) in enumerate(det):
            if j not in used:
                fp_list.append({"patient": pid, "debut": ds, "fin": de,
                                "duree_s": de - ds})

    n_reel = tp_c + fn_c
    df_paires = pd.DataFrame(paires)
    durees_reelles = [r["duree_reelle"] for r in paires]
    courte = [p for p in paires if p["duree_reelle"] <= 5.0]
    longue = [p for p in paires if p["duree_reelle"] > 10.0]
    metrics = {
        "taux_detection": tp_c / n_reel if n_reel else float("nan"),
        "fausses_alarmes_h": fp_c / heures if heures else float("nan"),
        "err_debut_moy_s": df_paires["err_debut_s"].mean() if len(df_paires) else float("nan"),
        "pct_courtes": (sum(1 for p in courte) / len(courte)) if courte else float("nan"),
        "pct_longues": (sum(1 for p in longue) / len(longue)) if longue else float("nan"),
    }
    # compte par patient
    rows = []
    for pid in sorted(det_by_pid):
        true = sz[sz["Patient"].astype(str) == str(pid)]
        n_r = len(true)
        sub_p = df_paires[df_paires["patient"] == pid] if len(df_paires) else df_paires
        sub_f = pd.DataFrame(fn_list)
        sub_fp = pd.DataFrame(fp_list)
        n_fn = len(sub_f[sub_f["patient"] == pid]) if len(sub_f) else 0
        n_fp = len(sub_fp[sub_fp["patient"] == pid]) if len(sub_fp) else 0
        rows.append({"patient": pid, "reelles": n_r,
                     "TP": len(sub_p[sub_p["patient"] == pid]) if len(sub_p) else 0,
                     "FN": n_fn, "FP": n_fp,
                     "taux_detection": (n_r - n_fn) / n_r if n_r else float("nan")})
    return {"window": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "event": {"tp": tp_c, "fp": fp_c, "fn": fn_c},
            "metrics": metrics, "per_patient": pd.DataFrame(rows),
            "fn": pd.DataFrame(fn_list), "fp": pd.DataFrame(fp_list),
            "paires": df_paires}


@st.cache_data(show_spinner=False)
def get_class_comparison(eeg_only: bool = True):
    """Moyennes par classe (hors-crise 0 vs crise 1) pour toutes les features (Q10)."""
    f = get_features()
    feats = [c for c in f.columns if c not in ("patient", "label", "t_start")]
    if eeg_only:
        feats = [c for c in feats if c.startswith("EEG")]
    y = f["label"].to_numpy().astype(int)
    m0 = f.loc[y == 0, feats].mean(numeric_only=True)
    m1 = f.loc[y == 1, feats].mean(numeric_only=True)
    eps = 1e-12
    var_pct = (m1 - m0) / (m0.abs() + eps) * 100.0
    ratio = np.where(np.abs(m0) > eps, m1 / np.where(np.abs(m0) > eps, m0, np.nan),
                     np.nan)
    return m0, m1, pd.Series(var_pct, index=feats), pd.Series(ratio, index=feats)



# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------
st.sidebar.title("Détection des crises d'absence")

seizures = get_seizures()
files = patient_files()
patients = sorted(files)
patient = st.sidebar.selectbox("Patient", patients)
rec_t0, rec_names, rec_sig, rec_fs = get_signal(files[patient])

st.sidebar.metric("Patients", len(patients))
st.sidebar.metric("Crises annotées", int(len(seizures)))
st.sidebar.caption(f"Patient courant : durée {rec_sig.shape[0]/rec_fs:.0f} s, "
                   f"{len(rec_names)} canaux, fs={rec_fs:.0f} Hz")

tabs = st.tabs(["Cohorte", "Explorateur patient", "Comparaison", "Dataset & corrélations",
                 "Modèles", "Démo détection", "Explicabilité",
                 "Analyse des erreurs", "Avant / pendant la crise"])


# ---------------------------------------------------------------------------
# Onglet 1 — Cohorte
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Cohorte")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(patients))
    c2.metric("Patients annotés", seizures["Patient"].nunique())
    c3.metric("Crises", len(seizures))
    c4.metric("Durée moyenne", f"{seizures.Duree_Sec.mean():.1f} s")

    durees = {pid: get_signal(files[pid])[2].shape[0] / get_signal(files[pid])[3] for pid in patients}
    st.subheader("Frise chronologique de la cohorte")
    fig = viz.plot_cohort_timeline(durees, seizures)
    st.pyplot(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)

    colA, colB = st.columns(2)
    with colA:
        fig = viz.plot_duration_violin(seizures)
        st.pyplot(fig); plt.close(fig)
    with colB:
        inter = viz.inter_crisis_intervals(seizures)
        fig = viz.plot_interval_hist(inter)
        st.pyplot(fig); plt.close(fig)

    st.subheader("Inventaire des signaux EEG (Partie I / Q1)")
    inv = []
    for pid, fpath in files.items():
        try:
            r = get_signal(fpath)
            rec_t, rec_n, rec_s, rec_fs = r
            inv.append({"Patient": pid, "Fichier": Path(fpath).name, "N_lignes": rec_s.shape[0],
                        "N_canaux": rec_s.shape[1], "Durée (s)": rec_s.shape[0]/rec_fs, "fs (Hz)": rec_fs,
                        "Canaux EEG": sum(1 for c in rec_n if str(c).startswith("EEG"))})
        except Exception as e:
            inv.append({"Patient": pid, "Fichier": Path(fpath).name, "Erreur": str(e)})
    st.dataframe(pd.DataFrame(inv), width='stretch')


# ---------------------------------------------------------------------------
# Onglet 2 — Explorateur patient
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header(f"Explorateur — {patient}")
    eeg_chans = [c for c in rec_names if c.startswith("EEG")]
    chans = st.multiselect("Canaux", rec_names, default=eeg_chans[:4], key="expl_chans")
    dur = rec_sig.shape[0] / rec_fs
    fen = st.slider("Zoom temporel (s)", 0.0, float(dur), (max(0.0, 30.0), min(dur, 60.0)))
    t = np.arange(rec_sig.shape[0]) / rec_fs
    mask = (t >= fen[0]) & (t <= fen[1])
    sel = seizures[seizures["Patient"].astype(str) == patient]

    if chans:
        fig = go.Figure()
        for i, ch in enumerate(chans):
            c = rec_names.index(ch)
            tt, xx = downsample(t[mask], rec_sig[mask, c])
            fig.add_trace(go.Scatter(x=tt, y=xx - i * 200, name=ch, line=dict(width=1)))
        for _, r in sel.iterrows():
            if r["Fin_Sec"] >= fen[0] and r["Debut_Sec"] <= fen[1]:
                fig.add_vrect(x0=max(r["Debut_Sec"], fen[0]), x1=min(r["Fin_Sec"], fen[1]),
                              fillcolor="red", opacity=0.2, line_width=0)
        fig.update_layout(height=400, title="Signal multi-canaux (crises en rouge)",
                          xaxis_title="Temps (s)", yaxis_title="Amplitude (µV)")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Spectrogramme")
    sp_ch = st.selectbox("Canal", rec_names, index=rec_names.index("EEGCz") if "EEGCz" in rec_names else 0)
    fig = viz.plot_spectrogram(rec_sig, rec_names, sp_ch, rec_fs, fmax=20, seizures=sel)
    st.pyplot(fig); plt.close(fig)


# ---------------------------------------------------------------------------
# Onglet 3 — Comparaison
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Comparaison entre patients")
    comp_patients = st.multiselect("Patients à comparer", patients, default=patients[:3])
    comp_ch = st.selectbox("Canal commun", [c for c in config.COMMON_EEG_CHANNELS], index=0)
    fig = go.Figure()
    for i, pid in enumerate(comp_patients):
        _, names, sig, fs = get_signal(files[pid])
        if comp_ch not in names:
            continue
        t = np.arange(sig.shape[0]) / fs
        tt, xx = downsample(t, sig[:, names.index(comp_ch)])
        fig.add_trace(go.Scatter(x=tt, y=xx, name=pid, line=dict(width=0.7)))
    fig.update_layout(height=400, title=f"Canal {comp_ch} — plusieurs patients",
                      xaxis_title="Temps (s)", yaxis_title="Amplitude (µV)")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Moyenne événementielle (alignement des crises, ±30 s)")
    ev_ch = st.selectbox("Canal (événementiel)", eeg_chans, index=0, key="ev_ch")
    if st.button("Calculer la moyenne événementielle"):
        half = int(30 * rec_fs)
        curves = []
        for _, r in seizures.iterrows():
            pid = str(r["Patient"])
            if pid not in files:
                continue
            _, names, sig, fs = get_signal(files[pid])
            if ev_ch not in names:
                continue
            c = names.index(ev_ch)
            s0 = int(round(r["Debut_Sec"] * fs))
            if s0 - half < 0 or s0 + half >= sig.shape[0]:
                continue
            seg = sig[s0 - half:s0 + half, c].astype(float)
            curves.append(seg - seg.mean())
        if curves:
            M = np.mean(curves, axis=0)
            tt = np.arange(-half, half) / rec_fs
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tt, y=M, name="Moyenne"))
            fig.add_trace(go.Scatter(x=tt, y=np.std(curves, axis=0), name="Écart-type",
                                     line=dict(dash="dot")))
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=400, xaxis_title="Temps relatif au début (s)",
                              yaxis_title="Amplitude (µV)")
            st.plotly_chart(fig, width='stretch')
            st.caption(f"{len(curves)} crises alignées.")
        else:
            st.warning("Aucune crise exploitable pour ce canal.")


# ---------------------------------------------------------------------------
# Onglet 4 — Dataset & corrélations
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Dataset — fenêtres, classes, corrélations")
    fdf = get_features()
    feats = [c for c in fdf.columns if c not in ("patient", "label", "t_start")]
    n_pos = int((fdf["label"] == 1).sum())
    n_neg = int((fdf["label"] == 0).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fenêtres (2 s)", f"{len(fdf):,}".replace(",", " "))
    k2.metric("Features", len(feats))
    k3.metric("Patients", fdf["patient"].nunique())
    k4.metric("Classe crise", f"{100 * n_pos / len(fdf):.2f} %")

    st.subheader("Répartition des classes (déséquilibre)")
    figc = go.Figure(data=[
        go.Bar(x=["Hors-crise (0)", "Crise (1)"], y=[n_neg, n_pos],
               marker_color=["#94a3b8", "#dc2626"],
               text=[str(n_neg), str(n_pos)], textposition="outside")])
    figc.update_layout(height=300, yaxis_title="Nombre de fenêtres",
                       title="~96 % de fenêtres hors-crise")
    st.plotly_chart(figc, width='stretch')
    st.caption("Déséquilibre structurel : l'accuracy d'un modèle « tout 0 » atteint "
               "déjà {:.1f} %. On utilise F1, MCC et AUPRC.".format(100 * n_neg / len(fdf)))

    st.subheader("Distribution d'une feature selon la classe")
    c1, c2 = st.columns(2)
    sel_feat = c1.selectbox("Feature", feats,
                            index=feats.index("EEGCz__energy") if "EEGCz__energy" in feats else 0,
                            key="ds_feat")
    sub = fdf[[sel_feat, "label"]].copy()
    if len(sub) > 6000:
        sub = sub.sample(6000, random_state=0)
    figf = go.Figure()
    for lab, col in ((0, "#94a3b8"), (1, "#dc2626")):
        figf.add_trace(go.Histogram(x=sub.loc[sub["label"] == lab, sel_feat],
                                    name="classe " + str(lab), opacity=0.65,
                                    marker_color=col, nbinsx=60))
    figf.update_layout(height=340, barmode="overlay",
                       xaxis_title=sel_feat, yaxis_title="Effectif")
    st.plotly_chart(figf, width='stretch')

    st.subheader("Matrice de corrélations (Pearson)")
    familles = {
        "Temporelles (mean/std/var/min/max/amp/rms/energy)": ["__mean", "__std", "__var", "__min", "__max", "__amp", "__rms", "__energy"],
        "Spectralles (bandes de puissance)": ["__pow_", "__rel_"],
        "Complexité (Hjorth, entropies, ondelettes)": ["__hjorth", "__entropy", "__perm_entropy", "__wav_", "__skew", "__kurtosis", "__line_length", "__dom_freq", "__spec_"],
    }
    fam = st.selectbox("Famille de features", list(familles), key="ds_fam")
    suf = familles[fam]
    chans = sorted({c.split("__")[0] for c in feats if c.startswith("EEG")})
    sel_ch = st.multiselect("Canaux", chans, default=chans[:8], key="ds_chans")
    cols_corr = [c for c in feats if c.startswith("EEG")
                 and any(c.endswith(s) for s in suf)
                 and c.split("__")[0] in sel_ch]
    if len(cols_corr) >= 2:
        samp = fdf[cols_corr].sample(min(4000, len(fdf)), random_state=0)             if len(fdf) > 4000 else fdf[cols_corr]
        cm = samp.corr(numeric_only=True)
        figm = go.Figure(data=go.Heatmap(z=cm.values, x=cm.columns, y=cm.columns,
                                         colorscale="RdBu_r", zmin=-1, zmax=1))
        figm.update_layout(height=max(340, 14 * len(cols_corr)),
                           title="{} features — {}".format(len(cols_corr), fam))
        st.plotly_chart(figm, width='stretch')
        hi = (cm.where(np.abs(cm) >= 0.90).stack()
              .reset_index().rename(columns={0: "r", "level_0": "a", "level_1": "b"}))
        hi = hi[hi["a"] < hi["b"]]
        if len(hi):
            st.warning("{} paires de features ont |r| ≥ 0.90 (multicolinéarité) : "
                       "features redondantes, à filtrer (MI) ou à regrouper."
                       .format(len(hi)))
        else:
            st.success("Aucune paire avec |r| ≥ 0.90 dans cette sélection.")
    else:
        st.info("Sélectionnez au moins 2 features.")


# ---------------------------------------------------------------------------
# Onglet 5 — Modèles
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Comparaison des modèles")
    rand = get_results("results_random_extended.parquet")
    if rand.empty:
        rand = get_results("results_random_split.parquet")
    gkf = get_results("results_groupkfold.parquet")
    lopo = get_results("results_lopo.parquet")

    st.subheader("Split aléatoire 80/20 (exigé — mais fuite par recouvrement)")
    if not rand.empty:
        st.dataframe(rand[["model", "accuracy", "precision", "recall", "f1", "mcc",
                           "roc_auc", "pr_auc"]].round(3), width='stretch')
    st.subheader("GroupKFold / LOPO (validation par patient — honnête)")
    for tab in (gkf, lopo):
        if not tab.empty:
            grp = tab.groupby("model")[["f1", "pr_auc", "recall", "precision", "mcc"]].agg(["mean", "std"])
            st.dataframe(grp.round(3), width='stretch')
            break

    sweep = get_results("threshold_sweep.parquet")
    if not sweep.empty:
        fig = go.Figure()
        for col in ["precision", "recall", "f1", "mcc"]:
            fig.add_trace(go.Scatter(x=sweep.threshold, y=sweep[col], name=col))
        fig.update_layout(height=380, xaxis_title="Seuil de décision", yaxis_title="Métrique",
                          title="Métriques fenêtre vs seuil")
        st.plotly_chart(fig, width='stretch')

    preds = get_predictions()
    st.subheader("Performance par patient (LOPO)")
    from sklearn.metrics import precision_score, recall_score, f1_score
    rows = []
    for pid, g in preds.groupby("patient"):
        pr = (g["score"] >= 0.5).astype(int)
        rows.append({"patient": pid, "n": len(g),
                     "precision": precision_score(g["label"], pr, zero_division=0),
                     "recall": recall_score(g["label"], pr, zero_division=0),
                     "f1": f1_score(g["label"], pr, zero_division=0)})
    st.dataframe(pd.DataFrame(rows).round(3), width='stretch')

    st.subheader("Matrice de confusion (Partie IX / Question 13)")
    cp1, cp2, cp3 = st.columns(3)
    _proto = cp1.selectbox("Protocole d'evaluation",
                           ["Split aleatoire 80/20", "GroupKFold (par patient)",
                            "LOPO (par patient)"], key="cm_proto")
    _map = {"Split aleatoire 80/20": "results_random_split.parquet",
            "GroupKFold (par patient)": "results_groupkfold.parquet",
            "LOPO (par patient)": "results_lopo.parquet"}
    _dfcm = get_results(_map[_proto])
    _models = sorted(_dfcm["model"].astype(str).unique()) if not _dfcm.empty else []
    _mdl = cp2.selectbox("Modele", _models, key="cm_mdl")
    _pct = cp3.checkbox("En % (par ligne)", key="cm_pct")
    if not _dfcm.empty and _mdl:
        _g = _dfcm[_dfcm["model"].astype(str) == _mdl]
        _row = _g.iloc[0]
        _tp = int(_row.get("tp", 0) or 0)
        _fp = int(_row.get("fp", 0) or 0)
        _fn = int(_row.get("fn", 0) or 0)
        _tn = int(_row.get("tn", 0) or 0)
        if _pct:
            _d0 = _tp + _fp or 1
            _d1 = _fn + _tn or 1
            _z = [[round(_tp / _d0 * 100, 1), round(_fp / _d0 * 100, 1)],
                  [round(_fn / _d1 * 100, 1), round(_tn / _d1 * 100, 1)]]
            _text = [["{} ({}%)".format(_tp, _z[0][0]), "{} ({}%)".format(_fp, _z[0][1])],
                     ["{} ({}%)".format(_fn, _z[1][0]), "{} ({}%)".format(_tn, _z[1][1])]]
            _title = "Matrice de confusion (en % par ligne)"
        else:
            _z = [[_tp, _fp], [_fn, _tn]]
            _text = [[str(_tp), str(_fp)], [str(_fn), str(_tn)]]
            _title = "Matrice de confusion (comptages)"
        _figcm = go.Figure(data=go.Heatmap(z=_z, text=_text, texttemplate="%{text}",
                                           colorscale="Blues",
                                           x=["Predit crise (1)", "Predit hors-crise (0)"],
                                           y=["Reel crise (1)", "Reel hors-crise (0)"],
                                           showscale=False))
        _figcm.update_layout(height=320, title=_title)
        st.plotly_chart(_figcm, width='stretch')
        _acc = (_tp + _tn) / (_tp + _tn + _fp + _fn + 1e-12)
        _prec = _tp / (_tp + _fp + 1e-12)
        _rec = _tp / (_tp + _fn + 1e-12)
        _f1 = 2 * _prec * _rec / (_prec + _rec + 1e-12)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", "{:.3f}".format(_acc))
        m2.metric("Precision", "{:.3f}".format(_prec))
        m3.metric("Recall", "{:.3f}".format(_rec))
        m4.metric("F1-score", "{:.3f}".format(_f1))
        st.caption("FP = fausse alarme, FN = crise manquee. A cause du fort desequilibre "
                   "(96 % hors-crise), l'Accuracy seule est trompeuse : un modele qui "
                   "predict toujours 0 obtient deja ~96 % avec Recall et F1 a 0.")


# ---------------------------------------------------------------------------
# Onglet 6 — Démo détection
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Démo détection — rejouer le EEG")
    preds = get_predictions()
    gpred = preds[preds["patient"] == patient].sort_values("t_start")
    sel = seizures[seizures["Patient"].astype(str) == patient]
    true_ev = list(map(tuple, sel[["Debut_Sec", "Fin_Sec"]].to_numpy())) if len(sel) else []
    intervals = []
    times = np.array([])
    sm = np.array([])

    c1, c2, c3 = st.columns(3)
    seuil = c1.slider("Seuil de décision", 0.05, 0.95, 0.30, 0.05)
    lissage = c2.slider("Lissage (fenêtres)", 1, 15, 5, 2)
    min_dur = c3.slider("Durée minimale détectée (s)", 0.0, 10.0, 2.0, 0.5)

    dur = float(rec_sig.shape[0] / rec_fs)
    t_curseur = st.slider("Curseur temporel (s)", 0.0, dur, 0.0, 0.5)

    if not gpred.empty:
        times = gpred["t_start"].to_numpy()
        sm, intervals = pp.postprocess(times, gpred["score"].to_numpy(),
                                       threshold=seuil, smooth_size=lissage,
                                       min_gap_s=2.0, min_dur_s=min_dur)
        fig = go.Figure()
        for i, (s, e) in enumerate(true_ev):
            fig.add_vrect(x0=s, x1=e, fillcolor="green", opacity=0.15, line_width=0)
        for i, (s, e) in enumerate(intervals):
            fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.15, line_width=0)
        fig.add_trace(go.Scatter(x=times, y=sm, name="Probabilité lissée",
                                 line=dict(color="darkorange")))
        fig.add_hline(y=seuil, line_dash="dash", line_color="red")
        fig.add_vline(x=t_curseur, line_color="black")
        alarme = any(s <= t_curseur <= e for s, e in intervals)
        fig.update_layout(height=400, xaxis_title="Temps (s)", yaxis_title="Probabilité",
                          title="Alerte crise" if alarme else "Pas de crise au curseur")
        st.plotly_chart(fig, width='stretch')
        if alarme:
            st.error("⚠️ CRISE DÉTECTÉE au curseur !")
        else:
            st.success("Pas de crise au curseur.")

        det = {patient: intervals}
        durs = {patient: dur}
        ev = event_metrics_by_patient(seizures, det, durs)
        st.dataframe(ev.round(3), width='stretch')
    else:
        st.warning("Pas de prédictions pour ce patient.")
        times = np.array([])
        sm = np.array([])



    st.subheader("Intervalles reconstruits vs annotations reelles (Q14)")
    _dfint = pd.DataFrame([{"Detecte #": i + 1, "Debut (s)": round(s, 1),
                            "Fin (s)": round(e, 1),
                            "Duree (s)": round(e - s, 1)}
                           for i, (s, e) in enumerate(intervals)])
    if _dfint.empty:
        st.info("Aucun intervalle detecte avec ces parametres.")
    else:
        st.dataframe(_dfint, width='stretch')
    _dfref = pd.DataFrame([{"Reelle #": i + 1, "Debut (s)": round(s, 1),
                            "Fin (s)": round(e, 1),
                            "Duree (s)": round(e - s, 1)}
                           for i, (s, e) in enumerate(true_ev)])
    if not _dfref.empty:
        st.subheader("Annotations reelles du patient selectionne")
        st.dataframe(_dfref, width='stretch')


# ---------------------------------------------------------------------------
# Onglet 7 — Explicabilité
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("Explicabilité")
    imp_path = config.CACHE_DIR / "feature_importance.csv"
    ci_path = config.CACHE_DIR / "channel_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0)["importance"].sort_values(ascending=False)
        st.subheader("Top 25 features (RandomForest)")
        st.bar_chart(imp.head(25))
    if ci_path.exists():
        ci = pd.read_csv(ci_path, index_col=0).iloc[:, 0]
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Importance par canal")
            st.bar_chart(ci.sort_values(ascending=False))
        with colB:
            st.subheader("Carte topographique")
            try:
                fig = viz.plot_channel_topomap(ci.to_dict())
                st.pyplot(fig); plt.close(fig)
            except Exception as e:  # pragma: no cover
                st.info(f"Carte indisponible : {e}")


# ---------------------------------------------------------------------------
# Onglet 8 — Analyse des erreurs (Q15 / Partie XI)
# ---------------------------------------------------------------------------
with tabs[7]:
    st.header("Analyse des erreurs du modèle (Question 15)")
    st.caption("Source : cache/predictions.parquet + annotations réelles. "
               "Réglages recalculés à la volée.")

    a1, a2, a3 = st.columns(3)
    thr = a1.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05, key="ea_thr")
    sm = a2.slider("Lissage (fenêtres médianes)", 1, 15, 5, 2, key="ea_sm")
    mind = a3.slider("Durée minimale détectée (s)", 0.0, 8.0, 2.0, 0.5, key="ea_md")

    ana = get_error_analysis(thr, sm, mind)
    w = ana["window"]
    e = ana["event"]
    m = ana["metrics"]

    st.subheader("1 — Niveau fenêtre (chaque fenêtre de 2 s compte)")
    acc = (w["tp"] + w["tn"]) / max(1, sum(w.values()))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vrais positifs (TP)", "{:,}".format(w["tp"]).replace(",", " "),
              help="Fenêtre de crise correctement détectée")
    c2.metric("Faux positifs (FP)", "{:,}".format(w["fp"]).replace(",", " "),
              help="Fenêtre hors-crise classée crise → fausse alarme")
    c3.metric("Faux négatifs (FN)", "{:,}".format(w["fn"]).replace(",", " "),
              help="Fenêtre de crise manquée")
    c4.metric("Vrais négatifs (TN)", "{:,}".format(w["tn"]).replace(",", " "),
              help="Fenêtre hors-crise correctement écartée")
    st.write("Accuracy = {:.3f} — le TN représente **{:.1f} %** des fenêtres "
             "prédites : sous ce déséquilibre, l'accuracy seule ne veut rien dire."
             .format(acc, 100 * w["tn"] / max(1, sum(w.values()))))

    st.subheader("2 — Niveau crise (comparaison aux annotations réelles)")
    mm1, mm2, mm3, mm4, mm5 = st.columns(5)
    mm1.metric("Taux de détection", "{:.1%}".format(m["taux_detection"]) if m["taux_detection"] == m["taux_detection"] else "—")
    mm2.metric("Fausses alarmes / h", "{:.2f}".format(m["fausses_alarmes_h"]) if m["fausses_alarmes_h"] == m["fausses_alarmes_h"] else "—")
    mm3.metric("Erreur début (moy.)", "{:.1f} s".format(m["err_debut_moy_s"]) if m["err_debut_moy_s"] == m["err_debut_moy_s"] else "—")
    mm4.metric("Crises ≤ 5 s détectées", "{:.0%}".format(m["pct_courtes"]) if m["pct_courtes"] == m["pct_courtes"] else "—")
    mm5.metric("Crises > 10 s détectées", "{:.0%}".format(m["pct_longues"]) if m["pct_longues"] == m["pct_longues"] else "—")

    pp_df = ana["per_patient"].sort_values("FN", ascending=False)
    st.dataframe(pp_df.round(3), width='stretch')

    figbar = go.Figure()
    figbar.add_trace(go.Bar(x=pp_df["patient"], y=pp_df["TP"], name="TP",
                            marker_color="#16a34a"))
    figbar.add_trace(go.Bar(x=pp_df["patient"], y=pp_df["FN"], name="FN",
                            marker_color="#dc2626"))
    figbar.add_trace(go.Bar(x=pp_df["patient"], y=pp_df["FP"], name="FP",
                            marker_color="#f59e0b"))
    figbar.update_layout(barmode="stack", height=360, yaxis_title="Nombre de crises",
                         title="Répartition des erreurs par patient")
    st.plotly_chart(figbar, width='stretch')

    worst = pp_df[(pp_df["reelles"] > 0) & (pp_df["FN"] > 0)]
    if len(worst):
        st.caption("Les erreurs se concentrent sur {} patient(s) ({}…) : c'est le "
                   "signal d'un problème de **généralisation inter-sujets**, que la "
                   "validation par patient (GroupKFold / LOPO) met en évidence."
                   .format(len(worst), ", ".join(worst["patient"].head(3))))

    st.subheader("3 — Le détail des erreurs")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Crises manquées (FN)**")
        if len(ana["fn"]):
            st.dataframe(ana["fn"].round(1), width='stretch')
        else:
            st.success("Aucune crise manquée.")
    with d2:
        st.markdown("**Fausses alarmes (FP)**")
        if len(ana["fp"]):
            st.dataframe(ana["fp"].round(1), width='stretch')
        else:
            st.success("Aucune fausse alarme.")
    with d3:
        st.markdown("**Crises détectées (TP) — latence**")
        if len(ana["paires"]):
            st.dataframe(ana["paires"].round(2), width='stretch')
        else:
            st.info("Aucune détection appariée.")


# ---------------------------------------------------------------------------
# Onglet 9 — Avant / pendant la crise (Q10 / Partie VII)
# ---------------------------------------------------------------------------
with tabs[8]:
    st.header("Avant vs pendant la crise — comparaison des features (Question 10)")

    b1, b2, b3 = st.columns([1, 1, 2])
    eeg_only = b1.checkbox("Canaux EEG uniquement", value=True, key="cc_eeg",
                           help="Les canaux ECG/EMG/SLI ont une amplitude ~6× plus "
                                "grande : leur variation domine et inverse le sens "
                                "des écarts si on les mélange.")
    n_top = b2.slider("Nombre de features affichées", 5, 40, 15, 5, key="cc_top")
    fam_map = {
        "Toutes": None,
        "Temporelles (mean/std/var/energy…)": ["__mean", "__std", "__var", "__energy",
                                               "__rms", "__amp", "__min", "__max"],
        "Puissances de bandes": ["__pow_", "__rel_"],
        "Complexité / spectral": ["__hjorth", "__entropy", "__wav_", "__spec_",
                                  "__dom_freq", "__line_length", "__skew", "__kurtosis",
                                  "__perm_entropy"],
    }
    fam_sel = b3.selectbox("Famille de features", list(fam_map), key="cc_fam")

    m0, m1, var_pct, ratios = get_class_comparison(eeg_only)
    feats = list(var_pct.index)
    if fam_map[fam_sel] is not None:
        keep = [c for c in feats if any(c.endswith(s) for s in fam_map[fam_sel])]
        var_pct = var_pct[keep]
        m0 = m0[keep]
        m1 = m1[keep]
        ratios = ratios[keep]

    st.subheader("Les 4 exemples explicitement demandés (Énergie, Delta, Theta, Écart-type)")
    cartes = [("EEGCz__energy", "Énergie (EEGCz)"),
              ("EEGCz__pow_Delta", "Puissance Delta (EEGCz)"),
              ("EEGCz__pow_Theta", "Puissance Theta (EEGCz)"),
              ("EEGCz__std", "Écart-type (EEGCz)")]
    cc = st.columns(4)
    for col, (name, label) in zip(cc, cartes):
        with col:
            if name in var_pct.index:
                st.metric(label, "×{:.2f}".format(ratios[name]),
                          delta="{:+.1f} %".format(var_pct[name]))
            elif name in list(get_features().columns):
                st.metric(label, "hors sélection")
            else:
                st.metric(label, "absent")

    st.subheader("Moyennes par classe (échelle logarithmique)")
    topn = var_pct.reindex(var_pct.abs().sort_values(ascending=False).index).head(n_top)
    figb = go.Figure()
    figb.add_trace(go.Bar(x=[c.split("__", 1)[-1] + " · " + c.split("__")[0]
                             for c in topn.index],
                          y=m0[topn.index].clip(lower=1e-12), name="Hors-crise (0)",
                          marker_color="#94a3b8"))
    figb.add_trace(go.Bar(x=[c.split("__", 1)[-1] + " · " + c.split("__")[0]
                             for c in topn.index],
                          y=m1[topn.index].clip(lower=1e-12), name="Crise (1)",
                          marker_color="#dc2626"))
    figb.update_layout(barmode="group", height=420, yaxis_type="log",
                       yaxis_title="Moyenne (échelle log)",
                       xaxis_title="Feature (suffixe · canal)")
    st.plotly_chart(figb, width='stretch')

    st.subheader("Top features par variation relative (classe 1 vs classe 0)")
    dftop = pd.DataFrame({"moyenne_0": m0[topn.index], "moyenne_1": m1[topn.index],
                          "ratio": ratios[topn.index], "variation_%": var_pct[topn.index]})
    st.dataframe(dftop.round(3), width='stretch')
    n_big = int((var_pct.abs() > 100).sum())
    st.info("**{} features** (sur {}) varient de plus de 100 % entre hors-crise et crise."
            .format(n_big, len(var_pct)))

    st.subheader("Puissances de bandes : repos vs crise")
    bandes = ["Delta", "Theta", "Alpha", "Beta"]
    rows = []
    for b in bandes:
        cb0 = [c for c in m0.index if c.endswith("__pow_" + b)]
        cb1 = [c for c in m1.index if c.endswith("__pow_" + b)]
        if not cb0:
            continue
        v0, v1 = float(m0[cb0].mean()), float(m1[cb1].mean())
        rows.append({"Bande": b, "Moy. hors-crise": v0, "Moy. crise": v1,
                     "Ratio": v1 / v0 if v0 else float("nan"),
                     "Variation %": (v1 - v0) / (abs(v0) + 1e-12) * 100})
    st.dataframe(pd.DataFrame(rows).round(3), width='stretch')
    st.caption("Toutes les puissances montent pendant la crise : la décharge "
               "d'absence (~3 Hz) ajoute de l'énergie, surtout en Delta/Theta — "
               "c'est le marqueur spectral principal recherché en Q10.")
