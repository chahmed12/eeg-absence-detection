"""Benchmark standardisé et reproductible de bout en bout : LightGBM vs EEGNet.

Mesure le temps d'exécution complet par fenêtre de 2,0 s (512 échantillons à 256 Hz) :
  1. Filtrage passe-bande 0.5-40 Hz + coupe-bande (notch) 50 Hz
  2. Extraction de descripteurs (260 features pour LightGBM) VS Tenseur brut + FoldScaler (EEGNet)
  3. Inférence du modèle (LightGBM 100 arbres VS EEGNet PyTorch CPU)
  4. Post-traitement causal glissant (filtre médian causal sans accès au futur)

Protocole :
- Échauffement (warm-up) : 5 répétitions
- Mesures chronométrées : 50 répétitions
- Métriques rapportées : Médiane, Moyenne, Écart-type (Std), Intervalle Interquartile (IQR)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier

from src import config, models
from src.deep_learning import EEGNet, FoldScaler
from src.features import extract_features
from src.postprocess import smooth_scores_causal
from src.preprocess import bandpass_notch


def run_benchmark(n_warmup: int = 5, n_runs: int = 50, seed: int = 42):
    print("=" * 80)
    print(f"BENCHMARK DE BOUT EN BOUT : LIGHTGBM vs EEGNET ({n_runs} répétitions, warm-up={n_warmup})")
    print("=" * 80)

    fs = config.FS  # 256.0 Hz
    chans = list(config.COMMON_EEG_CHANNELS)  # 8 canaux
    win_len = int(config.WINDOW_SEC * fs)                        # 512 échantillons
    step = int(config.WINDOW_SEC * (1.0 - config.OVERLAP) * fs)   # 256 échantillons

    rng = np.random.default_rng(seed)
    raw_window = rng.normal(loc=0.0, scale=20.0, size=(win_len, len(chans))).astype(np.float32)

    # Initialisation des modèles
    print("[1/4] Préparation des modèles sur 8 canaux...")
    # LightGBM entraîné succinctement sur données synthétiques pour posséder 100 arbres réels
    X_dummy = rng.normal(size=(500, 260)).astype(np.float32)
    y_dummy = (rng.uniform(size=500) > 0.95).astype(int)
    lgbm = LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1, n_jobs=1)
    lgbm.fit(X_dummy, y_dummy)

    # EEGNet PyTorch CPU
    torch.set_num_threads(1)  # standardisé mono-thread pour un benchmark d'ingénierie déterministe
    eegnet = EEGNet(n_channels=len(chans), n_samples=win_len, F1=8, D=2, F2=16, dropout=0.0)
    eegnet.eval()
    scaler = FoldScaler()
    scaler.mean = np.zeros((1, len(chans), 1), dtype=np.float32)
    scaler.std = np.ones((1, len(chans), 1), dtype=np.float32)

    # Buffer causal simulé pour le post-traitement (5 derniers scores)
    history_scores = [0.1, 0.15, 0.12, 0.08]

    # --- Benchmark LightGBM ---
    print("\n[2/4] Exécution du pipeline LightGBM...")
    lgbm_t_filter = []
    lgbm_t_feats = []
    lgbm_t_infer = []
    lgbm_t_post = []
    lgbm_t_total = []

    for i in range(n_warmup + n_runs):
        t0 = time.perf_counter()

        # 1. Filtrage passe-bande + notch
        f_sig = bandpass_notch(raw_window, fs=fs, lowcut=0.5, highcut=40.0, notch_freq=50.0, notch_q=30.0, order=4)
        t1 = time.perf_counter()

        # 2. Extraction features (260 features)
        feats, _ = extract_features(f_sig, fs, chans, win_len, step)
        feats_clean = feats.drop(columns=["t_start"], errors="ignore")
        feats_vec = np.zeros((1, 260), dtype=np.float32)
        feats_vec[0, :min(feats_clean.shape[1], 260)] = feats_clean.to_numpy()[0, :260]
        t2 = time.perf_counter()

        # 3. Inférence
        prob = lgbm.predict_proba(feats_vec)[:, 1]
        t3 = time.perf_counter()

        # 4. Post-traitement causal
        buf = np.array(history_scores + [float(prob[0])])
        _ = smooth_scores_causal(buf, size=5)
        t4 = time.perf_counter()

        if i >= n_warmup:
            lgbm_t_filter.append((t1 - t0) * 1000)
            lgbm_t_feats.append((t2 - t1) * 1000)
            lgbm_t_infer.append((t3 - t2) * 1000)
            lgbm_t_post.append((t4 - t3) * 1000)
            lgbm_t_total.append((t4 - t0) * 1000)

    # --- Benchmark EEGNet ---
    print("\n[3/4] Exécution du pipeline EEGNet...")
    eeg_t_filter = []
    eeg_t_tensor = []
    eeg_t_infer = []
    eeg_t_post = []
    eeg_t_total = []

    for i in range(n_warmup + n_runs):
        t0 = time.perf_counter()

        # 1. Filtrage (identique)
        f_sig = bandpass_notch(raw_window, fs=fs, lowcut=0.5, highcut=40.0, notch_freq=50.0, notch_q=30.0, order=4)
        t1 = time.perf_counter()

        # 2. Mise en forme tenseur & FoldScaler
        x_raw = f_sig.T[np.newaxis, :, :]  # (1, 8, 512)
        x_norm = scaler.transform(x_raw)
        x_tensor = torch.from_numpy(x_norm).unsqueeze(1)  # (1, 1, 8, 512)
        t2 = time.perf_counter()

        # 3. Inférence PyTorch CPU
        with torch.no_grad():
            logit = eegnet(x_tensor)
            prob = torch.sigmoid(logit).item()
        t3 = time.perf_counter()

        # 4. Post-traitement causal
        buf = np.array(history_scores + [prob])
        _ = smooth_scores_causal(buf, size=5)
        t4 = time.perf_counter()

        if i >= n_warmup:
            eeg_t_filter.append((t1 - t0) * 1000)
            eeg_t_tensor.append((t2 - t1) * 1000)
            eeg_t_infer.append((t3 - t2) * 1000)
            eeg_t_post.append((t4 - t3) * 1000)
            eeg_t_total.append((t4 - t0) * 1000)

    # Calcul des statistiques
    def stats(arr):
        a = np.array(arr)
        q25, q50, q75 = np.percentile(a, [25, 50, 75])
        return {
            "median": q50,
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "iqr": q75 - q25,
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    s_lgbm = {
        "filter": stats(lgbm_t_filter),
        "prep_feats": stats(lgbm_t_feats),
        "inference": stats(lgbm_t_infer),
        "postprocess": stats(lgbm_t_post),
        "total": stats(lgbm_t_total),
    }

    s_eeg = {
        "filter": stats(eeg_t_filter),
        "prep_feats": stats(eeg_t_tensor),
        "inference": stats(eeg_t_infer),
        "postprocess": stats(eeg_t_post),
        "total": stats(eeg_t_total),
    }

    print("\n" + "=" * 80)
    print("RÉSULTATS DÉTAILLÉS DU BENCHMARK (ms par fenêtre de 2 s)")
    print("=" * 80)
    print(f"{'Étape':<35} | {'LightGBM (médiane ± IQR)':<22} | {'EEGNet (médiane ± IQR)':<22}")
    print("-" * 85)
    print(f"{'1. Filtrage (0.5-40Hz + 50Hz notch)':<35} | {s_lgbm['filter']['median']:6.2f} ± {s_lgbm['filter']['iqr']:4.2f} ms      | {s_eeg['filter']['median']:6.2f} ± {s_eeg['filter']['iqr']:4.2f} ms")
    print(f"{'2. Préparation (Features vs Tenseur)':<35} | {s_lgbm['prep_feats']['median']:6.2f} ± {s_lgbm['prep_feats']['iqr']:4.2f} ms      | {s_eeg['prep_feats']['median']:6.2f} ± {s_eeg['prep_feats']['iqr']:4.2f} ms")
    print(f"{'3. Inférence (100 arbres vs ConvNet)':<35} | {s_lgbm['inference']['median']:6.2f} ± {s_lgbm['inference']['iqr']:4.2f} ms      | {s_eeg['inference']['median']:6.2f} ± {s_eeg['inference']['iqr']:4.2f} ms")
    print(f"{'4. Post-traitement causal glissant':<35} | {s_lgbm['postprocess']['median']:6.2f} ± {s_lgbm['postprocess']['iqr']:4.2f} ms      | {s_eeg['postprocess']['median']:6.2f} ± {s_eeg['postprocess']['iqr']:4.2f} ms")
    print("-" * 85)
    print(f"{'TOTAL DE BOUT EN BOUT':<35} | {s_lgbm['total']['median']:6.2f} ± {s_lgbm['total']['iqr']:4.2f} ms      | {s_eeg['total']['median']:6.2f} ± {s_eeg['total']['iqr']:4.2f} ms")
    print("=" * 85)

    ratio = s_lgbm['total']['median'] / max(s_eeg['total']['median'], 1e-6)
    print(f"\nAccélération globale observée (médianes) : {ratio:.2f}x plus rapide pour EEGNet")
    print(f"Budget temps réel disponible (pas de 1.0 s) : 1000.00 ms")
    print(f"Charge CPU relative du pipeline : LightGBM = {s_lgbm['total']['median']/10:.2f} % | EEGNet = {s_eeg['total']['median']/10:.2f} %")

    # Sauvegarde des résultats au format JSON
    out_path = config.CACHE_DIR / "experiments" / "benchmark_end_to_end.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"lightgbm": s_lgbm, "eegnet": s_eeg, "speedup": ratio, "n_runs": n_runs}, f, indent=2)
    print(f"\nRapport de benchmark sauvegardé dans {out_path}")


if __name__ == "__main__":
    run_benchmark(n_warmup=5, n_runs=50)
