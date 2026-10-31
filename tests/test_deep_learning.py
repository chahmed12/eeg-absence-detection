"""Tests unitaires pour le module Deep Learning (src/deep_learning.py).

Tous les tests reposent sur des tenseurs synthétiques générés à la volée,
garantissant une exécution rapide (< 3 s) et 100 % CI-friendly.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src import config
from src.deep_learning import (
    EEGDataset,
    EEGNet,
    FoldScaler,
    SpatioTemporalCNN,
    Temporal1DCNN,
    compute_pos_weight,
    evaluate_model,
    extract_patient_raw_windows,
    train_epoch,
    train_fold_nested,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_multichannel_sig():
    fs = 256
    n_samples = fs * 6  # 6 secondes
    n_chans = 10
    names = list(config.COMMON_EEG_CHANNELS) + ["EMG1", "SLI"]
    sig = np.random.randn(n_samples, n_chans).astype(np.float32)
    return sig, names


# ---------------------------------------------------------------------------
# 1. Extraction et fenêtrage brut
# ---------------------------------------------------------------------------
def test_extract_patient_raw_windows_shape(synthetic_multichannel_sig):
    sig, names = synthetic_multichannel_sig
    win_len = 512  # 2 s
    step = 256     # 50 % overlap

    X, starts = extract_patient_raw_windows(sig, names, win_len=win_len, step=step)
    assert X.ndim == 3
    assert X.shape[1] == 8        # 8 canaux communs
    assert X.shape[2] == win_len  # 512 pas de temps
    assert len(X) == len(starts)
    assert X.dtype == np.float32


def test_extract_patient_raw_windows_missing_channel():
    sig = np.random.randn(1024, 3).astype(np.float32)
    names = ["EEGFp1", "EEGFp2", "EEGC3"]  # manque les 5 autres canaux
    with pytest.raises(ValueError, match="Canal requis"):
        extract_patient_raw_windows(sig, names, win_len=512, step=256)


# ---------------------------------------------------------------------------
# 2. Normalisation par fold (Zéro fuite)
# ---------------------------------------------------------------------------
def test_fold_scaler_zero_leakage():
    X_train = np.random.randn(50, 8, 512).astype(np.float32) * 10.0 + 5.0
    X_test = np.random.randn(20, 8, 512).astype(np.float32) * 2.0 + 1.0

    scaler = FoldScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # La moyenne sur le train doit être quasi nulle et l'écart-type ~ 1
    assert np.allclose(X_train_norm.mean(axis=(0, 2)), 0.0, atol=1e-3)
    assert np.allclose(X_train_norm.std(axis=(0, 2)), 1.0, atol=1e-3)

    # Le test doit être transformé sans modifier les statistiques du train
    assert X_test_norm.shape == X_test.shape
    assert not np.isnan(X_test_norm).any()


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset & Augmentation
# ---------------------------------------------------------------------------
def test_eeg_dataset_getitem():
    X = np.random.randn(20, 8, 512).astype(np.float32)
    y = np.random.randint(0, 2, size=20).astype(np.float32)

    ds = EEGDataset(X, y, augment=False)
    assert len(ds) == 20
    x_item, y_item = ds[0]
    assert x_item.shape == (8, 512)
    assert isinstance(x_item, torch.Tensor)
    assert isinstance(y_item, torch.Tensor)


def test_eeg_dataset_with_augmentation():
    X = np.ones((10, 8, 512), dtype=np.float32)
    y = np.ones(10, dtype=np.float32)

    ds = EEGDataset(X, y, augment=True, seed=42)
    x_item, _ = ds[0]
    assert x_item.shape == (8, 512)
    assert torch.isfinite(x_item).all()


# ---------------------------------------------------------------------------
# 4. Modèle 1 (EXP-01) : Temporal1DCNN
# ---------------------------------------------------------------------------
def test_temporal_1d_cnn_forward():
    model = Temporal1DCNN(n_channels=8, kernel_time=63)
    x = torch.randn(4, 8, 512)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_temporal_1d_cnn_param_count():
    model = Temporal1DCNN(n_channels=8)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 40_000, f"Trop de paramètres pour le modèle minimal : {n_params}"


def test_spatiotemporal_cnn_forward():
    model = SpatioTemporalCNN(n_channels=8, kernel_time=64)
    x = torch.randn(4, 8, 512)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 20_000


# ---------------------------------------------------------------------------
# 5. Modèle 2 (EXP-02/03) : EEGNet
# ---------------------------------------------------------------------------
def test_eegnet_forward():
    model = EEGNet(n_channels=8, n_samples=512)
    # Accepte (B, 8, 512)
    x = torch.randn(4, 8, 512)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_eegnet_param_count():
    model = EEGNet(n_channels=8, n_samples=512)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # EEGNet doit être ultra-compact (< 5 000 paramètres)
    assert n_params < 5_000, f"EEGNet doit rester ultra-léger : {n_params} params trouvés"


# ---------------------------------------------------------------------------
# 6. Boucle d'entraînement & Métriques
# ---------------------------------------------------------------------------
def test_compute_pos_weight():
    y = np.array([0, 0, 0, 0, 1])  # ratio 4:1
    w = compute_pos_weight(y)
    assert abs(w.item() - 4.0) < 1e-4


def test_train_epoch_and_evaluate():
    torch.manual_seed(42)
    X = np.random.randn(16, 8, 512).astype(np.float32)
    y = np.random.randint(0, 2, size=16).astype(np.float32)

    ds = EEGDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=8)

    model = Temporal1DCNN(n_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    loss = train_epoch(model, loader, optimizer, criterion, device)
    assert loss > 0.0
    assert not np.isnan(loss)

    y_true, y_prob = evaluate_model(model, loader, device)
    assert len(y_true) == 16
    assert len(y_prob) == 16
    assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all()


# ---------------------------------------------------------------------------
# 7. Validation interne stricte par patient
# ---------------------------------------------------------------------------
def test_train_fold_nested_isolates_test_patients():
    import pandas as pd
    import torch.nn as nn

    rng = np.random.default_rng(0)
    patients = np.repeat([f"P{i}" for i in range(6)], 10)
    y = (rng.random(60) > 0.5).astype(np.float32)
    meta = pd.DataFrame({
        "patient": patients,
        "label": y.astype(int),
        "t_start": np.tile(np.arange(10, dtype=np.float32), 6),
    })
    X = rng.normal(size=(60, 8, 32)).astype(np.float32)
    train_idx = np.arange(50)
    test_idx = np.arange(50, 60)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8 * 32, 1)

        def forward(self, x):
            return self.fc(torch.flatten(x, start_dim=1)).squeeze(-1)

    res = train_fold_nested(
        TinyModel,
        X,
        y,
        meta,
        train_idx,
        test_idx,
        epochs=1,
        batch_size=8,
        seed=0,
    )

    train_patients = set(meta.iloc[res["inner_train_idx"]]["patient"])
    val_patients = set(meta.iloc[res["inner_val_idx"]]["patient"])
    test_patients = set(meta.iloc[test_idx]["patient"])
    assert train_patients.isdisjoint(val_patients)
    assert val_patients.isdisjoint(test_patients)
    assert len(res["inner_val_probs"]) == len(res["inner_val_idx"])
    assert len(res["test_probs"]) == len(test_idx)
    assert res["best_epoch"] >= 1
