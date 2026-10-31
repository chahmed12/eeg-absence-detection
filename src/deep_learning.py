"""Pipeline Deep Learning pour la détection des crises d'absence sur signaux EEG.

Ce module implémente la démarche expérimentale progressive :
1. Extraction et mise en cache des fenêtres temporelles brutes (8 canaux communs x 512 pas).
2. Normalisation z-score calculée exclusivement sur le train de chaque fold (zéro fuite).
3. Architectures testables :
   - Modèle 1 (EXP-01) : Temporal1DCNN (convolutions temporelles 1D pures).
   - Modèle 2 (EXP-02/03) : EEGNet (convolutions temporelles + spatiales depthwise).
4. Boucle d'entraînement avec gestion du déséquilibre (BCEWithLogitsLoss pondérée) et callbacks.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .io import list_mat_files, load_annotations, load_patient_signal, patient_id_from_stem
from .labeling import patient_seizures, sample_labels
from .windowing import window_labels_from_samples, window_params, window_starts


# ---------------------------------------------------------------------------
# 1. Extraction des fenêtres temporelles brutes
# ---------------------------------------------------------------------------
class RawWindowBatch(NamedTuple):
    """Ensemble des fenêtres brutes avec métadonnées."""
    X: np.ndarray          # (N, C=8, T=512), float32
    y: np.ndarray          # (N,), uint8
    patient: np.ndarray    # (N,), string
    t_start: np.ndarray    # (N,), float32


def extract_patient_raw_windows(
    sig: np.ndarray,
    names: list[str],
    win_len: int,
    step: int,
    target_channels: tuple[str, ...] = config.COMMON_EEG_CHANNELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Découpe le signal d'un patient en fenêtres brutes sur les canaux communs.

    Args:
        sig: signal (n_échantillons, n_canaux), float32 pré-traité.
        names: liste des noms de canaux d'origine.
        win_len, step: taille et pas de fenêtre en échantillons.
        target_channels: canaux conservés dans un ordre canonique stable.

    Returns:
        (X de forme (n_fenêtres, len(target_channels), win_len), t_starts).
    """
    canonical_names = [config.canonical_channel(n) for n in names]
    indices = []
    for target in target_channels:
        if target in canonical_names:
            indices.append(canonical_names.index(target))
        else:
            raise ValueError(f"Canal requis {target} absent parmi les canaux : {canonical_names}")

    # Extraction des 8 canaux dans l'ordre canonique fixé
    sig_selected = sig[:, indices]  # (n_samples, 8)
    starts = window_starts(sig.shape[0], win_len, step)

    # Empilement vectorisé : (n_fenêtres, 8, win_len)
    windows = [sig_selected[s : s + win_len, :].T for s in starts]
    if len(windows) == 0:
        return np.empty((0, len(target_channels), win_len), dtype=np.float32), np.empty(0, dtype=np.int64)

    X = np.stack(windows).astype(np.float32)
    return X, starts


def build_raw_windows_dataset(
    seizures: pd.DataFrame | None = None,
    cache_path: Path | None = None,
    force: bool = False,
    window_sec: float = config.WINDOW_SEC,
    overlap: float = config.OVERLAP,
) -> RawWindowBatch:
    """Charge ou extrait le dataset complet de fenêtres brutes pour tous les patients.

    Stocké au format .npz compressé dans le cache.
    """
    cache_path = Path(cache_path or (config.CACHE_DIR / "raw_windows.npz"))
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return RawWindowBatch(
            X=data["X"],
            y=data["y"],
            patient=data["patient"],
            t_start=data["t_start"],
        )

    if seizures is None:
        seizures, _ = load_annotations()

    win_len, step = window_params(config.FS, window_sec, overlap)
    all_X, all_y, all_pats, all_t0 = [], [], [], []

    for mat_path in list_mat_files():
        pid = patient_id_from_stem(mat_path.stem)
        rec = load_patient_signal(mat_path)
        X, starts = extract_patient_raw_windows(rec.sig, rec.names, win_len, step)
        if len(starts) == 0:
            continue

        y_samples = sample_labels(rec.sig.shape[0], rec.fs, patient_seizures(seizures, pid))
        y_win = window_labels_from_samples(y_samples, starts, win_len)

        all_X.append(X)
        all_y.append(y_win)
        all_pats.append(np.full(len(starts), pid))
        all_t0.append((starts / rec.fs).astype(np.float32))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0).astype(np.uint8)
    patient = np.concatenate(all_pats, axis=0)
    t_start = np.concatenate(all_t0, axis=0)

    config.ensure_dirs()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, patient=patient, t_start=t_start)

    return RawWindowBatch(X=X, y=y, patient=patient, t_start=t_start)


# ---------------------------------------------------------------------------
# 2. Normalisation z-score par fold (ZÉRO FUITE)
# ---------------------------------------------------------------------------
class FoldScaler:
    """Normaliseur calculé EXCLUSIVEMENT sur le train d'un fold."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> FoldScaler:
        # X: (N, C, T) -> moyenne et std par canal C
        self.mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        self.std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Le normaliseur doit être ajusté sur le train d'abord.")
        return ((X - self.mean) / self.std).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset & Augmentations
# ---------------------------------------------------------------------------
class EEGDataset(Dataset):
    """Dataset PyTorch pour fenêtres EEG (B, C, T)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, seed: int = 42):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            # 1. Amplitude scaling (0.85 à 1.15)
            scale = self.rng.uniform(0.85, 1.15)
            x = x * scale

            # 2. Channel dropout (masquage occasionnel d'1 canal)
            if self.rng.random() < 0.2:
                c_drop = self.rng.integers(0, x.shape[0])
                x[c_drop, :] = 0.0

        return x, y


# ---------------------------------------------------------------------------
# 4. Modèle 1 (EXP-01) : CNN 1D Temporel Minimal
# ---------------------------------------------------------------------------
class Temporal1DCNN(nn.Module):
    """Architecture minimale de test (Hypothèse 1 : dynamique temporelle pure).

    Entrée : (B, C=8, T=512)
    Applique des convolutions 1D sur l'axe temporel pour vérifier si la
    forme d'onde pointe-onde (250 ms) réduit la latence et les faux négatifs.
    """

    def __init__(self, n_channels: int = 8, kernel_time: int = 63, dropout: float = 0.3):
        super().__init__()
        # Étage 1 : capture un cycle d'oscillation (~250 ms à 256 Hz)
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=kernel_time, padding="same")
        self.bn1 = nn.BatchNorm1d(16)
        self.act1 = nn.ELU()
        self.pool1 = nn.MaxPool1d(kernel_size=4)  # 512 -> 128

        # Étage 2 : capture les enchaînements lents
        self.conv2 = nn.Conv1d(16, 32, kernel_size=31, padding="same")
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.ELU()
        self.pool2 = nn.AdaptiveAvgPool1d(1)      # 128 -> 1

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, 512)
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.pool1(h)
        h = self.act2(self.bn2(self.conv2(h)))
        h = self.pool2(h)
        h = h.squeeze(-1)  # (B, 32)
        h = self.drop(h)
        return self.fc(h).squeeze(-1)  # logit (B,)


# ---------------------------------------------------------------------------
# 4b. Modèle EXP-02 : CNN Spatio-Temporel (Ajout du filtre spatial inter-électrodes)
# ---------------------------------------------------------------------------
class SpatioTemporalCNN(nn.Module):
    """Architecture intermédiaire (Hypothèse 2 : ajout du filtrage spatial).

    Entrée : (B, C=8, T=512) ou (B, 1, C=8, T=512)
    1. Convolution temporelle : extrait les motifs fréquentiels d'onde.
    2. Convolution spatiale Depthwise : combine les 8 électrodes pour rejeter les
       artefacts asymétriques ou locaux (yeux, muscles) et vérifier la généralisation.
    """

    def __init__(self, n_channels: int = 8, kernel_time: int = 64, dropout: float = 0.3):
        super().__init__()
        # Étage 1 : Convolutions temporelles (1 -> 8 filtres)
        self.conv_time = nn.Conv2d(1, 8, (1, kernel_time), padding=(0, kernel_time // 2), bias=False)
        self.bn_time = nn.BatchNorm2d(8)

        # Étage 2 : Convolutions spatiales Depthwise (combine les 8 canaux)
        self.conv_spatial = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.bn_spatial = nn.BatchNorm2d(16)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Étage 3 : Convolutions temporelles secondaires
        self.conv2 = nn.Conv2d(16, 32, (1, 32), padding=(0, 16), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ELU()
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.drop2 = nn.Dropout(dropout)

        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        h = self.bn_time(self.conv_time(x))
        h = self.drop1(self.pool1(self.act1(self.bn_spatial(self.conv_spatial(h)))))
        h = self.drop2(self.pool2(self.act2(self.bn2(self.conv2(h)))))
        h = torch.flatten(h, 1)
        return self.fc(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 5. Modèle 2 (EXP-02/03) : EEGNet (Spatio-temporel complet)
# ---------------------------------------------------------------------------
class EEGNet(nn.Module):
    """Architecture EEGNet compacte (Lawhern et al., 2018).

    Entrée : (B, 1, C=8, T=512)
    Sépare strictement le filtrage temporel (oscillations) et spatial (scalp).
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_samples: int = 512,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_len: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        # Bloc 1 : Convolution temporelle
        self.conv_time = nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len // 2), bias=False)
        self.bn_time = nn.BatchNorm2d(F1)

        # Bloc 2 : Convolution spatiale (Depthwise sur tous les canaux)
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn_spatial = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Bloc 3 : Convolutions séparables
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn_sep = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Tête de classification finale
        # Calcul automatique de la dimension aplatie
        out_len = n_samples // (4 * 8)
        self.fc = nn.Linear(F2 * out_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Si x est (B, C, T), on ajoute la dimension canal d'image -> (B, 1, C, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Bloc 1 & 2
        h = self.bn_time(self.conv_time(x))
        h = self.bn_spatial(self.conv_spatial(h))
        h = self.drop1(self.pool1(self.act1(h)))

        # Bloc 3
        h = self.conv_sep_depth(h)
        h = self.bn_sep(self.conv_sep_point(h))
        h = self.drop2(self.pool2(self.act2(h)))

        h = torch.flatten(h, 1)
        return self.fc(h).squeeze(-1)  # logit (B,)


# ---------------------------------------------------------------------------
# 6. Boucle d'entraînement & Métriques
# ---------------------------------------------------------------------------
def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Calcule le poids optimal pour BCEWithLogitsLoss face au déséquilibre."""
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    ratio = max(n_neg / max(n_pos, 1.0), 1.0)
    return torch.tensor([ratio], dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Entraîne le modèle sur une époque complète."""
    model.train()
    total_loss = 0.0
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les probabilités (sigmoid) et renvoie (y_true, y_prob)."""
    model.eval()
    all_probs, all_y = [], []
    for x_b, y_b in loader:
        x_b = x_b.to(device)
        logits = model(x_b)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_y.append(y_b.numpy())
    return np.concatenate(all_y), np.concatenate(all_probs)


def train_fold_nested(
    model_factory: Callable[[], nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    val_fraction: float = 0.25,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = config.RANDOM_SEED,
    device: torch.device | None = None,
) -> dict:
    """Entraîne un fold avec validation interne par patient et test externe aveugle.

    Protocole strictement imbriqué :
    1. le train externe est séparé en train interne / validation interne par patient ;
    2. le checkpoint et le scheduler sont sélectionnés sur la validation interne ;
    3. un modèle final est réentraîné sur tout le train externe pendant le nombre
       d'époques retenu, puis évalue le test externe, jamais utilisé avant.

    Returns:
        Dictionnaire avec les indices, le meilleur epoch, les probabilités de
        validation interne et de test.
    """
    from .models import patient_validation_split

    device = device or torch.device("cpu")
    inner_train_idx, inner_val_idx = patient_validation_split(
        meta, train_idx, val_fraction=val_fraction, seed=seed
    )

    # --- Modèle interne : sélection du checkpoint sur validation interne ---
    torch.manual_seed(seed)
    scaler_inner = FoldScaler().fit(X[inner_train_idx])
    X_inner = scaler_inner.transform(X[inner_train_idx])
    X_val = scaler_inner.transform(X[inner_val_idx])
    y_inner = y[inner_train_idx]
    y_val = y[inner_val_idx]

    loader_inner = DataLoader(
        EEGDataset(X_inner, y_inner, augment=True, seed=seed),
        batch_size=batch_size,
        shuffle=True,
    )
    loader_val = DataLoader(
        EEGDataset(X_val, y_val, augment=False),
        batch_size=batch_size,
        shuffle=False,
    )

    model_inner = model_factory().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(y_inner).to(device)
    )
    optimizer = torch.optim.Adam(model_inner.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_f1 = -1.0
    best_epoch = 1
    best_state = None
    from sklearn.metrics import f1_score

    for epoch in range(1, epochs + 1):
        train_epoch(model_inner, loader_inner, optimizer, criterion, device)
        _, val_probs = evaluate_model(model_inner, loader_val, device)
        f1 = f1_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)
        scheduler.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_state = copy.deepcopy(model_inner.state_dict())

    if best_state is not None:
        model_inner.load_state_dict(best_state)
    _, inner_val_probs = evaluate_model(model_inner, loader_val, device)

    # --- Modèle final : tout le train externe, nombre d'époques retenu ---
    torch.manual_seed(seed + 1000)
    scaler_full = FoldScaler().fit(X[train_idx])
    X_full = scaler_full.transform(X[train_idx])
    X_test = scaler_full.transform(X[test_idx])
    y_full = y[train_idx]

    loader_full = DataLoader(
        EEGDataset(X_full, y_full, augment=True, seed=seed + 1000),
        batch_size=batch_size,
        shuffle=True,
    )
    loader_test = DataLoader(
        EEGDataset(X_test, y[test_idx], augment=False),
        batch_size=batch_size,
        shuffle=False,
    )

    model_full = model_factory().to(device)
    criterion_full = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(y_full).to(device)
    )
    optimizer_full = torch.optim.Adam(model_full.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(max(1, best_epoch)):
        train_epoch(model_full, loader_full, optimizer_full, criterion_full, device)

    y_test, test_probs = evaluate_model(model_full, loader_test, device)

    return {
        "inner_train_idx": inner_train_idx,
        "inner_val_idx": inner_val_idx,
        "best_epoch": best_epoch,
        "best_inner_f1": float(best_f1),
        "inner_val_y": y_val,
        "inner_val_probs": inner_val_probs,
        "test_y": y_test,
        "test_probs": test_probs,
    }
