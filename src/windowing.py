"""Segmentation en fenêtres glissantes et étiquetage des fenêtres.

Convention (cahier des charges) : fenêtre de 2 s (512 échantillons à 256 Hz),
recouvrement 50 % (pas de 256 échantillons). Une fenêtre porte la classe 1 si
au moins ``config.LABEL_THRESHOLD`` de ses échantillons sont en crise.

Le nombre théorique de fenêtres est ``floor((N - W) / S) + 1``.
"""

from __future__ import annotations

import numpy as np

from . import config


def window_params(
    fs: float = config.FS,
    window_sec: float = config.WINDOW_SEC,
    overlap: float = config.OVERLAP,
) -> tuple[int, int]:
    """Convertit (durée de fenêtre, recouvrement) en (longueur W, pas S) en échantillons."""
    w = int(round(window_sec * fs))
    s = int(round(w * (1.0 - overlap)))
    if w <= 0 or s <= 0:
        raise ValueError("Paramètres de fenêtrage invalides (W ou S <= 0).")
    return w, s


def count_windows(n_samples: int, win_len: int, step: int) -> int:
    """Nombre de fenêtres complètes : ``floor((N - W) / S) + 1`` (0 si trop court)."""
    if n_samples < win_len:
        return 0
    return (n_samples - win_len) // step + 1


def window_starts(n_samples: int, win_len: int, step: int) -> np.ndarray:
    """Indices de début de chaque fenêtre."""
    return np.arange(0, n_samples - win_len + 1, step, dtype=np.int64)


def iter_windows(
    sig: np.ndarray, win_len: int, step: int
) -> "iter":
    """Itère sur (start, window) sans matérialiser toutes les fenêtres en mémoire."""
    for start in window_starts(sig.shape[0], win_len, step):
        yield int(start), sig[start:start + win_len]


def window_labels_from_samples(
    labels: np.ndarray,
    starts: np.ndarray,
    win_len: int,
    threshold: float = config.LABEL_THRESHOLD,
) -> np.ndarray:
    """Étiquette chaque fenêtre à partir du vecteur de classes par échantillon.

    Vectorisé via une somme cumulée (aucune boucle Python sur les fenêtres).
    """
    cs = np.concatenate(([0], np.cumsum(labels, dtype=np.int64)))
    moyens = (cs[starts + win_len] - cs[starts]) / float(win_len)
    return (moyens >= threshold).astype(np.uint8)
