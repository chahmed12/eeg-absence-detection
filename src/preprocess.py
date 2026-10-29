"""Pré-traitement du signal EEG : filtrage passe-bande + réjection secteur.

Le signal brut issu des ``.mat`` contient un **offset DC** important (l'énergie
brute atteint ~1e8, soit des amplitudes non physiologiques) et du bruit haute
fréquence (activité musculaire, ronflement du secteur à 50 Hz). On applique
donc, **avant tout fenêtrage ou extraction de caractéristiques** :

1. un **passe-bande** Butterworth (0,5–40 Hz) en phase nulle (``filtfilt``) :
   il retire l'offset DC et la dérive lente (< 0,5 Hz) ainsi que le bruit
   haute fréquence (> 40 Hz), tout en préservant la décharge pointe-onde de
   l'absence (~3 Hz) ;
2. un **coupe-bande** (notch) à la fréquence du secteur (50 Hz) pour éliminer
   le ronflement de la ligne électrique.

Le filtrage est appliqué **canal par canal** (passe-bande puis notch). Sans ce
pré-traitement, les features d'énergie dominent artificiellement et le spectre
est pollué, ce qui biaise l'interprétation des bandes de fréquences.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from . import config


def _sos_bandpass(fs: float, lowcut: float, highcut: float, order: int) -> np.ndarray:
    """Coefficients SOS d'un passe-bande Butterworth (bornes en Hz)."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.99)
    return sp_signal.butter(order, [low, high], btype="bandpass", output="sos")


def _sos_notch(fs: float, freq: float, q: float) -> np.ndarray:
    """Coefficients SOS d'un coupe-bande (notch) à ``freq`` Hz."""
    b, a = sp_signal.iirnotch(freq, q, fs=fs)
    return sp_signal.tf2sos(b, a)


def bandpass_notch(
    sig: np.ndarray,
    fs: float = config.FS,
    lowcut: float = config.EEG_LOWCUT,
    highcut: float = config.EEG_HIGHCUT,
    notch_freq: float = config.NOTCH_FREQ,
    notch_q: float = config.NOTCH_Q,
    order: int = config.FILTER_ORDER,
) -> np.ndarray:
    """Applique un passe-bande (phase nulle) puis un notch à chaque canal.

    Args:
        sig: signal (n_échantillons, n_canaux).
        fs: fréquence d'échantillonnage.
        lowcut, highcut: bornes du passe-bande (Hz).
        notch_freq, notch_q: fréquence et facteur de qualité du coupe-bande.
        order: ordre du filtre Butterworth passe-bande.

    Returns:
        Signal filtré, même forme, en ``float32``.
    """
    x = np.asarray(sig, dtype=np.float64)
    mono = x.ndim == 1
    if mono:
        x = x[:, None]

    sos_bp = _sos_bandpass(fs, lowcut, highcut, order)
    sos_no = _sos_notch(fs, notch_freq, notch_q)

    out = np.empty_like(x, dtype=np.float64)
    for c in range(x.shape[1]):
        col = x[:, c]
        if not np.all(np.isfinite(col)):
            col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
        y = sp_signal.sosfiltfilt(sos_bp, col)
        y = sp_signal.sosfiltfilt(sos_no, y)
        out[:, c] = y

    out = out.astype(np.float32)
    return out[:, 0] if mono else out


def preprocess_signal(sig: np.ndarray, fs: float = config.FS) -> np.ndarray:
    """Point d'entrée du pré-traitement (respecte ``config.PREPROCESS_ENABLED``).

    Si le pré-traitement est désactivé, le signal est seulement converti en
    ``float32`` (aucune copie inutile).
    """
    if not config.PREPROCESS_ENABLED:
        return np.asarray(sig, dtype=np.float32)
    return bandpass_notch(
        sig,
        fs=fs,
        lowcut=config.EEG_LOWCUT,
        highcut=config.EEG_HIGHCUT,
        notch_freq=config.NOTCH_FREQ,
        notch_q=config.NOTCH_Q,
        order=config.FILTER_ORDER,
    )
