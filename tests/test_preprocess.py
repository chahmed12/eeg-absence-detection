"""Tests du pré-traitement du signal EEG.

Vérifie le filtrage passe-bande Butterworth (0,5–40 Hz, phase nulle) et le
coupe-bande notch à 50 Hz, ainsi que le point d'entrée preprocess_signal().
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp_signal

from src.preprocess import bandpass_notch, preprocess_signal
from src import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
FS = 256
N = FS * 10  # 10 secondes


def _sine(freq: float, n: int = N, fs: float = FS, amp: float = 1.0) -> np.ndarray:
    """Signal sinusoïdal pur à une fréquence donnée."""
    t = np.arange(n) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# bandpass_notch — forme du signal de sortie
# ---------------------------------------------------------------------------
def test_output_shape_mono():
    """Un signal mono (1D) ressort mono."""
    sig = _sine(3.0)
    out = bandpass_notch(sig, fs=FS)
    assert out.shape == sig.shape


def test_output_shape_multichannel():
    """Un signal multicanal (N, C) ressort avec la même forme."""
    sig = np.column_stack([_sine(3.0), _sine(10.0), _sine(25.0)])  # (N, 3)
    out = bandpass_notch(sig, fs=FS)
    assert out.shape == sig.shape


def test_output_dtype_is_float32():
    """La sortie est toujours en float32, indépendamment du dtype d'entrée."""
    sig = _sine(3.0).astype(np.float64)
    out = bandpass_notch(sig, fs=FS)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# bandpass_notch — filtrage passe-bande
# ---------------------------------------------------------------------------
def test_bandpass_preserves_inband_signal():
    """Un sinus à 3 Hz (dans la bande 0,5–40 Hz) est majoritairement conservé."""
    sig = _sine(3.0, amp=100.0)
    out = bandpass_notch(sig, fs=FS)
    # L'énergie de sortie doit être significative (> 10 % de l'entrée)
    assert np.var(out) > 0.1 * np.var(sig)


def test_bandpass_attenuates_dc():
    """Un signal DC (0 Hz) est fortement atténué par le passe-bas 0,5 Hz."""
    dc = np.ones(N, dtype=np.float32) * 1000.0
    out = bandpass_notch(dc, fs=FS)
    # L'énergie résiduelle doit être < 1 % de l'entrée
    assert np.var(out) < 0.01 * np.var(dc) + 1e-6


def test_bandpass_attenuates_high_frequency():
    """Un sinus à 80 Hz (hors bande) est fortement atténué."""
    hf = _sine(80.0, amp=100.0)
    out = bandpass_notch(hf, fs=FS)
    assert np.var(out) < 0.05 * np.var(hf) + 1e-6


# ---------------------------------------------------------------------------
# bandpass_notch — notch à 50 Hz
# ---------------------------------------------------------------------------
def test_notch_attenuates_50hz():
    """Un sinus à 50 Hz est atténué par le coupe-bande notch."""
    sig_50 = _sine(50.0, amp=100.0)
    # On court-circuite le passe-bande en fixant des bornes larges
    out = bandpass_notch(sig_50, fs=FS, lowcut=0.5, highcut=60.0,
                         notch_freq=50.0, notch_q=30.0)
    # Énergie résiduelle à 50 Hz doit être très faible
    freqs, psd = sp_signal.periodogram(out, fs=FS)
    idx_50 = np.argmin(np.abs(freqs - 50.0))
    idx_3 = np.argmin(np.abs(freqs - 3.0))
    # La puissance à 50 Hz doit être bien inférieure à celle à 3 Hz
    assert psd[idx_50] < psd[idx_3] * 10


# ---------------------------------------------------------------------------
# bandpass_notch — robustesse aux NaN / inf
# ---------------------------------------------------------------------------
def test_nan_handling():
    """Les NaN dans le signal d'entrée ne provoquent pas d'erreur."""
    sig = _sine(3.0)
    sig[100] = np.nan
    out = bandpass_notch(sig, fs=FS)
    assert out.shape == sig.shape
    assert np.isfinite(out).all()


def test_inf_handling():
    """Les valeurs inf dans le signal d'entrée ne provoquent pas d'erreur."""
    sig = _sine(3.0)
    sig[50] = np.inf
    out = bandpass_notch(sig, fs=FS)
    assert out.shape == sig.shape


# ---------------------------------------------------------------------------
# preprocess_signal — point d'entrée
# ---------------------------------------------------------------------------
def test_preprocess_signal_enabled(monkeypatch):
    """Quand PREPROCESS_ENABLED=True, appelle bandpass_notch."""
    monkeypatch.setattr(config, "PREPROCESS_ENABLED", True)
    sig = _sine(3.0, amp=50.0)
    out = preprocess_signal(sig, fs=FS)
    assert out.shape == sig.shape
    assert out.dtype == np.float32
    # Le signal doit avoir été modifié (filtrage non trivial)
    assert not np.allclose(out, sig.astype(np.float32), atol=1.0)


def test_preprocess_signal_disabled(monkeypatch):
    """Quand PREPROCESS_ENABLED=False, seule la conversion dtype est appliquée."""
    monkeypatch.setattr(config, "PREPROCESS_ENABLED", False)
    sig = _sine(3.0)
    out = preprocess_signal(sig, fs=FS)
    np.testing.assert_array_almost_equal(out, sig.astype(np.float32), decimal=5)


def test_preprocess_multichannel(monkeypatch):
    """preprocess_signal accepte les signaux multicanaux."""
    monkeypatch.setattr(config, "PREPROCESS_ENABLED", True)
    sig = np.column_stack([_sine(3.0), _sine(10.0)])
    out = preprocess_signal(sig, fs=FS)
    assert out.shape == sig.shape
    assert out.dtype == np.float32
