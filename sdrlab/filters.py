"""Filter design (FIR + IIR) and GPU-accelerated application.

Design happens on the CPU with SciPy (cheap, and SciPy has ``remez``); the
resulting small coefficient arrays are pushed to the active backend and the
actual filtering / resampling runs there.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as sp_signal

from .backend import asnumpy, signal, timed, xp


# --------------------------------------------------------------------------- #
# Design (CPU)                                                                 #
# --------------------------------------------------------------------------- #
def fir_lowpass(fs: float, cutoff_hz: float, numtaps: int = 129, window: str = "hann"):
    """Windowed-sinc FIR low-pass. Returns real taps (NumPy)."""
    return sp_signal.firwin(numtaps, cutoff_hz, fs=fs, window=window).astype(np.float64)


def fir_lowpass_remez(fs: float, pass_hz: float, stop_hz: float, numtaps: int = 129):
    """Parks-McClellan (equiripple) FIR low-pass. Returns real taps (NumPy)."""
    return sp_signal.remez(
        numtaps, [0, pass_hz, stop_hz, fs / 2], [1, 0], fs=fs
    ).astype(np.float64)


def fir_bandpass(fs: float, low_hz: float, high_hz: float, numtaps: int = 257,
                 window: str = "hann"):
    return sp_signal.firwin(
        numtaps, [low_hz, high_hz], fs=fs, window=window, pass_zero=False
    ).astype(np.float64)


def iir_lowpass_sos(fs: float, cutoff_hz: float, order: int = 6, kind: str = "butter"):
    """IIR low-pass as second-order sections. ``kind`` in butter/cheby1/ellip."""
    if kind == "butter":
        return sp_signal.butter(order, cutoff_hz, fs=fs, output="sos")
    if kind == "cheby1":
        return sp_signal.cheby1(order, 1.0, cutoff_hz, fs=fs, output="sos")
    if kind == "ellip":
        return sp_signal.ellip(order, 1.0, 60.0, cutoff_hz, fs=fs, output="sos")
    raise ValueError(f"unknown IIR kind: {kind!r}")  # pragma: no cover


def deemphasis_sos(fs: float, tau: float = 75e-6):
    """One-pole de-emphasis (75 us in the Americas/Korea, 50 us elsewhere)."""
    a1 = np.exp(-1.0 / (fs * tau))
    b = np.array([1.0 - a1, 0.0])
    a = np.array([1.0, -a1])
    return sp_signal.tf2sos(b, a)


# --------------------------------------------------------------------------- #
# Frequency response (CPU, for plots)                                          #
# --------------------------------------------------------------------------- #
def fir_response(taps, fs: float, n: int = 8192):
    w, h = sp_signal.freqz(asnumpy(taps), worN=n, fs=fs)
    return w, h


def sos_response(sos, fs: float, n: int = 8192):
    w, h = sp_signal.sosfreqz(sos, worN=n, fs=fs)
    return w, h


# --------------------------------------------------------------------------- #
# Application (active backend / GPU)                                           #
# --------------------------------------------------------------------------- #
def apply_fir(x, taps, mode: str = "same", timing: dict | None = None, label="FIR filter"):
    """Convolve ``x`` with ``taps`` on the backend via overlap FFT convolution."""
    S = signal()
    xt = xp.asarray(x)
    tt = xp.asarray(np.asarray(taps), dtype=xt.real.dtype if xt.dtype.kind == "c" else xt.dtype)
    with timed(label, timing):
        y = S.fftconvolve(xt, tt, mode=mode)
    return y


def apply_iir(x, sos, timing: dict | None = None, label="IIR filter"):
    """Zero-phase-optional causal IIR via second-order sections on the backend."""
    S = signal()
    xt = xp.asarray(x)
    st = xp.asarray(np.asarray(sos, dtype=np.float64))
    with timed(label, timing):
        y = S.sosfilt(st, xt)
    return y


def lowpass_decimate(x, fs: float, cutoff_hz: float, decim: int, numtaps: int = 193,
                     window: str = "hann", timing: dict | None = None, label=None):
    """Anti-alias FIR low-pass followed by integer decimation, run on the backend.

    Returns ``(y, fs / decim)``.
    """
    taps = fir_lowpass(fs, cutoff_hz, numtaps=numtaps, window=window)
    y = apply_fir(x, taps, mode="same", timing=timing,
                  label=label or f"LPF+decimate /{decim}")
    return y[::decim], fs / decim
