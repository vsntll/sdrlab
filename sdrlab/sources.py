"""IQ signal sources: synthetic baseband captures, file readers, and RTL-SDR.

All generators return ``complex64`` arrays on the active backend (GPU when
CuPy is available). Sample rates and frequencies are in Hz.
"""
from __future__ import annotations

import numpy as np

from .backend import USING_GPU, xp

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Message / audio synthesis                                                    #
# --------------------------------------------------------------------------- #
def synth_message(duration: float, fs: float, kind: str = "tones", seed: int = 0):
    """A real-valued baseband "audio" message evaluated on the fs-rate grid.

    ``kind='tones'`` sums a few sinusoids; ``kind='chirp'`` sweeps 200 Hz..8 kHz.
    Returned peak amplitude is 1.0.
    """
    n = int(round(duration * fs))
    t = xp.arange(n, dtype=xp.float64) / fs
    if kind == "chirp":
        f0, f1 = 200.0, 8000.0
        k = (f1 - f0) / duration
        msg = xp.sin(TWO_PI * (f0 * t + 0.5 * k * t * t))
    elif kind == "tones":
        msg = (
            1.00 * xp.sin(TWO_PI * 440.0 * t)
            + 0.50 * xp.sin(TWO_PI * 1200.0 * t)
            + 0.30 * xp.sin(TWO_PI * 3300.0 * t)
        )
        # slow tremolo so the spectrogram shows time structure
        msg *= 0.8 + 0.2 * xp.sin(TWO_PI * 2.0 * t)
    else:  # pragma: no cover
        raise ValueError(f"unknown message kind: {kind!r}")
    msg = msg / xp.max(xp.abs(msg))
    return msg.astype(xp.float32)


# --------------------------------------------------------------------------- #
# Modulators                                                                   #
# --------------------------------------------------------------------------- #
def fm_modulate(message, fs: float, deviation: float):
    """Frequency-modulate a real message onto a complex baseband carrier."""
    msg = xp.asarray(message, dtype=xp.float64)
    phase = TWO_PI * deviation * xp.cumsum(msg) / fs
    return xp.exp(1j * phase).astype(xp.complex64)


def am_modulate(message, fs: float, mod_index: float = 0.6):
    """Standard (DSB) AM with carrier: (1 + m*msg) on a complex baseband carrier."""
    msg = xp.asarray(message, dtype=xp.float64)
    env = 1.0 + mod_index * msg
    return env.astype(xp.complex64)


# --------------------------------------------------------------------------- #
# Composite captures                                                           #
# --------------------------------------------------------------------------- #
def _tone(n, fs, freq, amp=1.0, phase=0.0):
    k = xp.arange(n, dtype=xp.float64)
    return (amp * xp.exp(1j * (TWO_PI * freq * k / fs + phase))).astype(xp.complex64)


def noise(n: int, power_db: float = -30.0, seed: int = 0):
    """Complex AWGN with the given power in dB relative to unit amplitude."""
    rng = np.random.default_rng(seed)
    sigma = float(10.0 ** (power_db / 20.0) / np.sqrt(2.0))
    re = rng.standard_normal(n).astype(np.float32)
    im = rng.standard_normal(n).astype(np.float32)
    z = (re + 1j * im) * sigma
    return xp.asarray(z.astype(np.complex64))


def multitone_capture(
    duration: float = 0.4,
    fs: float = 2.4e6,
    tones=((-600e3, 0.8), (-120e3, 1.0), (300e3, 0.5), (850e3, 0.35)),
    noise_db: float = -32.0,
    seed: int = 1,
):
    """Wideband capture of several CW carriers plus noise - feeds the analyzer."""
    n = int(round(duration * fs))
    x = noise(n, noise_db, seed)
    for freq, amp in tones:
        x = x + _tone(n, fs, freq, amp)
    return x.astype(xp.complex64), fs


def fm_broadcast_capture(
    duration: float = 4.0,
    fs: float = 1.20e6,
    f_offset: float = -250e3,
    deviation: float = 75e3,
    message_kind: str = "tones",
    interferer: bool = True,
    noise_db: float = -34.0,
    seed: int = 2,
):
    """A WBFM "station" placed at ``f_offset`` inside a wider captured band.

    Returns ``(iq, fs, meta)`` where ``meta`` carries the parameters the
    receiver needs (offset, deviation) and the clean reference message.
    """
    n = int(round(duration * fs))
    message = synth_message(duration, fs, message_kind, seed)
    baseband = fm_modulate(message, fs, deviation)

    k = xp.arange(n, dtype=xp.float64)
    station = baseband * xp.exp(1j * TWO_PI * f_offset * k / fs).astype(xp.complex64)

    x = station.astype(xp.complex64) + noise(n, noise_db, seed)
    if interferer:
        # An adjacent narrowband FM station 400 kHz above, plus a CW pilot.
        other_msg = synth_message(duration, fs, "chirp", seed + 7)
        other = fm_modulate(other_msg, fs, 25e3)
        other = other * xp.exp(1j * TWO_PI * (f_offset + 400e3) * k / fs).astype(xp.complex64)
        x = x + 0.7 * other.astype(xp.complex64)
        x = x + 0.2 * _tone(n, fs, f_offset + 700e3, 1.0)

    meta = {
        "fs": fs,
        "f_offset": f_offset,
        "deviation": deviation,
        "message": message,
        "message_fs": fs,
        "duration": duration,
    }
    return x.astype(xp.complex64), fs, meta


# --------------------------------------------------------------------------- #
# File + hardware readers                                                       #
# --------------------------------------------------------------------------- #
def read_iq_file(path: str, fmt: str = "cu8", max_samples: int | None = None):
    """Load recorded IQ. ``fmt`` is 'cu8' (rtl_sdr), 'ci16', or 'cf32'."""
    if fmt == "cu8":
        raw = np.fromfile(path, dtype=np.uint8, count=-1 if max_samples is None else 2 * max_samples)
        iq = (raw.astype(np.float32) - 127.5) / 127.5
        iq = iq[0::2] + 1j * iq[1::2]
    elif fmt == "ci16":
        raw = np.fromfile(path, dtype=np.int16, count=-1 if max_samples is None else 2 * max_samples)
        iq = raw.astype(np.float32) / 32768.0
        iq = iq[0::2] + 1j * iq[1::2]
    elif fmt == "cf32":
        iq = np.fromfile(path, dtype=np.complex64, count=-1 if max_samples is None else max_samples)
    else:  # pragma: no cover
        raise ValueError(f"unknown IQ format: {fmt!r}")
    return xp.asarray(iq.astype(np.complex64))


def read_rtl_sdr(
    n_samples: int,
    center_freq: float = 100.1e6,
    sample_rate: float = 2.4e6,
    gain="auto",
):  # pragma: no cover - requires hardware
    """Read a block from a real RTL-SDR dongle (needs ``pyrtlsdr`` + hardware)."""
    try:
        from rtlsdr import RtlSdr
    except ImportError as exc:
        raise RuntimeError("pyrtlsdr is not installed: pip install pyrtlsdr") from exc

    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        sdr.gain = gain
        samples = sdr.read_samples(n_samples)
    finally:
        sdr.close()
    return xp.asarray(np.asarray(samples, dtype=np.complex64))
