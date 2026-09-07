"""Modulation / demodulation chains built from the filter + backend primitives."""
from __future__ import annotations

import numpy as np

from .backend import asnumpy, signal, timed, xp
from .filters import apply_fir, apply_iir, deemphasis_sos, fir_lowpass, lowpass_decimate

TWO_PI = 2.0 * np.pi


def frequency_shift(iq, fs: float, shift_hz: float):
    """Multiply by a complex exponential to bring ``shift_hz`` down to DC."""
    x = xp.asarray(iq, dtype=xp.complex64)
    k = xp.arange(x.size, dtype=xp.float64)
    lo = xp.exp(-1j * TWO_PI * shift_hz * k / fs).astype(xp.complex64)
    return x * lo


def quadrature_demod(iq, fs: float, gain: float | None = None):
    """Discriminator: instantaneous frequency from the phase of x[n]*conj(x[n-1]).

    With ``gain=None`` the output is instantaneous frequency in Hz; pass
    ``gain=1/deviation`` (or ``fs/(2*pi*deviation)``) to recover a unit message.
    """
    x = xp.asarray(iq, dtype=xp.complex64)
    prod = x[1:] * xp.conj(x[:-1])
    disc = xp.angle(prod)
    if gain is None:
        return disc * (fs / TWO_PI)
    return disc * gain


def am_demod(iq, dc_block: bool = True):
    """Envelope detector for AM."""
    x = xp.asarray(iq)
    env = xp.abs(x)
    if dc_block:
        env = env - xp.mean(env)
    return env


def fm_receiver(
    iq,
    fs_in: float,
    f_offset: float,
    deviation: float = 75e3,
    channel_bw: float = 200e3,
    audio_fs: float = 48e3,
    audio_bw: float = 15e3,
    deemphasis_tau: float = 75e-6,
    timing: dict | None = None,
):
    """Full WBFM broadcast receiver: tune -> channelize -> discriminate -> audio.

    Returns ``(audio, audio_fs)`` with ``audio`` a host float32 array in [-1, 1].

    Pipeline
    --------
    1. Digital down-conversion to baseband (complex mixer).
    2. FIR channel-select low-pass + integer decimation to an IF rate.
    3. Quadrature (polar) discriminator.
    4. One-pole IIR de-emphasis.
    5. FIR audio low-pass + decimation to ``audio_fs``.
    6. DC removal and peak normalisation.
    """
    x = frequency_shift(iq, fs_in, f_offset)

    # Stage 2: channel selection. Pick a decimation that lands near 4x audio_fs.
    if_target = max(4.0 * audio_fs, 2.2 * channel_bw)
    decim1 = max(1, int(fs_in // if_target))
    x, fs_if = lowpass_decimate(
        x, fs_in, channel_bw / 2.0, decim1, numtaps=257,
        timing=timing, label=f"channel LPF /{decim1}",
    )

    # Stage 3: discriminator, scaled so full deviation -> +-1.0.
    with timed("quadrature demod", timing):
        audio_if = quadrature_demod(x, fs_if, gain=1.0 / deviation)

    # Stage 4: de-emphasis (IIR).
    audio_if = apply_iir(audio_if, deemphasis_sos(fs_if, deemphasis_tau),
                         timing=timing, label="de-emphasis IIR")

    # Stage 5: audio-band low-pass, then rational resample to exactly audio_fs.
    from math import gcd

    audio_if = apply_fir(audio_if, fir_lowpass(fs_if, audio_bw, numtaps=257),
                         timing=timing, label="audio LPF")
    g = gcd(int(round(fs_if)), int(round(audio_fs)))
    up, down = int(round(audio_fs)) // g, int(round(fs_if)) // g
    with timed(f"resample {down}/{up} -> audio", timing):
        audio = signal().resample_poly(audio_if, up, down)
    fs_a = audio_fs

    # Stage 6: clean up.
    audio = audio - xp.mean(audio)
    peak = xp.max(xp.abs(audio))
    audio = xp.where(peak > 0, audio / peak, audio)
    return asnumpy(audio).astype(np.float32), fs_a


def write_wav(path: str, audio, fs: float) -> None:
    """Write a mono 16-bit PCM WAV (pure stdlib, no SciPy needed)."""
    import wave

    a = np.asarray(asnumpy(audio), dtype=np.float64)
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(round(fs)))
        w.writeframes(pcm.tobytes())


def audio_quality(recovered, reference, fs_rec: float, fs_ref: float):
    """Rough SNR (dB) between the recovered audio and a resampled reference."""
    import scipy.signal as sp

    ref = np.asarray(asnumpy(reference), dtype=np.float64)
    rec = np.asarray(asnumpy(recovered), dtype=np.float64)
    ref = sp.resample_poly(ref, int(fs_rec), int(fs_ref))
    n = min(ref.size, rec.size)
    ref, rec = ref[:n], rec[:n]
    ref = (ref - ref.mean()) / (np.std(ref) + 1e-12)
    rec = (rec - rec.mean()) / (np.std(rec) + 1e-12)
    lag = int(np.argmax(sp.correlate(rec, ref, mode="same")) - n // 2)
    if lag > 0:
        rec, ref = rec[lag:], ref[:-lag or None]
    elif lag < 0:
        rec, ref = rec[:lag], ref[-lag:]
    scale = np.dot(rec, ref) / np.dot(ref, ref)
    err = rec - scale * ref
    return 10.0 * np.log10(np.sum((scale * ref) ** 2) / (np.sum(err ** 2) + 1e-20))
