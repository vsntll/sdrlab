"""FFT-based spectral analysis: periodogram PSD, Welch PSD, and spectrograms.

The heavy transforms run on the active backend (GPU under CuPy). Results are
returned as host NumPy arrays ready for matplotlib.
"""
from __future__ import annotations

import numpy as np

from .backend import asnumpy, fft, signal, timed, xp


def _window(name: str, n: int):
    # scipy.signal.get_window returns NumPy; cupyx.scipy.signal returns CuPy.
    return xp.asarray(signal().get_window(name, n), dtype=xp.float64)


def periodogram_psd(iq, fs: float, nfft: int | None = None, window: str = "hann",
                    timing: dict | None = None):
    """Single-shot FFT power spectral density of a complex capture.

    Returns ``(freqs_hz, psd_db)`` with ``freqs`` centred on 0 (fftshift order).
    ``psd_db`` is 10*log10 of power, normalised for window gain.
    """
    x = xp.asarray(iq, dtype=xp.complex64)
    n = min(int(nfft or x.size), int(x.size))
    x = x[:n]
    w = _window(window, n)
    win_pow = xp.sum(w * w)

    F = fft()
    with timed("periodogram FFT", timing):
        X = F.fftshift(F.fft(x * w, n=n))
        psd = (xp.abs(X) ** 2) / (fs * win_pow)
        psd_db = 10.0 * xp.log10(psd + 1e-20)

    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    return freqs, asnumpy(psd_db)


def welch_psd(iq, fs: float, nperseg: int = 4096, noverlap: int | None = None,
              window: str = "hann", timing: dict | None = None):
    """Averaged (Welch) PSD - lower variance than a single periodogram."""
    x = xp.asarray(iq, dtype=xp.complex64)
    S = signal()
    if noverlap is None:
        noverlap = nperseg // 2
    with timed("Welch PSD", timing):
        f, pxx = S.welch(
            x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
            return_onesided=False, detrend=False, scaling="density",
        )
        pxx_db = 10.0 * xp.log10(xp.abs(pxx) + 1e-20)
    f = asnumpy(f)
    order = np.argsort(f)
    return f[order], asnumpy(pxx_db)[order]


def spectrogram_db(iq, fs: float, nperseg: int = 2048, noverlap: int | None = None,
                   window: str = "hann", timing: dict | None = None):
    """Return ``(t_s, f_hz, S_db)`` with frequency in fftshift order for a waterfall."""
    x = xp.asarray(iq, dtype=xp.complex64)
    S = signal()
    if noverlap is None:
        noverlap = nperseg // 4
    with timed("spectrogram STFT", timing):
        f, t, Sxx = S.spectrogram(
            x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
            return_onesided=False, detrend=False, mode="psd",
        )
        S_db = 10.0 * xp.log10(xp.abs(Sxx) + 1e-20)
    f = asnumpy(f)
    order = np.argsort(f)
    return asnumpy(t), f[order], asnumpy(S_db)[order, :]


def peak_frequencies(freqs, psd_db, n_peaks: int = 5, min_separation_hz: float = 50e3,
                     dynamic_range_db: float = 40.0, prominence_db: float = 15.0):
    """Crude peak picker for labelling carriers in the analyzer output.

    A bin qualifies only if it is within ``dynamic_range_db`` of the strongest
    bin *and* at least ``prominence_db`` above the median (noise floor), so a
    quiet or noisy capture returns fewer than ``n_peaks``.
    """
    floor = float(np.median(psd_db))
    threshold = max(float(np.max(psd_db)) - dynamic_range_db, floor + prominence_db)
    idx = np.argsort(psd_db)[::-1]
    chosen: list[int] = []
    for i in idx:
        if psd_db[i] < threshold:
            break
        if all(abs(freqs[i] - freqs[j]) > min_separation_hz for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= n_peaks:
            break
    chosen.sort(key=lambda j: freqs[j])
    return [(float(freqs[j]), float(psd_db[j])) for j in chosen]
