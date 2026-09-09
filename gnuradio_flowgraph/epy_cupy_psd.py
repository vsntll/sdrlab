#!/usr/bin/env python3
"""GNU Radio Embedded Python Block: GPU (CuPy) FFT power-spectrum.

Drop this into a GRC flowgraph as an "Embedded Python Block" (or import it in a
hand-written flowgraph). It takes a complex stream and emits, for every
``fft_size`` input samples, ``fft_size`` float values: the windowed power
spectrum in dB, in fftshift (DC-centred) order.

The transform runs on the GPU through CuPy when it is importable and a CUDA
device is present; otherwise it transparently falls back to NumPy, so the same
block works on machines without a GPU.

This is the GNU Radio bridge to the standalone pipeline in ``sdrlab/``.
"""
from __future__ import annotations

import numpy as np
from gnuradio import gr

try:
    import cupy as _cp

    _cp.cuda.runtime.getDeviceCount()
    _XP = _cp
    _ON_GPU = True
except Exception:  # pragma: no cover - depends on host
    _XP = np
    _ON_GPU = False


class blk(gr.sync_block):
    """Windowed |FFT|^2 -> dB, computed on the GPU via CuPy."""

    def __init__(self, fft_size: int = 4096, window: str = "hann",
                 sample_rate: float = 1.0):
        gr.sync_block.__init__(
            self, name=f"CuPy PSD ({'GPU' if _ON_GPU else 'CPU'})",
            in_sig=[np.complex64], out_sig=[np.float32],
        )
        self.fft_size = int(fft_size)
        self.sample_rate = float(sample_rate)
        w = np.hanning(self.fft_size) if window == "hann" else np.ones(self.fft_size)
        self._win = _XP.asarray(w.astype(np.float32))
        self._win_pow = float((w * w).sum())
        self.set_output_multiple(self.fft_size)

    def freqs(self) -> np.ndarray:
        """Bin centre frequencies (Hz), matching the fftshift output order."""
        return np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0 / self.sample_rate))

    def work(self, input_items, output_items):
        x = input_items[0]
        out = output_items[0]
        n_frames = min(len(x), len(out)) // self.fft_size
        if n_frames == 0:
            return 0

        buf = _XP.asarray(x[: n_frames * self.fft_size]).reshape(n_frames, self.fft_size)
        X = _XP.fft.fftshift(_XP.fft.fft(buf * self._win, axis=1), axes=1)
        psd = _XP.abs(X) ** 2 / (self.sample_rate * self._win_pow)
        psd_db = (10.0 * _XP.log10(psd + 1e-20)).astype(_XP.float32)

        result = _cp.asnumpy(psd_db) if _ON_GPU else psd_db
        out[: n_frames * self.fft_size] = result.reshape(-1)
        return n_frames * self.fft_size


# Quick smoke test (run with the radioconda interpreter): python epy_cupy_psd.py
if __name__ == "__main__":
    b = blk(1024, sample_rate=1e6)
    sig = (np.exp(2j * np.pi * 100e3 * np.arange(4096) / 1e6)
           + 0.01 * (np.random.randn(4096) + 1j * np.random.randn(4096))).astype(np.complex64)
    o = np.zeros(4096, dtype=np.float32)
    n = b.work([sig], [o])
    f = b.freqs()
    peak = f[np.argmax(o[:1024])]
    print(f"backend: {'GPU (CuPy)' if _ON_GPU else 'CPU (NumPy)'}, "
          f"processed {n} samples, peak bin at {peak/1e3:.1f} kHz (expected 100.0)")
