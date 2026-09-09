#!/usr/bin/env python3
"""GNU Radio spectrum analyzer whose FFT stage runs on the GPU via CuPy.

A simulated multitone complex source is fed through the ``epy_cupy_psd`` block
(CuPy FFT power spectrum, NumPy fallback), the per-frame spectra are averaged,
and the strongest carriers are printed. Run with the radioconda interpreter:

    C:\\Users\\avasa\\radioconda\\python.exe gnuradio_flowgraph/spectrum_gpu.py

This is the GNU Radio counterpart of ``scripts/demo_spectrum.py``.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from gnuradio import analog, blocks, gr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import epy_cupy_psd  # noqa: E402


class spectrum_gpu(gr.top_block):
    def __init__(self, samp_rate=2_400_000, fft_size=4096, n_frames=400,
                 tones=((-600e3, 0.8), (-120e3, 1.0), (300e3, 0.5), (850e3, 0.35))):
        gr.top_block.__init__(self, "GPU spectrum analyzer")
        self.psd = epy_cupy_psd.blk(fft_size=fft_size, window="hann", sample_rate=samp_rate)

        adder = blocks.add_vcc(1)
        for i, (freq, amp) in enumerate(tones):
            self.connect(
                analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, freq, amp),
                (adder, i))
        noise = analog.noise_source_c(analog.GR_GAUSSIAN, 10 ** (-32 / 20), 0)
        mix = blocks.add_vcc(1)
        self.connect(adder, (mix, 0))
        self.connect(noise, (mix, 1))

        head = blocks.head(gr.sizeof_gr_complex, fft_size * n_frames)
        self.sink = blocks.vector_sink_f()
        self.connect(mix, head, self.psd, self.sink)

    def averaged_psd_db(self, fft_size):
        frames = np.asarray(self.sink.data(), dtype=np.float64).reshape(-1, fft_size)
        lin = 10.0 ** (frames / 10.0)
        return 10.0 * np.log10(lin.mean(axis=0) + 1e-20)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samp-rate", type=float, default=2.4e6)
    p.add_argument("--fft-size", type=int, default=4096)
    p.add_argument("--frames", type=int, default=400)
    args = p.parse_args()

    tb = spectrum_gpu(args.samp_rate, args.fft_size, args.frames)
    tb.run()

    psd_db = tb.averaged_psd_db(args.fft_size)
    freqs = tb.psd.freqs()
    floor = np.median(psd_db)
    order = np.argsort(psd_db)[::-1]
    print(f"backend: {'GPU (CuPy)' if epy_cupy_psd._ON_GPU else 'CPU (NumPy)'}   "
          f"({args.frames} x {args.fft_size}-pt FFT averaged)")
    print("strongest carriers:")
    picked = []
    for i in order:
        if psd_db[i] < psd_db[order[0]] - 40:
            break
        if all(abs(freqs[i] - freqs[j]) > 50e3 for j in picked):
            picked.append(i)
        if len(picked) >= 6:
            break
    for i in sorted(picked, key=lambda j: freqs[j]):
        print(f"  {freqs[i]/1e3:+10.1f} kHz   {psd_db[i] - floor:6.1f} dB above floor")


if __name__ == "__main__":
    main()
