"""Spectrum-analyzer demo: FFT periodogram, Welch PSD, and a waterfall.

    python scripts/demo_spectrum.py                 # synthetic multitone capture
    python scripts/demo_spectrum.py --iq cap.bin --fmt cu8 --fs 2.4e6 --fc 100.1e6
    SDRLAB_FORCE_CPU=1 python scripts/demo_spectrum.py     # CPU reference run
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sdrlab import backend, sources, spectral  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iq", help="recorded IQ file (default: synthesize a capture)")
    p.add_argument("--fmt", default="cu8", choices=["cu8", "ci16", "cf32"])
    p.add_argument("--fs", type=float, default=2.4e6, help="sample rate [Hz]")
    p.add_argument("--fc", type=float, default=0.0, help="center frequency for axis labels [Hz]")
    p.add_argument("--nfft", type=int, default=1 << 18)
    p.add_argument("--duration", type=float, default=0.4)
    args = p.parse_args()

    print(f"backend: {backend.describe()}")
    backend.warmup()

    if args.iq:
        iq = sources.read_iq_file(args.iq, args.fmt)
        fs = args.fs
        print(f"loaded {iq.size:,} samples from {args.iq}")
    else:
        iq, fs = sources.multitone_capture(duration=args.duration, fs=args.fs)
        print(f"synthesized {iq.size:,} samples @ {fs/1e6:.2f} MS/s")

    timing: dict[str, float] = {}
    f_per, psd = spectral.periodogram_psd(iq, fs, nfft=min(args.nfft, iq.size), timing=timing)
    f_w, pxx = spectral.welch_psd(iq, fs, nperseg=8192, timing=timing)
    t_sp, f_sp, S_db = spectral.spectrogram_db(iq, fs, nperseg=4096, timing=timing)

    fc = args.fc
    peaks = spectral.peak_frequencies(f_per, psd, n_peaks=5)
    print("\nstrongest carriers:")
    for fpk, ppk in peaks:
        print(f"  {(fc + fpk)/1e6:10.4f} MHz   {ppk:7.1f} dB")

    to_mhz = lambda x: (fc + x) / 1e6  # noqa: E731
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))

    ax[0].plot(to_mhz(f_per), psd, lw=0.6, color="#1f77b4")
    for fpk, ppk in peaks:
        ax[0].annotate(f"{to_mhz(fpk):.3f}", (to_mhz(fpk), ppk),
                       textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax[0].set_title(f"FFT periodogram PSD  (N={min(args.nfft, iq.size):,}, Hann)")
    ax[0].set(ylabel="dB/Hz", xlabel="MHz" if fc else "baseband MHz")
    ax[0].grid(alpha=0.3)

    ax[1].plot(to_mhz(f_w), pxx, lw=0.9, color="#d62728")
    ax[1].set_title("Welch averaged PSD  (nperseg=8192, 50% overlap)")
    ax[1].set(ylabel="dB/Hz", xlabel="MHz" if fc else "baseband MHz")
    ax[1].grid(alpha=0.3)

    extent = [t_sp[0], t_sp[-1], to_mhz(f_sp[0]), to_mhz(f_sp[-1])]
    im = ax[2].imshow(S_db, origin="lower", aspect="auto", extent=extent,
                      cmap="turbo", vmax=float(S_db.max()),
                      vmin=np.percentile(S_db, 75))
    ax[2].set_title("Waterfall / spectrogram")
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("MHz" if fc else "baseband MHz")
    fig.colorbar(im, ax=ax[2], label="dB")

    fig.suptitle(f"Spectrum analyzer  -  {backend.describe()}", fontsize=11)
    fig.tight_layout()
    out = OUT / "spectrum.png"
    fig.savefig(out, dpi=110)
    print(f"\ntiming: " + "  ".join(f"{k}={v*1e3:.1f}ms" for k, v in timing.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
