"""Filter showcase: design FIR (window + Parks-McClellan) and IIR filters,
plot their responses, and apply them on the backend to a noisy multitone.

    python scripts/demo_filters.py
    SDRLAB_FORCE_CPU=1 python scripts/demo_filters.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sdrlab import backend, filters, sources, spectral  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    print(f"backend: {backend.describe()}")
    backend.warmup()

    fs = 2.4e6
    cutoff = 200e3
    iq, _ = sources.multitone_capture(
        duration=0.3, fs=fs,
        tones=((-700e3, 0.7), (-90e3, 1.0), (60e3, 0.9), (500e3, 0.8)),
        noise_db=-20.0,
    )

    fir_win = filters.fir_lowpass(fs, cutoff, numtaps=201, window="hann")
    fir_eq = filters.fir_lowpass_remez(fs, 170e3, 260e3, numtaps=201)
    sos_butter = filters.iir_lowpass_sos(fs, cutoff, order=8, kind="butter")
    sos_ellip = filters.iir_lowpass_sos(fs, cutoff, order=6, kind="ellip")

    timing: dict[str, float] = {}
    y_win = filters.apply_fir(iq, fir_win, timing=timing, label="FIR (Hann) apply")
    y_eq = filters.apply_fir(iq, fir_eq, timing=timing, label="FIR (remez) apply")
    y_bw = filters.apply_iir(iq, sos_butter, timing=timing, label="IIR (butter) apply")
    y_el = filters.apply_iir(iq, sos_ellip, timing=timing, label="IIR (ellip) apply")

    # ---- responses -------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    for taps, name, c in [(fir_win, "FIR Hann 201", "#1f77b4"),
                          (fir_eq, "FIR remez 201", "#ff7f0e")]:
        w, h = filters.fir_response(taps, fs)
        ax[0, 0].plot(w / 1e3, 20 * np.log10(np.abs(h) + 1e-9), label=name, color=c)
    for sos, name, c in [(sos_butter, "IIR butter-8", "#2ca02c"),
                         (sos_ellip, "IIR ellip-6", "#d62728")]:
        w, h = filters.sos_response(sos, fs)
        ax[0, 0].plot(w / 1e3, 20 * np.log10(np.abs(h) + 1e-9), label=name, color=c)
    ax[0, 0].axvline(cutoff / 1e3, ls="--", color="k", lw=0.8)
    ax[0, 0].set(title="magnitude response", xlabel="kHz", ylabel="dB",
                 xlim=(0, 600), ylim=(-100, 5))
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    for taps, name, c in [(fir_win, "FIR Hann", "#1f77b4"), (fir_eq, "FIR remez", "#ff7f0e")]:
        w, h = filters.fir_response(taps, fs)
        ax[0, 1].plot(w / 1e3, np.unwrap(np.angle(h)), label=name, color=c)
    for sos, name, c in [(sos_butter, "IIR butter-8", "#2ca02c"),
                         (sos_ellip, "IIR ellip-6", "#d62728")]:
        w, h = filters.sos_response(sos, fs)
        ax[0, 1].plot(w / 1e3, np.unwrap(np.angle(h)), label=name, color=c)
    ax[0, 1].set(title="phase response (FIR is linear-phase)", xlabel="kHz",
                 ylabel="rad", xlim=(0, 400))
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

    # ---- effect on the signal ------------------------------------------------
    f0, p_in = spectral.periodogram_psd(iq, fs, nfft=1 << 16)
    _, p_fir = spectral.periodogram_psd(y_win, fs, nfft=1 << 16)
    _, p_iir = spectral.periodogram_psd(y_bw, fs, nfft=1 << 16)
    ax[1, 0].plot(f0 / 1e3, p_in, lw=0.5, color="0.6", label="input")
    ax[1, 0].plot(f0 / 1e3, p_fir, lw=0.7, color="#1f77b4", label="after FIR Hann")
    ax[1, 0].axvline(cutoff / 1e3, ls="--", color="k", lw=0.8)
    ax[1, 0].set(title="spectrum: FIR low-pass", xlabel="kHz", ylabel="dB/Hz")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(f0 / 1e3, p_in, lw=0.5, color="0.6", label="input")
    ax[1, 1].plot(f0 / 1e3, p_iir, lw=0.7, color="#2ca02c", label="after IIR butter-8")
    ax[1, 1].axvline(cutoff / 1e3, ls="--", color="k", lw=0.8)
    ax[1, 1].set(title="spectrum: IIR low-pass", xlabel="kHz", ylabel="dB/Hz")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

    fig.suptitle(f"FIR vs IIR filtering  -  {backend.describe()}", fontsize=11)
    fig.tight_layout()
    png = OUT / "filters.png"
    fig.savefig(png, dpi=110)
    print("apply timing: " + "  ".join(f"{k}={v*1e3:.2f}ms" for k, v in timing.items()))
    print(f"wrote {png}")
    print("\nnote: the FIR path parallelises well on the GPU (FFT convolution);")
    print("      sosfilt is a sequential recurrence, so IIR sees far less speedup.")


if __name__ == "__main__":
    main()
