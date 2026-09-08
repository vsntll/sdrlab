"""WBFM receiver demo: synthesize a broadcast capture, demodulate to a WAV.

    python scripts/demo_fm.py                       # synthetic station
    python scripts/demo_fm.py --iq fm.bin --fmt cu8 --fs 2.4e6 --offset -250e3
    SDRLAB_FORCE_CPU=1 python scripts/demo_fm.py    # CPU reference run

Produces outputs/fm_audio.wav and outputs/fm_chain.png (spectra + spectrograms
before and after demodulation).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sdrlab import backend, modulation, sources, spectral  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iq", help="recorded IQ file (default: synthesize a station)")
    p.add_argument("--fmt", default="cu8", choices=["cu8", "ci16", "cf32"])
    p.add_argument("--fs", type=float, default=1.20e6, help="capture sample rate [Hz]")
    p.add_argument("--offset", type=float, default=-250e3, help="station offset from center [Hz]")
    p.add_argument("--deviation", type=float, default=75e3)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--message", default="tones", choices=["tones", "chirp"])
    p.add_argument("--audio-fs", type=float, default=48e3)
    args = p.parse_args()

    print(f"backend: {backend.describe()}")
    backend.warmup()

    ref_msg = ref_fs = None
    if args.iq:
        iq = sources.read_iq_file(args.iq, args.fmt)
        fs = args.fs
        print(f"loaded {iq.size:,} samples from {args.iq}")
    else:
        iq, fs, meta = sources.fm_broadcast_capture(
            duration=args.duration, fs=args.fs, f_offset=args.offset,
            deviation=args.deviation, message_kind=args.message,
        )
        ref_msg, ref_fs = meta["message"], meta["message_fs"]
        print(f"synthesized {iq.size:,} samples @ {fs/1e6:.2f} MS/s, "
              f"station at {args.offset/1e3:+.0f} kHz")

    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    audio, afs = modulation.fm_receiver(
        iq, fs, args.offset, deviation=args.deviation, audio_fs=args.audio_fs, timing=timing,
    )
    wall = time.perf_counter() - t0

    wav = OUT / "fm_audio.wav"
    modulation.write_wav(str(wav), audio, afs)
    print(f"\ndemod wall time: {wall*1e3:.1f} ms for {args.duration:.1f}s of audio "
          f"({args.duration/wall:.0f}x realtime)")
    if ref_msg is not None:
        snr = modulation.audio_quality(audio, ref_msg, afs, ref_fs)
        print(f"recovered-audio SNR vs reference message: {snr:.1f} dB")
    print(f"wrote {wav}")

    # --- figure: RF spectrum, audio spectrum, and both spectrograms ----------
    f_rf, psd_rf = spectral.periodogram_psd(iq, fs, nfft=min(1 << 18, iq.size))
    f_au, psd_au = spectral.periodogram_psd(
        audio.astype(np.complex64), afs, nfft=min(1 << 15, audio.size))
    t_rf, ff_rf, S_rf = spectral.spectrogram_db(iq, fs, nperseg=4096)
    t_au, ff_au, S_au = spectral.spectrogram_db(audio.astype(np.complex64), afs, nperseg=1024)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(f_rf / 1e3, psd_rf, lw=0.6)
    ax[0, 0].axvspan((args.offset - args.deviation * 2) / 1e3,
                     (args.offset + args.deviation * 2) / 1e3, color="orange", alpha=0.2,
                     label="tuned channel")
    ax[0, 0].set(title="captured RF spectrum", xlabel="kHz from center", ylabel="dB/Hz")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    half = psd_au[len(psd_au) // 2:]
    fpos = f_au[len(f_au) // 2:]
    ax[0, 1].plot(fpos / 1e3, half, lw=0.8, color="#2ca02c")
    ax[0, 1].set(title="recovered audio spectrum", xlabel="kHz", ylabel="dB/Hz",
                 xlim=(0, afs / 2e3))
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].imshow(S_rf, origin="lower", aspect="auto", cmap="turbo",
                    extent=[t_rf[0], t_rf[-1], ff_rf[0] / 1e3, ff_rf[-1] / 1e3],
                    vmin=np.percentile(S_rf, 40), vmax=np.percentile(S_rf, 99.5))
    ax[1, 0].set(title="RF waterfall", xlabel="s", ylabel="kHz")

    ax[1, 1].imshow(S_au[len(ff_au) // 2:], origin="lower", aspect="auto", cmap="magma",
                    extent=[t_au[0], t_au[-1], 0, afs / 2e3],
                    vmin=np.percentile(S_au, 55), vmax=np.percentile(S_au, 99.5))
    ax[1, 1].set(title="audio spectrogram", xlabel="s", ylabel="kHz")

    fig.suptitle(f"FM demodulation chain  -  {backend.describe()}", fontsize=11)
    fig.tight_layout()
    png = OUT / "fm_chain.png"
    fig.savefig(png, dpi=110)
    print("stage timing: " + "  ".join(f"{k}={v*1e3:.1f}ms" for k, v in timing.items()))
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
