"""Benchmark the FFT and FIR-filtering stages: GPU (CuPy) vs CPU (NumPy/SciPy).

    python scripts/benchmark.py
    python scripts/benchmark.py --sizes 16 22 --reps 30

Runs the CPU path in a separate subprocess (SDRLAB_FORCE_CPU=1) so both
backends are measured from a clean state, then writes outputs/benchmark.png.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"


def _bench(log2_sizes, reps):
    import numpy as np

    from sdrlab import backend

    xp = backend.xp
    S = backend.signal()
    F = backend.fft()
    backend.warmup()
    taps = xp.asarray(np.hanning(128).astype(np.float32))

    rows = []
    for k in log2_sizes:
        n = 1 << k
        x = xp.asarray((np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64))

        F.fft(x); backend.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            X = F.fft(x)
        backend.synchronize()
        fft_ms = (time.perf_counter() - t0) / reps * 1e3

        S.fftconvolve(x, taps, mode="same"); backend.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            y = S.fftconvolve(x, taps, mode="same")
        backend.synchronize()
        fir_ms = (time.perf_counter() - t0) / reps * 1e3

        # round-trip including host<->device transfer (GPU only; ~memcpy on CPU)
        t0 = time.perf_counter()
        for _ in range(reps):
            xd = xp.asarray(backend.asnumpy(x))
            Xr = F.fft(xd)
            _ = backend.asnumpy(Xr)
        backend.synchronize()
        rt_ms = (time.perf_counter() - t0) / reps * 1e3

        rows.append({"n": n, "log2n": k, "fft_ms": fft_ms, "fir_ms": fir_ms, "roundtrip_ms": rt_ms})
        print(f"  2^{k:<2} ({n:>10,})  FFT {fft_ms:9.3f} ms   FIR {fir_ms:9.3f} ms   +xfer {rt_ms:9.3f} ms")
    return {"backend": backend.describe(), "gpu": backend.USING_GPU, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs=2, default=(12, 23), metavar=("LOG2_MIN", "LOG2_MAX"))
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--_emit", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    log2_sizes = list(range(args.sizes[0], args.sizes[1] + 1))

    if args._emit:
        print(json.dumps(_bench(log2_sizes, args.reps)))
        return

    from sdrlab import backend

    print(f"primary backend: {backend.describe()}")
    print("\n== primary run ==")
    primary = _bench(log2_sizes, args.reps)

    other_env = dict(os.environ)
    other_env["SDRLAB_FORCE_CPU"] = "1" if primary["gpu"] else "0"
    label = "CPU" if primary["gpu"] else "GPU"
    print(f"\n== {label} run (subprocess) ==")
    proc = subprocess.run(
        [sys.executable, __file__, "--_emit", "--sizes", str(args.sizes[0]),
         str(args.sizes[1]), "--reps", str(args.reps)],
        capture_output=True, text=True, env=other_env,
    )
    other = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            other = json.loads(line)
        else:
            print(line)
    if other is None:
        print(proc.stderr[-2000:])
        print("second backend unavailable; plotting primary only")

    gpu = primary if primary["gpu"] else other
    cpu = other if primary["gpu"] else primary

    _plot(gpu, cpu)


def _plot(gpu, cpu) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for j, (key, title) in enumerate([
        ("fft_ms", "FFT"), ("fir_ms", "FIR (128-tap FFT conv)"), ("roundtrip_ms", "FFT + host<->device")
    ]):
        for data, name, c in [(cpu, "CPU", "#d62728"), (gpu, "GPU", "#2ca02c")]:
            if not data:
                continue
            ns = [r["n"] for r in data["rows"]]
            ms = [r[key] for r in data["rows"]]
            ax[j].loglog(ns, ms, "o-", label=name, color=c)
        ax[j].set(title=title, xlabel="samples", ylabel="ms per call")
        ax[j].grid(alpha=0.3, which="both")
        ax[j].legend()

    if gpu and cpu:
        g_last, c_last = gpu["rows"][-1], cpu["rows"][-1]
        fft_x = c_last["fft_ms"] / g_last["fft_ms"]
        fir_x = c_last["fir_ms"] / g_last["fir_ms"]
        fig.suptitle(
            f"CuPy vs NumPy/SciPy   |   {gpu['backend']}   |   "
            f"at {g_last['n']:,} samples: FFT {fft_x:.0f}x, FIR {fir_x:.0f}x "
            f"(compute only; transfer erodes it)", fontsize=11)
    fig.tight_layout()
    png = OUT / "benchmark.png"
    fig.savefig(png, dpi=110)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
