"""One command to check every pipeline still works.

    python scripts/selftest.py            # sdrlab (this venv) + GNU Radio if found
    python scripts/selftest.py --quick    # sdrlab only, shorter signals

Runs each demo/flowgraph headless and asserts the output is sane (files created,
tones recovered, FM audio SNR above threshold). Exit code 0 = all passed.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT))

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            msg = fn() or "ok"
            _results.append((name, True, msg))
        except Exception as exc:  # noqa: BLE001
            _results.append((name, False, f"{type(exc).__name__}: {exc}"))
    return deco


def _tone_levels(path, tones=(440, 1200, 3300)):
    w = wave.open(str(path))
    fs = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    freq = np.fft.rfftfreq(len(a), 1 / fs)
    spec /= spec.max()
    return {t: 20 * np.log10(spec[max(0, np.argmin(abs(freq - t)) - 3):
                                  np.argmin(abs(freq - t)) + 4].max() + 1e-9) for t in tones}, a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dur = "2" if args.quick else "4"

    from sdrlab import backend, filters, modulation, sources, spectral

    @check("backend")
    def _():
        return backend.describe()

    @check("spectral: 4 carriers found")
    def _():
        iq, fs = sources.multitone_capture(duration=0.3)
        f, psd = spectral.periodogram_psd(iq, fs, nfft=1 << 17)
        peaks = spectral.peak_frequencies(f, psd)
        got = sorted(round(p[0] / 1e3) for p in peaks)
        assert got == [-600, -120, 300, 850], got
        return f"{got} kHz"

    @check("filters: FIR stopband < -40 dB, IIR runs")
    def _():
        taps = filters.fir_lowpass(2.4e6, 200e3, numtaps=201)
        w, h = filters.fir_response(taps, 2.4e6)
        stop = 20 * np.log10(np.abs(h[w > 400e3]).max())
        assert stop < -40, stop
        iq, fs = sources.multitone_capture(duration=0.1)
        filters.apply_iir(iq, filters.iir_lowpass_sos(fs, 200e3, 8))
        return f"FIR stopband {stop:.0f} dB"

    @check("modulation: FM audio SNR > 12 dB")
    def _():
        iq, fs, meta = sources.fm_broadcast_capture(duration=float(dur))
        audio, afs = modulation.fm_receiver(iq, fs, meta["f_offset"], meta["deviation"])
        snr = modulation.audio_quality(audio, meta["message"], afs, meta["message_fs"])
        assert snr > 12, snr
        modulation.write_wav(str(OUT / "selftest_fm.wav"), audio, afs)
        return f"SNR {snr:.1f} dB, audio_fs {afs:.0f}"

    @check("scripts/demo_fm.py runs and writes WAV")
    def _():
        r = subprocess.run([sys.executable, str(ROOT / "scripts/demo_fm.py"),
                            "--duration", dur], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-500:]
        lv, _a = _tone_levels(OUT / "fm_audio.wav")
        assert lv[440] > -3 and lv[1200] > -12, lv
        return "tones " + ", ".join(f"{k}:{v:.0f}dB" for k, v in lv.items())

    @check("scripts/demo_spectrum.py + demo_filters.py + benchmark.py")
    def _():
        for scr, extra in [("demo_spectrum.py", []), ("demo_filters.py", []),
                           ("benchmark.py", ["--sizes", "12", "18", "--reps", "5"])]:
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / scr), *extra],
                               capture_output=True, text=True, cwd=ROOT)
            assert r.returncode == 0, f"{scr}: {r.stderr[-400:]}"
        return "3 scripts ok, PNGs in outputs/"

    # ---- GNU Radio side (separate interpreter) ------------------------------
    rc = os.environ.get("RADIOCONDA_PYTHON") or r"C:\Users\avasa\radioconda\python.exe"
    if pathlib.Path(rc).exists() or shutil.which(rc):
        @check("GNU Radio: fm_rx_sim.py recovers tones")
        def _():
            r = subprocess.run([rc, str(ROOT / "gnuradio_flowgraph/fm_rx_sim.py"),
                                "--seconds", dur, "--out", str(OUT / "gr_fm.wav")],
                               capture_output=True, text=True, cwd=ROOT)
            assert r.returncode == 0, r.stderr[-500:]
            lv, a = _tone_levels(OUT / "gr_fm.wav")
            assert lv[440] > -3, lv
            assert np.abs(a).max() < 0.999, "clipping"
            return "tones " + ", ".join(f"{k}:{v:.0f}dB" for k, v in lv.items())

        @check("GNU Radio: spectrum_gpu.py finds 4 carriers")
        def _():
            r = subprocess.run([rc, str(ROOT / "gnuradio_flowgraph/spectrum_gpu.py")],
                               capture_output=True, text=True, cwd=ROOT)
            assert r.returncode == 0, r.stderr[-500:]
            for khz in ("-600.0", "-120.1", "+300.0", "+850.2"):
                assert khz in r.stdout, r.stdout
            gpu = "GPU (CuPy)" in r.stdout
            return f"4 carriers, backend={'GPU' if gpu else 'CPU'}"

        @check("GNU Radio: fm_receiver.grc compiles (grcc)")
        def _():
            grcc = pathlib.Path(rc).parent / "Scripts/grcc.exe"
            grc = ROOT / "gnuradio_flowgraph/fm_receiver.grc"
            if not grcc.exists() or not grc.exists():
                return "skipped (grcc or .grc missing)"
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                r = subprocess.run([str(grcc), "-o", td, str(grc)],
                                   capture_output=True, text=True)
                assert "Compilation error" not in (r.stdout + r.stderr), r.stdout[-500:]
                assert (pathlib.Path(td) / "fm_receiver.py").exists()
            return "grcc ok"
    else:
        _results.append(("GNU Radio", True, "skipped (radioconda not found)"))

    print()
    width = max(len(n) for n, _, _ in _results)
    for name, ok, msg in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {msg}")
    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
