# sdrlab — a small GPU-accelerated SDR signal-processing demo

A compact software-defined-radio DSP demo in Python: **FFT spectral analysis**,
**FIR/IIR filtering**, and a **full FM modulation/demodulation chain**, with the
FFT and FIR stages **accelerated on the GPU with CuPy** (transparent NumPy
fallback). No RTL-SDR hardware required — the captures are synthesized — but the
same code reads real dongle captures.

```
sdrlab/            reusable DSP library
  backend.py       CuPy/NumPy dispatch (`xp`), timing, GPU warm-up
  sources.py       IQ sources: synthetic captures, .iq/.cfile readers, RTL-SDR
  spectral.py      periodogram PSD, Welch PSD, spectrogram  (FFT on the backend)
  filters.py       FIR (window + Parks–McClellan) & IIR (butter/cheby/ellip) design + apply
  modulation.py    FM/AM modulate, quadrature demod, de-emphasis, full fm_receiver()
scripts/
  demo_spectrum.py spectrum analyzer: periodogram + Welch + waterfall  -> outputs/spectrum.png
  demo_fm.py       WBFM receiver: synth capture -> audio  -> outputs/fm_audio.wav + fm_chain.png
  demo_filters.py  FIR vs IIR design/response/effect showcase           -> outputs/filters.png
  benchmark.py     CuPy vs NumPy/SciPy for FFT + FIR across sizes        -> outputs/benchmark.png
gnuradio_flowgraph/
  fm_rx_sim.py     the same FM receiver as a real GNU Radio flowgraph (needs radioconda)
  epy_cupy_psd.py  GNU Radio Embedded Python Block: FFT power spectrum on the GPU
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

CPU-only machine? Install just `numpy scipy matplotlib` and everything still
runs (or set `SDRLAB_FORCE_CPU=1` to force it even when CuPy is present).

This project was set up and tested with:
CuPy 14.2 / CUDA 12.9 wheels on an **RTX 5060 Laptop (Blackwell, sm_120)**,
Python 3.14, Windows 11.

## Run

```bash
python scripts/demo_spectrum.py
python scripts/demo_fm.py
python scripts/demo_filters.py
python scripts/benchmark.py
```

Each writes a PNG (and `demo_fm` a WAV) into `outputs/` and prints per-stage
timings. Add `SDRLAB_FORCE_CPU=1` in front of any of them for a CPU reference.

### Using a real RTL-SDR / recorded IQ

```bash
# record with the rtl-sdr tools, then:
python scripts/demo_fm.py       --iq capture.bin --fmt cu8 --fs 2.4e6 --offset -300e3
python scripts/demo_spectrum.py --iq capture.bin --fmt cu8 --fs 2.4e6 --fc 100.1e6
```
`--fmt` is `cu8` (rtl_sdr raw), `ci16`, or `cf32` (GNU Radio complex float).
Live capture via `sdrlab.sources.read_rtl_sdr()` needs `pip install pyrtlsdr`
plus `librtlsdr` and a dongle.

## What each piece demonstrates

| Requirement | Where |
|---|---|
| Simulated signal source (no hardware) | `sources.fm_broadcast_capture`, `multitone_capture` |
| FFT-based spectral analysis | `spectral.periodogram_psd` / `welch_psd` / `spectrogram_db` |
| FIR filtering | `filters.fir_lowpass` (windowed-sinc), `fir_lowpass_remez` (equiripple) |
| IIR filtering | `filters.iir_lowpass_sos`, `filters.deemphasis_sos` (one-pole) |
| Modulation | `sources.fm_modulate`, `am_modulate` |
| Demodulation chain | `modulation.fm_receiver`: tune → FIR channelize → discriminator → IIR de-emphasis → resample |
| CuPy acceleration | `backend.py` selects `xp`; FFT & FIR run via `cupyx.scipy` on the GPU |

## Notes on the GPU acceleration

* **FFT and FIR (FFT convolution) parallelise well** — ~50–60× over SciPy at a
  few million samples on this laptop GPU (see `benchmark.png`).
* **Host↔device transfer is the real cost.** Moving 8M complex64 each way (~30 ms)
  dwarfs the 1.8 ms transform. `sdrlab` keeps the signal resident on the GPU for
  the whole pipeline and only copies the small final audio / plot arrays back.
* **Recursive IIR (`sosfilt`) barely speeds up** — each output sample depends on
  the previous one, so there is little to parallelise. `demo_filters.py` shows
  this directly. De-emphasis is cheap enough that it does not matter here.

## The GNU Radio version

`gnuradio_flowgraph/` contains the same FM receiver built from real GNU Radio
blocks, plus a CuPy-backed FFT block. It needs a separate GNU Radio 3.10+
runtime — see `gnuradio_flowgraph/README.md`.
