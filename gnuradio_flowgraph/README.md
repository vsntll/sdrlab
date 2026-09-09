# GNU Radio version of the FM receiver

The pipeline in `../sdrlab/` is written straight on NumPy/CuPy so it runs under
the same Python 3.14 venv as the rest of the project. **GNU Radio itself cannot
be pip-installed** and does not support Python 3.14 yet, so these two files need
a separate GNU Radio runtime.

## Getting a GNU Radio runtime (Windows)

Install **radioconda** — a self-contained conda distribution that bundles GNU
Radio 3.10, gr-osmosdr, SoapySDR and the RTL-SDR drivers:

* https://github.com/ryanvolz/radioconda/releases → `radioconda-Windows-x86_64.exe`

It brings its own Python (~3.11/3.12). After installing:

```bat
:: from the "radioconda prompt" (Start menu) or:
%USERPROFILE%\radioconda\Scripts\activate

python -c "from gnuradio import gr; print(gr.version())"
```

Optionally add CuPy into that environment for the GPU FFT block:

```bat
pip install cupy-cuda12x nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
```

## Run

```bat
python fm_rx_sim.py --seconds 6 --out ..\outputs\gr_fm.wav
python epy_cupy_psd.py        :: smoke-test the GPU FFT block
```

`fm_rx_sim.py` is a hand-written flowgraph, *not* a `.grc` file, so
`gnuradio-companion` will not open it directly. To edit it visually, recreate
the chain in GRC:

```
  Signal Source (f) ×3  ─▶ Add ─▶ Frequency Mod ─▶ Rotator (+offset) ─▶ Add ◀─ Noise Source (c)
                                                                          │
  Rotator (−offset) ◀───────────────────────────────────────────────────┘
        │
        ▼
  Low Pass Filter (decim 5, cutoff 100 k)     ← FIR channel select
        ▼
  Quadrature Demod  (gain = if_rate / (2π·75k))
        ▼
  FM Deemphasis  (τ = 75 µs)                   ← one-pole IIR
        ▼
  Rational Resampler  (192k → 48k)
        ▼
  Low Pass Filter (cutoff 15 k)  ─▶  WAV File Sink
```

Add an **Embedded Python Block** pointing at `epy_cupy_psd.py` and feed it the
post-channel-filter stream to get a GPU-computed spectrum you can wire into a
QT GUI Vector Sink.

## Mapping to `sdrlab`

| GNU Radio block | `sdrlab` equivalent |
|---|---|
| `analog.frequency_modulator_fc` | `sources.fm_modulate` |
| `blocks.rotator_cc` | `modulation.frequency_shift` |
| `filter.fir_filter_ccf` + `firdes.low_pass` | `filters.lowpass_decimate` |
| `analog.quadrature_demod_cf` | `modulation.quadrature_demod` |
| `analog.fm_deemph` | `filters.deemphasis_sos` + `filters.apply_iir` |
| `filter.rational_resampler_fff` | `cupyx.scipy.signal.resample_poly` |
| Embedded Python Block (`epy_cupy_psd`) | `spectral.periodogram_psd` |
