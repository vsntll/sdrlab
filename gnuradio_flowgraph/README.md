# GNU Radio version of the FM receiver + spectrum analyzer

The pipeline in `../sdrlab/` is written straight on NumPy/CuPy so it runs under
the project's Python 3.14 venv. **GNU Radio cannot be pip-installed** and does
not support Python 3.14, so these files need a separate GNU Radio runtime.

## Installed runtime on this machine

**radioconda 2025.03.14 → GNU Radio 3.10.12.0**, at `C:\Users\avasa\radioconda`
(Python 3.12). `cupy-cuda12x` 14.2 + the `nvidia-*-cu12` (CUDA 12.9) wheels are
installed into it too, so the GPU FFT block runs on the RTX 5060. Interpreter:

```
C:\Users\avasa\radioconda\python.exe
```

`gnuradio-companion` is on the Start menu (and at
`C:\Users\avasa\radioconda\Scripts\gnuradio-companion.exe`).

To reproduce on another machine: install
[radioconda](https://github.com/radioconda/radioconda) (silent:
`radioconda-*-Windows-x86_64.exe /S /D=%USERPROFILE%\radioconda`), then
`"%USERPROFILE%\radioconda\python.exe" -m pip install cupy-cuda12x nvidia-cufft-cu12
nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-curand-cu12 nvidia-cublas-cu12`.

## Run

```bat
set PY=C:\Users\avasa\radioconda\python.exe

%PY% fm_rx_sim.py --seconds 5 --out ..\outputs\gr_fm.wav   :: WBFM receiver -> WAV
%PY% spectrum_gpu.py                                        :: GPU-FFT spectrum analyzer
%PY% epy_cupy_psd.py                                        :: smoke-test the CuPy block
%PY% make_grc.py                                            :: regenerate fm_receiver.grc
```

Open **`fm_receiver.grc`** in gnuradio-companion for the live-GUI version (RF
spectrum + waterfall + recovered-audio spectrum + audio playback), or compile it
headless: `grcc fm_receiver.grc`.

Verified output:
* `fm_rx_sim.py` → `gr_fm.wav`, 5 s @ 48 kHz, the 440/1200/3300 Hz tones recovered
  (with the 2 Hz tremolo, matching `sdrlab`).
* `spectrum_gpu.py` → prints the 4 simulated carriers (−600/−120/+300/+850 kHz),
  FFT run on the RTX 5060 through `epy_cupy_psd.blk`.
* `fm_receiver.grc` → compiles with `grcc`, runs in gnuradio-companion.

`../scripts/selftest.py` (run with the project venv) exercises all of the above
plus the `sdrlab` pipelines in one go.

## Files

| file | what it is |
|---|---|
| `fm_rx_sim.py` | hand-written `gr.top_block`: simulated capture → tune → FIR channelize → quadrature demod → `analog.fm_deemph` (IIR) → resample → WAV |
| `spectrum_gpu.py` | multitone source → `epy_cupy_psd` block → frame-averaged PSD → peak list |
| `epy_cupy_psd.py` | **Embedded Python Block**: windowed \|FFT\|² → dB, computed on the GPU with CuPy (NumPy fallback), fftshift order |
| `fm_receiver.grc` | the FM receiver as a gnuradio-companion flowgraph with live QT GUI sinks (generated, `grcc`-validated) |
| `make_grc.py` | regenerates `fm_receiver.grc` via `gnuradio.grc.core` so it never drifts from the installed block defaults |

`fm_receiver.grc` is the GUI-editable version (open it in gnuradio-companion).
`fm_rx_sim.py` is the scripted equivalent — same chain, no GUI, WAV out:

```
  Signal Source (f) x3  --> Add ------.
  Signal Source (f, 2 Hz, +0.8) ------+--> Multiply --> x0.5 --> Frequency Mod
                                                                     |
                                            Rotator (+offset) <-------'
                                                 |
                Noise Source (c) --> Add <--------'
                                      |
                                    Head --> Throttle --> Rotator (-offset)
                                                                |
  Low Pass Filter (decim 5, cutoff 100 k)  <- FIR channel select
        v
  Quadrature Demod  (gain = if_rate / (2*pi*75k))
        v
  FM Deemphasis  (tau = 75 us)                <- one-pole IIR
        v
  Rational Resampler  (192k -> 48k)
        v
  Low Pass Filter (cutoff 15 k, gain 1.1)  -->  WAV File Sink
```

`spectrum_gpu.py` shows the other pattern: an **Embedded Python Block**
(`epy_cupy_psd.py`) fed a complex stream, producing a GPU-computed power
spectrum you can wire into a QT GUI Vector Sink.

## Mapping to `sdrlab`

| GNU Radio block | `sdrlab` equivalent |
|---|---|
| `analog.frequency_modulator_fc` | `sources.fm_modulate` |
| `blocks.rotator_cc` | `modulation.frequency_shift` |
| `filter.fir_filter_ccf` + `firdes.low_pass` | `filters.lowpass_decimate` |
| `analog.quadrature_demod_cf` | `modulation.quadrature_demod` |
| `analog.fm_deemph` | `filters.deemphasis_sos` + `filters.apply_iir` |
| `filter.rational_resampler_fff` | `cupyx.scipy.signal.resample_poly` |
| `epy_cupy_psd.blk` | `spectral.periodogram_psd` |
