#!/usr/bin/env python3
"""GNU Radio WBFM receiver flowgraph (simulated source, no hardware needed).

This is the "real GNU Radio" counterpart to the pure-CuPy pipeline in
``sdrlab/``. It needs a GNU Radio 3.10+ runtime, which on Windows means
installing radioconda (https://github.com/ryanvolz/radioconda) and running
this file with *that* interpreter:

    conda activate radioconda
    python gnuradio_flowgraph/fm_rx_sim.py --seconds 6 --out outputs/gr_fm.wav

Signal chain
------------
  tone bank -> FM modulator -> +offset -> +noise            (simulated capture)
  rotator (tune)  ->  low-pass FIR decimator  (channel select)
  quadrature demod (discriminator)
  FM de-emphasis (single-pole IIR, analog.fm_deemph)
  rational resampler -> audio band low-pass -> WAV sink

Swap the "SIMULATED CAPTURE" section for an ``osmosdr.source`` (gr-osmosdr /
SoapySDR) or a ``blocks.file_source`` to run it against a real RTL-SDR or a
recorded ``.cfile``.
"""
from __future__ import annotations

import argparse
import time

from gnuradio import analog, blocks, filter as gr_filter, gr
from gnuradio.filter import firdes

try:  # the window enums moved namespaces across GNU Radio versions
    from gnuradio.fft import window
except ImportError:  # pragma: no cover
    from gnuradio.filter import window  # type: ignore

TWO_PI = 2 * 3.141592653589793


class fm_rx_sim(gr.top_block):
    def __init__(self, seconds: float = 6.0, out_wav: str = "gr_fm.wav",
                 capture_rate: float = 960_000, station_offset: float = -250_000,
                 deviation: float = 75_000, audio_rate: int = 48_000):
        gr.top_block.__init__(self, "FM receiver (simulated)")

        channel_decim = 5                       # 960k -> 192k IF
        if_rate = capture_rate / channel_decim
        channel_bw = 200_000

        # ---------------- SIMULATED CAPTURE ---------------------------------
        # A small tone bank stands in for program audio, summed at capture rate.
        tones = [(440, 0.6), (1200, 0.35), (3300, 0.2)]
        adder = blocks.add_vff(1)
        for i, (f, a) in enumerate(tones):
            src = analog.sig_source_f(capture_rate, analog.GR_COS_WAVE, f, a, 0)
            self.connect(src, (adder, i))

        fm_mod = analog.frequency_modulator_fc(TWO_PI * deviation / capture_rate)
        # Shift the station off zero so the channel filter has work to do.
        offset = blocks.rotator_cc(TWO_PI * station_offset / capture_rate)
        noise = analog.noise_source_c(analog.GR_GAUSSIAN, 0.02, 0)
        mix = blocks.add_vcc(1)
        self.connect(adder, fm_mod, offset, (mix, 0))
        self.connect(noise, (mix, 1))

        # ---------------- RECEIVER ----------------------------------------
        # 1. Tune: undo the offset so the wanted station sits at DC.
        tuner = blocks.rotator_cc(-TWO_PI * station_offset / capture_rate)

        # 2. Channel-select FIR low-pass + decimation (960k -> 192k).
        chan_taps = firdes.low_pass(1.0, capture_rate, channel_bw / 2, channel_bw / 4,
                                    window.WIN_HAMMING)
        chan = gr_filter.fir_filter_ccf(channel_decim, chan_taps)

        # 3. Discriminator: instantaneous frequency -> message.
        quad = analog.quadrature_demod_cf(if_rate / (TWO_PI * deviation))

        # 4. De-emphasis: single-pole IIR (75 us).
        deemph = analog.fm_deemph(if_rate, 75e-6)

        # 5. Resample IF -> audio_rate and low-pass to the audio band.
        resamp = gr_filter.rational_resampler_fff(
            interpolation=int(audio_rate), decimation=int(if_rate))
        audio_taps = firdes.low_pass(1.0, audio_rate, 15_000, 2_000, window.WIN_HAMMING)
        audio_lpf = gr_filter.fir_filter_fff(1, audio_taps)

        wav = blocks.wavfile_sink(
            out_wav, 1, int(audio_rate), blocks.FORMAT_WAV, blocks.FORMAT_PCM_16)

        self.connect(mix, tuner, chan, quad, deemph, resamp, audio_lpf, wav)
        self._seconds = seconds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--out", default="gr_fm.wav")
    p.add_argument("--offset", type=float, default=-250_000)
    args = p.parse_args()

    tb = fm_rx_sim(seconds=args.seconds, out_wav=args.out, station_offset=args.offset)
    tb.start()
    time.sleep(args.seconds)
    tb.stop()
    tb.wait()
    print(f"wrote {args.out}  ({args.seconds:.0f}s of demodulated audio)")


if __name__ == "__main__":
    main()
