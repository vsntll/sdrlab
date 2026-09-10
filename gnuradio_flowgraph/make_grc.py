#!/usr/bin/env python3
"""Generate fm_receiver.grc from GNU Radio's own GRC library.

Hand-written .grc files drift out of sync with block defaults across GNU Radio
versions; building the flow graph through ``gnuradio.grc.core`` instead lets GRC
fill in every parameter itself, so the result always opens cleanly in the
installed gnuradio-companion. Run with the radioconda interpreter:

    C:\\Users\\avasa\\radioconda\\python.exe gnuradio_flowgraph/make_grc.py

Then open ``gnuradio_flowgraph/fm_receiver.grc`` in gnuradio-companion, or
compile headless:  grcc gnuradio_flowgraph/fm_receiver.grc
"""
from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BLOCKS_PATH = pathlib.Path(sys.executable).parent / "Library/share/gnuradio/grc/blocks"
if BLOCKS_PATH.is_dir():
    os.environ.setdefault("GRC_BLOCKS_PATH", str(BLOCKS_PATH))

import gnuradio.grc as grc  # noqa: E402
from gnuradio.grc.core.platform import Platform  # noqa: E402


def _platform() -> Platform:
    p = Platform(version="3.10", version_parts=(3, 10, 12), prefs=None,
                 install_prefix=os.path.dirname(grc.__file__))
    p.build_library()
    return p


_counter: dict[str, int] = {}


def _block(fg, block_id, params, x, y):
    b = fg.new_block(block_id)
    if "id" not in params:
        n = _counter.get(block_id, 0)
        _counter[block_id] = n + 1
        params = {"id": f"{block_id}_{n}", **params}
    for k, v in params.items():
        b.params[k].set_value(str(v))
    b.states["coordinate"] = [x, y]
    return b


def build_fm_receiver(fg) -> None:
    o = fg.options_block
    o.params["id"].set_value("fm_receiver")
    o.params["title"].set_value("WBFM receiver (simulated capture)")
    o.params["generate_options"].set_value("qt_gui")

    cap, dev, off, ifr, arate = "960000", "75000", "-250000", "192000", "48000"
    _block(fg, "import", {"imports": "import math"}, 8, 8)
    _block(fg, "variable", {"id": "capture_rate", "value": cap}, 8, 80)
    _block(fg, "variable", {"id": "deviation", "value": dev}, 130, 80)
    _block(fg, "variable", {"id": "station_offset", "value": off}, 250, 80)
    _block(fg, "variable", {"id": "if_rate", "value": ifr}, 400, 80)
    _block(fg, "variable", {"id": "audio_rate", "value": arate}, 520, 80)

    t1 = _block(fg, "analog_sig_source_x", {"type": "float", "samp_rate": "capture_rate",
               "waveform": "analog.GR_COS_WAVE", "freq": "440", "amp": "1.0"}, 8, 180)
    t2 = _block(fg, "analog_sig_source_x", {"type": "float", "samp_rate": "capture_rate",
               "waveform": "analog.GR_COS_WAVE", "freq": "1200", "amp": "0.5"}, 8, 280)
    t3 = _block(fg, "analog_sig_source_x", {"type": "float", "samp_rate": "capture_rate",
               "waveform": "analog.GR_COS_WAVE", "freq": "3300", "amp": "0.3"}, 8, 380)
    trem = _block(fg, "analog_sig_source_x", {"type": "float", "samp_rate": "capture_rate",
               "waveform": "analog.GR_SIN_WAVE", "freq": "2", "amp": "0.2", "offset": "0.8"}, 8, 480)
    tones = _block(fg, "blocks_add_xx", {"type": "float", "num_inputs": "3"}, 200, 240)
    trem_mul = _block(fg, "blocks_multiply_xx", {"type": "float", "num_inputs": "2"}, 340, 280)
    msg = _block(fg, "blocks_multiply_const_vxx", {"type": "float", "const": "0.5"}, 480, 296)
    fm = _block(fg, "analog_frequency_modulator_fc",
                {"sensitivity": "2*math.pi*deviation/capture_rate"}, 620, 296)
    rot_up = _block(fg, "blocks_rotator_cc",
                    {"phase_inc": "2*math.pi*station_offset/capture_rate"}, 780, 296)
    noise = _block(fg, "analog_noise_source_x",
                   {"type": "complex", "noise_type": "analog.GR_GAUSSIAN", "amp": "0.02"}, 620, 420)
    chan_mix = _block(fg, "blocks_add_xx", {"type": "complex", "num_inputs": "2"}, 940, 320)
    thr = _block(fg, "blocks_throttle2",
                 {"type": "complex", "samples_per_second": "capture_rate"}, 1060, 320)

    rot_dn = _block(fg, "blocks_rotator_cc",
                    {"phase_inc": "-2*math.pi*station_offset/capture_rate"}, 1200, 320)
    chan = _block(fg, "low_pass_filter",
                  {"type": "fir_filter_ccf", "decim": "5", "interp": "1", "gain": "1",
                   "samp_rate": "capture_rate", "cutoff_freq": "100000", "width": "25000"}, 1340, 288)
    quad = _block(fg, "analog_quadrature_demod_cf",
                  {"gain": "if_rate/(2*math.pi*deviation)"}, 1560, 320)
    deemph = _block(fg, "analog_fm_deemph", {"samp_rate": "if_rate", "tau": "75e-6"}, 1720, 320)
    resamp = _block(fg, "rational_resampler_xxx",
                    {"type": "fff", "interp": "1", "decim": "4"}, 1860, 312)
    audio_lpf = _block(fg, "low_pass_filter",
                       {"type": "fir_filter_fff", "decim": "1", "interp": "1", "gain": "1.1",
                        "samp_rate": "audio_rate", "cutoff_freq": "15000", "width": "2000"}, 2000, 288)

    asink = _block(fg, "audio_sink", {"samp_rate": "int(audio_rate)", "num_inputs": "1"}, 2240, 240)
    rf_fft = _block(fg, "qtgui_freq_sink_x",
                    {"type": "complex", "fftsize": "4096", "bw": "capture_rate",
                     "name": '"captured RF spectrum"', "gui_hint": "0,0,1,1"}, 1200, 460)
    rf_wf = _block(fg, "qtgui_waterfall_sink_x",
                   {"type": "complex", "fftsize": "1024", "bw": "capture_rate",
                    "name": '"RF waterfall"', "gui_hint": "1,0,1,1"}, 1200, 580)
    au_fft = _block(fg, "qtgui_freq_sink_x",
                    {"type": "float", "fftsize": "2048", "bw": "audio_rate",
                     "name": '"recovered audio spectrum"', "gui_hint": "0,1,1,1"}, 2240, 460)

    fg.rewrite()

    def c(a, ao, b, bo="0"):
        fg.connect(a.get_source(str(ao)), b.get_sink(str(bo)))

    c(t1, 0, tones, 0); c(t2, 0, tones, 1); c(t3, 0, tones, 2)
    c(tones, 0, trem_mul, 0); c(trem, 0, trem_mul, 1)
    c(trem_mul, 0, msg); c(msg, 0, fm); c(fm, 0, rot_up)
    c(rot_up, 0, chan_mix, 0); c(noise, 0, chan_mix, 1)
    c(chan_mix, 0, thr); c(thr, 0, rot_dn); c(thr, 0, rf_fft); c(thr, 0, rf_wf)
    c(rot_dn, 0, chan); c(chan, 0, quad); c(quad, 0, deemph)
    c(deemph, 0, resamp); c(resamp, 0, audio_lpf)
    c(audio_lpf, 0, asink); c(audio_lpf, 0, au_fft)


def main() -> None:
    p = _platform()
    fg = p.make_flow_graph()
    build_fm_receiver(fg)
    out = HERE / "fm_receiver.grc"
    p.save_flow_graph(str(out), fg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
