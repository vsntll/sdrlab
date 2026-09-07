"""Compute backend selection: CuPy (GPU) with a transparent NumPy fallback.

Everything else in ``sdrlab`` imports ``xp`` from here instead of importing
``numpy``/``cupy`` directly, so the same DSP code runs on either device.

Set ``SDRLAB_FORCE_CPU=1`` to skip the GPU even when CuPy is installed.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
import warnings

# --- Work around a broken PATH entry before importing cupy -------------------
# CuPy's Windows startup calls os.add_dll_directory() on a directory it derives
# from the first CUDA runtime DLL it finds on PATH. Some vendor tools (HP OMEN
# Broadcast, NVIDIA Broadcast) ship an old nvrtc and leave a now-missing folder
# on PATH, which makes that call raise FileNotFoundError. Drop those entries so
# CuPy finds the CUDA 12 libraries from the nvidia-*-cu12 wheels instead.
_BAD_PATH_MARKERS = ("OMEN", "NVIDIA BROADCAST", "NVIDIA CORPORATION\\NVIDIA BROADCAST")
_kept = []
for _p in os.environ.get("PATH", "").split(os.pathsep):
    _u = _p.upper()
    if any(m in _u for m in _BAD_PATH_MARKERS):
        continue
    _kept.append(_p)
os.environ["PATH"] = os.pathsep.join(_kept)
warnings.filterwarnings("ignore", message="CUDA path could not be detected")

# --- Select the array module ----------------------------------------------------
_FORCE_CPU = os.environ.get("SDRLAB_FORCE_CPU", "").lower() not in ("", "0", "false", "no")

USING_GPU = False
GPU_NAME = ""
_gpu_import_error: Exception | None = None

if not _FORCE_CPU:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as _cp
        if _cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device visible")
        # Force context creation now so failures surface here, not mid-pipeline.
        _cp.zeros(1, dtype=_cp.complex64) + 0
        xp = _cp
        USING_GPU = True
        GPU_NAME = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception as exc:  # pragma: no cover - depends on host
        _gpu_import_error = exc

if not USING_GPU:
    import numpy as xp  # type: ignore  # noqa: F401


def asnumpy(a):
    """Return a NumPy copy of ``a`` whether it lives on the host or the GPU."""
    if USING_GPU and isinstance(a, xp.ndarray):
        return xp.asnumpy(a)
    import numpy as _np

    return _np.asarray(a)


def to_device(a):
    """Move a host array onto the active backend (no-op on the NumPy backend)."""
    return xp.asarray(a)


def synchronize() -> None:
    """Block until queued GPU work is done (no-op on CPU). Needed for timing."""
    if USING_GPU:
        xp.cuda.Stream.null.synchronize()


def signal():
    """Return the active ``scipy.signal``-compatible module."""
    if USING_GPU:
        import cupyx.scipy.signal as _s
    else:
        import scipy.signal as _s
    return _s


def fft():
    """Return the active ``scipy.fft``-compatible module."""
    if USING_GPU:
        import cupyx.scipy.fft as _f
    else:
        import scipy.fft as _f
    return _f


@contextlib.contextmanager
def timed(label: str, out: dict | None = None):
    """Context manager that reports wall time for a DSP stage (GPU-synced)."""
    synchronize()
    t0 = time.perf_counter()
    yield
    synchronize()
    dt = time.perf_counter() - t0
    if out is not None:
        out[label] = dt
    print(f"  [{'GPU' if USING_GPU else 'CPU'}] {label:<28} {dt * 1e3:8.2f} ms")


def warmup() -> None:
    """Run each kernel family once so later timings exclude JIT / plan building."""
    if not USING_GPU:
        return
    import numpy as _np

    x = xp.asarray((_np.random.randn(4096) + 1j * _np.random.randn(4096)).astype(_np.complex64))
    fft().fft(x)
    S = signal()
    S.fftconvolve(x, xp.asarray(_np.hanning(65)))
    S.sosfilt(xp.asarray(S.butter(4, 0.2, output="sos")), x.real)
    S.welch(x, nperseg=512)
    S.resample_poly(x.real, 4, 25)
    S.spectrogram(x, nperseg=256)
    synchronize()


def describe() -> str:
    if USING_GPU:
        return f"CuPy {xp.__version__} on {GPU_NAME} (compute {xp.cuda.Device(0).compute_capability})"
    reason = f" ({_gpu_import_error})" if _gpu_import_error else ""
    return f"NumPy {xp.__version__} on CPU{reason}"


if __name__ == "__main__":
    print(describe(), file=sys.stderr)
