"""sdrlab - a tiny GPU-accelerated SDR DSP demo (CuPy with NumPy fallback)."""
from . import backend, filters, modulation, sources, spectral

__all__ = ["backend", "sources", "spectral", "filters", "modulation"]
__version__ = "0.1.0"
