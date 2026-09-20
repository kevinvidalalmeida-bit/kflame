"""KAVA: native free flames and FGM tables. Optional references load explicitly."""
import os as _os

# Configure before any scientific import; respect explicit user overrides.
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('NUMBA_NUM_THREADS', '4')
__version__ = '0.1.0'
