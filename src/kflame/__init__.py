"""KFLAME: native free flames and FGM tables. Optional references load explicitly."""
import os as _os

# Configure before any scientific import; respect explicit user overrides.
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('NUMBA_NUM_THREADS', '4')
__version__ = '0.1.0'


def __getattr__(name):
    # Keep CLI/help imports light; scientific modules load on first API access.
    if name in ('solve_flame', 'generate_fgm'):
        from . import api
        return getattr(api, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['solve_flame', 'generate_fgm']
