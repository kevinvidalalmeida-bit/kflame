"""Import guard for independence audits, inherited by spawned Python workers.

Add this directory to PYTHONPATH to emulate Cantera being unavailable. This is
test infrastructure only; production does not install an import hook.
"""
import os
import sys


class _NoCantera:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cantera' or fullname.startswith('cantera.'):
            raise ImportError('Cantera blocked by the native independence audit')
        return None


sys.meta_path.insert(0, _NoCantera())


def _no_exported_transport(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).replace('\\', '/').split('/')[-1]
        if name == 'cantera_transport_poly_coeffs.json':
            raise RuntimeError('Exported Cantera transport fits blocked by the native audit')


sys.addaudithook(_no_exported_transport)
