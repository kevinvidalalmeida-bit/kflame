"""First mechanism-specific codegen: exact sparse equilibrium/production sums.

Retain the production rate, falloff and signed mass-action expressions. Only
stoichiometric topology is emitted as constant-index operations in the same
summation order. No species or reaction is removed and no Jacobian is changed.
"""
from contextlib import contextmanager
import hashlib
import inspect
import textwrap
import time
from unittest.mock import patch
import numpy as np
from numba import njit
import species_backend_native as backend
from cold_strategy_kernels import replace_once

_KERNELS = {}


def generate(reference, args):
    source = textwrap.dedent(inspect.getsource(reference.py_func))
    source = source[source.index('def '):]
    names = list(inspect.signature(reference.py_func).parameters)
    values = dict(zip(names, args))
    counts, indices, nu = (values[k] for k in ('net_count', 'net_idx', 'net_nu'))
    nr, nk = len(counts), len(values['W'])
    delta = ['def generated_delta(g):', f'    out = np.empty({nr}, dtype=np.float64)']
    production = ['def generated_production(q, W, out, m):']
    for r in range(nr):
        delta.append(f'    out[{r}] = 0.0')
        for ii in range(int(counts[r])):
            k, coefficient = int(indices[r, ii]), float(nu[r, ii])
            if not 0 <= k < nk or not np.isfinite(coefficient):
                raise ValueError('Invalid stoichiometric topology')
            delta.append(f'    out[{r}] += {coefficient!r} * g[{k}]')
            production.append(f'    out[{k}, m] += {coefficient!r} * q[{r}] * W[{k}]')
    delta.append('    return out')
    production.append('    return')
    helpers = '\n'.join(delta)+'\n\n'+'\n'.join(production)+'\n'
    ns = dict(vars(backend))
    exec(compile(helpers, '<generated-stoichiometry>', 'exec'), ns)
    ns['generated_delta'] = njit(cache=False)(ns['generated_delta'])
    ns['generated_production'] = njit(cache=False)(ns['generated_production'])
    source = replace_once(source, '        for r in range(n_rxn):',
        '        delta_all = generated_delta(g_RT)\n'
        '        rates = np.empty(n_rxn, dtype=np.float64)\n'
        '        for r in range(n_rxn):')
    source = replace_once(source,
        '            delta_g = 0.0\n'
        '            for ii in range(net_count[r]):\n'
        '                k = net_idx[r, ii]\n'
        '                delta_g += net_nu[r, ii] * g_RT[k]',
        '            delta_g = delta_all[r]')
    source = replace_once(source,
        '            for ii in range(net_count[r]):\n'
        '                k = net_idx[r, ii]\n'
        '                omega_out[k, m] += net_nu[r, ii] * q * W[k]',
        '            rates[r] = q\n'
        '        generated_production(rates, W, omega_out, m)')
    exec(compile(source, '<generated-chemistry-core>', 'exec'), ns)
    kernel = njit(parallel=True, cache=False)(ns[reference.py_func.__name__])
    return kernel, helpers+'\n'+source


class GeneratedChemistry:
    def __init__(self, reference, records):
        self.reference, self.records, self.kernels = reference, records, _KERNELS
        self.reference_hash = hashlib.sha256(inspect.getsource(reference.py_func).encode()).hexdigest()
        names = list(inspect.signature(reference.py_func).parameters)
        self.positions = [names.index(k) for k in ('net_count', 'net_idx', 'net_nu')]

    def __call__(self, *args):
        # Content key protects against another mechanism with the same size or
        # an in-place metadata change. Its cost is INCLUDED in every timed call.
        h = hashlib.sha256()
        for i in self.positions:
            a = args[i]
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
        key = (len(args[6]), h.hexdigest(), self.reference_hash)
        if key not in self.kernels:
            started = time.perf_counter()
            candidate, source = generate(self.reference, args)
            candidate(*args)
            self.kernels[key] = candidate
            self.records.append(dict(kind='generated_compile', topology_sha256=key[1],
                source_sha256=hashlib.sha256(source.encode()).hexdigest(), source=source,
                generation_and_first_call_s=time.perf_counter()-started))
        else:
            self.kernels[key](*args)
        self.records.append(dict(kind='generated_chemistry', nodes=int(args[0].size),
                                 topology_sha256=key[1]))


@contextmanager
def generated_chemistry_context(records):
    candidate = GeneratedChemistry(backend._eval_thermo_kinetics_sparse_numba_core, records)
    with patch.object(backend, '_eval_thermo_kinetics_sparse_numba_core', candidate):
        yield candidate
