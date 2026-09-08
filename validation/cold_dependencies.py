"""Exact FD dependencies: no velocity chemistry, explicit thermal indexing."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import equations
import species_backend_native as backend_module


def precompute(x, problem, cache, rel_perturb, abs_perturb, eps, thermal, records):
    start = equations._profile_start(problem)
    n, k, nv = (int(cache[key]) for key in ('n_pts', 'n_sp', 'nv'))
    state = np.asarray(x).reshape(n, nv)
    delta = np.abs(state)*rel_perturb + abs_perturb
    delta[delta <= 0] = max(abs(rel_perturb), abs(eps), 1e-10)
    delta = np.where(state < 0, -delta, delta)
    # Only T and Y columns require a new composition-dependent evaluation.
    temperatures = np.repeat(state[:, 1], nv-1)
    composition = np.repeat(state[:, 2:].T, nv-1, axis=1)
    thermal_states = np.column_stack((state[:, 1], state[:, 1]+delta[:, 1])).ravel()
    thermal_index = np.repeat(2*np.arange(n), nv-1)
    for j in range(n):
        first = j*(nv-1)
        temperatures[first] = thermal_states[2*j+1]
        thermal_index[first] += 1
        for species in range(k):
            composition[species, first+1+species] += delta[j, 2+species]
    omega, enthalpy = np.empty_like(composition), np.empty_like(composition)
    backend = equations._jacobian_backend(problem)
    def evaluate(*args):
        return thermal.indexed(thermal_states, thermal_index, *args)
    with patch.object(backend_module, '_eval_thermo_kinetics_sparse_numba_core', evaluate):
        rho, cp = backend.eval_grid_thermo_kinetics_into(temperatures, composition, omega, enthalpy)
    out = (np.empty((n, nv)), np.empty((n, nv)),
           np.empty((k, n, nv)), np.empty((k, n, nv)))
    out[0][:, 0], out[1][:, 0] = cache['rho'], cache['cp_n']
    out[2][:, :, 0], out[3][:, :, 0] = cache['omega'], cache['hk_n']
    out[0][:, 1:], out[1][:, 1:] = rho.reshape(n, nv-1), cp.reshape(n, nv-1)
    out[2][:, :, 1:] = omega.reshape(k, n, nv-1)
    out[3][:, :, 1:] = enthalpy.reshape(k, n, nv-1)
    records.append(dict(kind='dependencies', nodes=n, thermal_states=2*n,
                        chemistry_states=n*(nv-1), skipped_velocity_states=n))
    equations._profile_record(problem, 'jacobian_precompute_thermochem', start)
    return out


@contextmanager
def dependency_context(thermal, records):
    def candidate(*args, **kwargs):
        try:
            return precompute(*args, **kwargs, thermal=thermal, records=records)
        except Exception as exc:
            records.append(dict(kind='dependencies', exception=repr(exc)))
            raise
    with patch.object(equations, '_precompute_block_tridiag_center_thermo', candidate):
        yield
