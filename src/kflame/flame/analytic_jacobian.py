"""Direct analytic 1-D block assembly with frozen transport coefficients.

The derivative follows the same fixed upwind/closure branches as the local
residual. Transport coefficients are frozen, including multicomponent/Soret;
their explicit dependence on gradients and composition is differentiated.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def face_partials(x, invW, z, coeff, multi, thermal, is_multi, molar):
    nodes, nv = x.shape
    ns = nv - 2
    flux = np.empty((nodes-1, ns))
    partials = np.zeros((nodes-1, 2, ns, nv))
    for f in prange(nodes-1):
        dz = z[f+1]-z[f]
        yl, yr = x[f, 2:], x[f+1, 2:]
        sl, sr = np.dot(yl, invW), np.dot(yr, invW)
        xl, xr = yl*invW/sl, yr*invW/sr
        raw = np.empty(ns)
        if is_multi:
            left, right = np.zeros(ns), np.zeros(ns)
            for k in range(ns):
                for v in range(ns):
                    left[k] += multi[k, v, f]*xl[v]
                    right[k] += multi[k, v, f]*xr[v]
            for k in range(ns):
                raw[k] = (right[k]-left[k])/dz
                for v in range(ns):
                    partials[f, 0, k, v+2] = -(multi[k, v, f]-left[k])*invW[v]/sl/dz
                    partials[f, 1, k, v+2] = (multi[k, v, f]-right[k])*invW[v]/sr/dz
        else:
            for k in range(ns):
                raw[k] = -coeff[k, f]*((xr[k]-xl[k]) if molar else (yr[k]-yl[k]))/dz
            total = np.sum(raw)
            for v in range(ns):
                suml, sumr = 0.0, 0.0
                for k in range(ns):
                    dl = invW[v]/sl*((1.0 if k == v else 0.0)-xl[k]) if molar else (1.0 if k == v else 0.0)
                    dr = invW[v]/sr*((1.0 if k == v else 0.0)-xr[k]) if molar else (1.0 if k == v else 0.0)
                    partials[f, 0, k, v+2] = coeff[k, f]*dl/dz
                    partials[f, 1, k, v+2] = -coeff[k, f]*dr/dz
                    suml += partials[f, 0, k, v+2]
                    sumr += partials[f, 1, k, v+2]
                for k in range(ns):
                    partials[f, 0, k, v+2] -= yl[k]*suml + (total if k == v else 0.0)
                    partials[f, 1, k, v+2] -= yl[k]*sumr
            raw -= yl*total
        for k in range(ns):
            flux[f, k] = raw[k]-thermal[k, f]*(x[f+1, 1]-x[f, 1])/dz
            partials[f, 0, k, 1] = thermal[k, f]/dz
            partials[f, 1, k, 1] = -thermal[k, f]/dz
    return flux, partials


@njit(cache=True, inline='always')
def _delta(node, col, var, target):
    return 1.0 if node == col and var == target else 0.0


@njit(cache=True, inline='always')
def _dflux(df, face, col, k, v):
    if col == face:
        return df[face, 0, k, v]
    if col == face+1:
        return df[face, 1, k, v]
    return 0.0


@njit(cache=True, parallel=True)
def assemble_blocks(x, z, rho, cp, omega, h, drho, dcp, domega, cp_molar,
                    flux, df, lam, invW, yin, energy, has_profile, fixed,
                    outlet_flux, threshold, active_columns):
    nodes, nv = x.shape
    ns = nv-2
    blocks = np.zeros((nodes, 3, nv, nv))
    for jj in prange(nodes):
        j = np.int64(jj)
        u = x[j, 0]
        close = np.argmax(x[j, 2:])
        for col in range(max(0, j-1), min(nodes, j+2)):
            if not active_columns[col]:
                continue
            side = col-j+1
            for v in range(nv):
                du = _delta(j, col, v, 0)
                dt = _delta(j, col, v, 1)
                rr = drho[j, v] if col == j else 0.0
                cc = dcp[j, v] if col == j else 0.0
                dm = rho[j]*du + u*rr
                if j == 0 or j == nodes-1:
                    other = 1 if j == 0 else j-1
                    dmo = (rho[other]*_delta(other, col, v, 0)
                           + x[other, 0]*(drho[other, v] if col == other else 0.0))
                    blocks[j, side, 0, v] = (dm-dmo)/(z[1]-z[0]) if j == 0 else dm-dmo
                    blocks[j, side, 1, v] = dt
                    if j == nodes-1 and (energy or not has_profile):
                        blocks[j, side, 1, v] -= _delta(other, col, v, 1)
                    for k in range(ns):
                        dy = _delta(j, col, v, k+2)
                        if k == close:
                            val = -1.0 if col == j and v >= 2 else 0.0
                        elif j == 0:
                            val = (-_dflux(df, 0, col, k, v)
                                   + dm*(yin[k]-x[j, k+2])-rho[j]*u*dy)
                        elif outlet_flux:
                            val = _dflux(df, j-1, col, k, v)+dm*x[j, k+2]+rho[j]*u*dy
                        else:
                            val = dy-_delta(other, col, v, k+2)
                        blocks[j, side, k+2, v] = val
                    if threshold > 0:
                        for row in range(nv):
                            if not (side == 1 and row == v) and abs(blocks[j, side, row, v]) <= threshold:
                                blocks[j, side, row, v] = 0.0
                    continue
                dzm, dzp, dz2 = z[j]-z[j-1], z[j+1]-z[j], z[j+1]-z[j-1]
                if fixed >= 0 and j == fixed:
                    blocks[j, side, 0, v] = dt if energy else dm
                else:
                    other = j-1 if fixed >= 0 and j > fixed else j+1
                    dmo = (rho[other]*_delta(other, col, v, 0)
                           + x[other, 0]*(drho[other, v] if col == other else 0.0))
                    blocks[j, side, 0, v] = -(dm-dmo)/dzm if other == j-1 else -(dmo-dm)/dzp
                up = j if u > 0.0 else j+1
                down = up-1
                dzup = z[up]-z[down]
                gradT = (x[up, 1]-x[down, 1])/dzup
                dgradT = (_delta(up, col, v, 1)-_delta(down, col, v, 1))/dzup
                cond = -2*(lam[j]*(x[j+1, 1]-x[j, 1])/dzp
                           -lam[j-1]*(x[j, 1]-x[j-1, 1])/dzm)/dz2
                dcond = -2*(lam[j]*(_delta(j+1, col, v, 1)-dt)/dzp
                            -lam[j-1]*(dt-_delta(j-1, col, v, 1))/dzm)/dz2
                en, den = 0.0, 0.0
                for k in range(ns):
                    dw = domega[j, k, v] if col == j else 0.0
                    fm, fp = flux[j-1, k], flux[j, k]
                    dfm = _dflux(df, j-1, col, k, v)
                    dfp = _dflux(df, j, col, k, v)
                    dh = cp_molar[k, j]*dt
                    gradh = (h[k, up]-h[k, down])/dzup
                    dgradh = (cp_molar[k, up]*_delta(up, col, v, 1)
                              -cp_molar[k, down]*_delta(down, col, v, 1))/dzup
                    en += (h[k, j]*omega[k, j]+.5*(fm+fp)*gradh)*invW[k]
                    den += (dh*omega[k, j]+h[k, j]*dw
                            +.5*(dfm+dfp)*gradh+.5*(fm+fp)*dgradh)*invW[k]
                    gradY = (x[up, k+2]-x[down, k+2])/dzup
                    dgradY = (_delta(up, col, v, k+2)-_delta(down, col, v, k+2))/dzup
                    diff = 2*(fp-fm)/dz2
                    ddiff = 2*(dfp-dfm)/dz2
                    blocks[j, side, k+2, v] = (-du*gradY-u*dgradY+(dw-ddiff)/rho[j]
                                             -(omega[k, j]-diff)*rr/rho[j]**2)
                if energy:
                    blocks[j, side, 1, v] = (-du*gradT-u*dgradT-(dcond+den)/(rho[j]*cp[j])
                        +(cond+en)/(rho[j]*cp[j])*(rr/rho[j]+cc/cp[j]))
                else:
                    blocks[j, side, 1, v] = dt
                if threshold > 0:
                    for row in range(nv):
                        if not (side == 1 and row == v) and abs(blocks[j, side, row, v]) <= threshold:
                            blocks[j, side, row, v] = 0.0
    return blocks


def build_analytic_blocks(x, problem, column_nodes=None):
    from kflame.flame.equations import (build_local_jacobian_cache, BlockTridiagJacobian,
                                       _profile_start, _profile_record, _outlet_species_flux_bc,
                                       _jacobian_backend)
    from kflame.chemistry.analytic import thermochemical_partials
    start = _profile_start(problem)
    cache = build_local_jacobian_cache(x, problem)
    b = _jacobian_backend(problem)
    nodes, ns = problem.n_points, problem.n_species
    state = np.ascontiguousarray(x.reshape(nodes, ns+2))
    tic = _profile_start(problem)
    active = np.ones(nodes, dtype=np.bool_)
    if column_nodes is None:
        drho, dcp, domega, cp_molar = thermochemical_partials(b, state[:, 1], state[:, 2:].T)
    else:
        active[:] = False
        active[column_nodes] = True
        selected = state[column_nodes]
        rr, cc, ww, hh = thermochemical_partials(b, selected[:, 1], selected[:, 2:].T)
        drho, dcp = np.zeros((nodes, ns+2)), np.zeros((nodes, ns+2))
        domega, cp_molar = np.zeros((nodes, ns, ns+2)), np.zeros((ns, nodes))
        drho[column_nodes], dcp[column_nodes], domega[column_nodes] = rr, cc, ww
        cp_molar[:, column_nodes] = hh
    _profile_record(problem, 'jacobian_analytic_thermochemistry', tic)
    tic = _profile_start(problem)
    multi = cache['multi_face_coeff']
    thermal = cache['soret_face_coeff']
    if multi is None:
        multi = np.zeros((1, 1, 1))
    if thermal is None:
        thermal = np.zeros((ns, nodes-1))
    flux, df = face_partials(state, b.invW, cache['z'], cache['face_coeff'], multi,
                            thermal, cache['flux_model'] == 'multicomponent', cache['basis_molar'])
    blocks = assemble_blocks(state, cache['z'], cache['rho'], cache['cp_n'], cache['omega'],
        cache['hk_n'], drho, dcp, domega, cp_molar, flux, df, cache['lam_face'], b.invW,
        cache['Y_in'], cache['solve_energy'], cache['has_T_prof'], cache['j_fixed'],
        _outlet_species_flux_bc(problem), float(getattr(problem, 'jacobian_threshold', 0.0)), active)
    result = BlockTridiagJacobian(np.ascontiguousarray(blocks[1:, 0]),
        np.ascontiguousarray(blocks[:, 1]), np.ascontiguousarray(blocks[:-1, 2]))
    result.use_compiled_substitution = bool(getattr(problem, 'use_compiled_block_substitution', True))
    _profile_record(problem, 'jacobian_analytic_spatial', tic)
    _profile_record(problem, 'jacobian_build', start)
    return result
