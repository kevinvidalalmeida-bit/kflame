import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from state import C_T, C_U, C_Y, build_transient_mask

def _corrected_flux_jax(Y_L, Y_R, rho_f, D_f, dz, W, W_mix_f, basis: str):
    if basis in ("molar", "mole"):
        W_mix_L = 1.0 / jnp.sum(Y_L / W[:, None], axis=0)
        W_mix_R = 1.0 / jnp.sum(Y_R / W[:, None], axis=0)
        X_L = Y_L * (W_mix_L[None, :] / W[:, None])
        X_R = Y_R * (W_mix_R[None, :] / W[:, None])
        dphi = (X_R - X_L) / dz[None, :]
        J_star = -rho_f[None, :] * (W[:, None] / W_mix_f[None, :]) * D_f * dphi
    else:
        dphi = (Y_R - Y_L) / dz[None, :]
        J_star = -rho_f[None, :] * D_f * dphi
    return J_star - Y_L * jnp.sum(J_star, axis=0, keepdims=True)

def _residual_jax_pure(
    x, z, T_in, Y_in, W, invW, P, j_fixed, T_fixed_point,
    T_prof_dev, solve_energy, basis, rdt, mask, x_old_dev, x_older_dev, use_bdf2,
    eval_grid_thermo_kinetics_jax, eval_faces_jax
):
    n_pts = z.shape[0]
    n_sp = W.shape[0]
    nv = 2 + n_sp

    x_r = x.reshape(n_pts, nv)
    u = x_r[:, C_U]
    T = x_r[:, C_T]
    Y = x_r[:, C_Y:].T

    rho, cp_n, omega, hk_n = eval_grid_thermo_kinetics_jax(T, Y)

    T_face = 0.5 * (T[:-1] + T[1:])
    Y_face = 0.5 * (Y[:, :-1] + Y[:, 1:])
    rho_face, D_face, lam_face, W_mix_face, sumd_face = eval_faces_jax(T_face, Y_face)
    dz_face = z[1:] - z[:-1]
    
    flux = _corrected_flux_jax(Y[:, :-1], Y[:, 1:], rho_face, D_face, dz_face, W, W_mix_face, basis)

    F = jnp.zeros((n_pts, nv))

    # Left boundary (idx 0)
    dz0 = z[1] - z[0]
    F = F.at[0, C_U].set(-(rho[1] * u[1] - rho[0] * u[0]) / dz0)
    if solve_energy:
        F = F.at[0, C_T].set(T[0] - T_in)
    else:
        F = F.at[0, C_T].set(T[0] - (T_prof_dev[0] if T_prof_dev is not None else T_in))

    mdot_in = rho[0] * u[0]
    left_species = (-(flux[:, 0] + mdot_in * Y[:, 0]) + mdot_in * Y_in)
    k_idx = jnp.arange(n_sp)
    k_exc = jnp.argmax(Y[:, 0])
    
    F_Y0 = jnp.where(k_idx == k_exc, 1.0 - jnp.sum(Y[:, 0]), left_species)
    F = F.at[0, C_Y:C_Y + n_sp].set(F_Y0)

    # Interior (1:-1)
    if n_pts > 2:
        j = jnp.arange(1, n_pts - 1)
        dzm = z[1:-1] - z[:-2]
        dzp = z[2:] - z[1:-1]
        dz2 = z[2:] - z[:-2]
        fm = flux[:, :-1]
        fp = flux[:, 1:]

        cont_forward = -(rho[2:] * u[2:] - rho[1:-1] * u[1:-1]) / dzp
        if j_fixed is None:
            F = F.at[1:-1, C_U].set(cont_forward)
        else:
            cont_backward = -(rho[1:-1] * u[1:-1] - rho[:-2] * u[:-2]) / dzm
            if solve_energy:
                cont_anchor = T[1:-1] - T_fixed_point
            else:
                cont_anchor = rho[1:-1] * u[1:-1] - rho[0] * 0.3
            
            F_U_int = jnp.where(
                j == j_fixed,
                cont_anchor,
                jnp.where(j > j_fixed, cont_backward, cont_forward),
            )
            F = F.at[1:-1, C_U].set(F_U_int)

        rho_j = rho[1:-1]
        rho_u_j = rho_j * u[1:-1]
        jloc = jnp.where(u[1:-1] > 0.0, j, j + 1)
        jloc_m = jloc - 1
        dz_up = z[jloc] - z[jloc_m]

        if solve_energy:
            dTdz = (T[jloc] - T[jloc_m]) / dz_up
            cond = -2.0 * (
                lam_face[1:] * (T[2:] - T[1:-1]) / dzp
                - lam_face[:-1] * (T[1:-1] - T[:-2]) / dzm
            ) / dz2
            dhk_dz = (hk_n[:, jloc] - hk_n[:, jloc_m]) / dz_up[None, :]
            flx = 0.5 * (fm + fp)
            en_sum = (
                jnp.sum(hk_n[:, 1:-1] * omega[:, 1:-1] * invW[:, None], axis=0)
                + jnp.sum(flx * dhk_dz * invW[:, None], axis=0)
            )
            F_T_int = (-cp_n[1:-1] * rho_u_j * dTdz - cond - en_sum) / (rho_j * cp_n[1:-1])
            F = F.at[1:-1, C_T].set(F_T_int)
        else:
            if T_prof_dev is not None:
                F = F.at[1:-1, C_T].set(T[1:-1] - T_prof_dev[1:-1])
            else:
                F = F.at[1:-1, C_T].set(T[1:-1] - T_in)

        dYdz = (Y[:, jloc] - Y[:, jloc_m]) / dz_up[None, :]
        conv = rho_u_j[None, :] * dYdz
        diff = 2.0 * (fp - fm) / dz2[None, :]
        F_Y_int = ((omega[:, 1:-1] - conv - diff) / rho_j[None, :]).T
        F = F.at[1:-1, C_Y:C_Y + n_sp].set(F_Y_int)

    # Right boundary (-1)
    F = F.at[-1, C_U].set(rho[-1] * u[-1] - rho[-2] * u[-2])
    if solve_energy:
        F = F.at[-1, C_T].set(T[-1] - T[-2])
    else:
        if T_prof_dev is not None:
            F = F.at[-1, C_T].set(T[-1] - T_prof_dev[-1])
        else:
            F = F.at[-1, C_T].set(T[-1] - T[-2])
            
    right_species = Y[:, -1] - Y[:, -2]
    k_exc_R = jnp.argmax(Y[:, -1])
    F_Y_R = jnp.where(k_idx == k_exc_R, 1.0 - jnp.sum(Y[:, -1]), right_species)
    F = F.at[-1, C_Y:C_Y + n_sp].set(F_Y_R)

    F_flat = F.ravel()
    
    # Transient
    if x_old_dev is not None:
        if use_bdf2 and x_older_dev is not None:
            F_flat -= mask * (1.5 * rdt * x - 2.0 * rdt * x_old_dev + 0.5 * rdt * x_older_dev)
        else:
            F_flat -= mask * rdt * (x - x_old_dev)

    return F_flat

class JaxEvaluator:
    def __init__(self, problem):
        self.problem = problem
        mech = problem.backend.mech
        self.n_sp = mech.n_species
        
        # Load parameters onto device
        self.W = jnp.array(mech.molecular_weights)
        self.invW = jnp.array(mech.inv_molecular_weights)
        self.P = float(problem.P)
        self.basis = str(getattr(problem, "flux_gradient_basis", "molar")).lower()
        
        # Thermo/Kinetics closures
        from materiales.kinetics_jax import vmap_net_production_rates_jax
        from materiales.thermo_native import NativeThermo
        
        thermo = NativeThermo(mech, xp=jnp)
        
        A_hi = jnp.array([r.A for r in mech.reactions])
        b_hi = jnp.array([r.b for r in mech.reactions])
        Ea_hi = jnp.array([r.Ea for r in mech.reactions])

        A_lo = jnp.array([r.A_low for r in mech.reactions])
        b_lo = jnp.array([r.b_low for r in mech.reactions])
        Ea_lo = jnp.array([r.Ea_low for r in mech.reactions])

        is_three_body = jnp.array([r.rtype == "three-body" for r in mech.reactions])
        is_falloff = jnp.array([r.rtype == "falloff" for r in mech.reactions])
        is_reversible = jnp.array([r.reversible for r in mech.reactions])
        has_troe = jnp.array([r.has_troe for r in mech.reactions])

        troe_A = jnp.array([r.troe_A for r in mech.reactions])
        troe_T3 = jnp.array([r.troe_T3 for r in mech.reactions])
        troe_T1 = jnp.array([r.troe_T1 for r in mech.reactions])
        troe_T2 = jnp.array([r.troe_T2 for r in mech.reactions])

        delta_nu = jnp.sum(mech.nu_net, axis=0)
        nu_r = jnp.array(mech.nu_reactants)
        nu_p = jnp.array(mech.nu_products)
        nu_net = jnp.array(mech.nu_net)
        eff = jnp.array(mech.efficiencies)
        
        def eval_grid_thermo_kinetics_jax(T, Y):
            cp_mass, hk, g_RT = thermo.cp_R(T) * 8314.46261815324, thermo.h_RT(T) * 8314.46261815324 * T[None, :], thermo.g_RT(T)
            # Recompute proper vectors (simplification, real backend does this properly)
            cp_mass = jnp.sum(Y * cp_mass * self.invW[:, None], axis=0)
            inv_wmix = jnp.sum(Y * self.invW[:, None], axis=0)
            rho = self.P / (inv_wmix * 8314.46261815324 * T)
            
            C = rho[None, :] * Y * self.invW[:, None]
            
            wdot = vmap_net_production_rates_jax(
                T, C.T, g_RT.T, A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
                is_three_body, is_falloff, is_reversible, has_troe,
                troe_A, troe_T3, troe_T1, troe_T2,
                nu_r, nu_p, nu_net, eff, delta_nu
            ).T
            omega_mass = wdot * self.W[:, None]
            
            return rho, cp_mass, omega_mass, hk

        from materiales.transport_native import NativeTransport
        from materiales.transport_jax import vmap_eval_faces_poly_jax
        
        trans = NativeTransport(mech, xp=jnp)
        cond_poly = trans._cond_poly
        diff_poly = trans._diff_poly
        
        def eval_faces_jax(T_face, Y_face):
            rho, Dm, lam, wm, sumd = vmap_eval_faces_poly_jax(T_face, Y_face.T, self.P, self.invW, self.W, cond_poly, diff_poly)
            return rho, Dm.T, lam, wm, sumd.T

        self.eval_grid = eval_grid_thermo_kinetics_jax
        self.eval_faces = eval_faces_jax
        
        # Build core compiled residual function closure
        def compiled_residual(x, z, T_in, Y_in, j_fixed, T_fixed_point, T_prof_dev, solve_energy, rdt, mask, x_old_dev, x_older_dev, use_bdf2):
            return _residual_jax_pure(
                x, z, T_in, Y_in, self.W, self.invW, self.P, j_fixed, T_fixed_point,
                T_prof_dev, solve_energy, self.basis, rdt, mask, x_old_dev, x_older_dev, use_bdf2,
                self.eval_grid, self.eval_faces
            )
            
        self.compiled_residual = jax.jit(compiled_residual, static_argnames=["solve_energy", "use_bdf2"])
        self.compiled_jacobian = jax.jit(jax.jacfwd(compiled_residual, argnums=0), static_argnames=["solve_energy", "use_bdf2"])
        
    def evaluate_residual(self, x, problem, rdt=0.0, x_old=None, x_older=None, transient_order=1):
        import numpy as np
        z = jnp.asarray(problem.z)
        T_in = float(problem.T_in)
        Y_in = jnp.asarray(problem.Y_in)
        j_fixed = int(problem.j_fixed) if problem.j_fixed is not None else -1
        T_fixed_point = float(problem.T_fixed_point) if hasattr(problem, 'T_fixed_point') else 0.0
        T_prof_dev = jnp.asarray(problem.T_profile_fixed) if getattr(problem, 'T_profile_fixed', None) is not None else None
        solve_energy = bool(problem.solve_energy)
        mask = jnp.asarray(build_transient_mask(z.shape[0], self.n_sp, solve_energy=solve_energy)) if rdt > 0 else None
        x_old_dev = jnp.asarray(x_old) if x_old is not None else None
        x_older_dev = jnp.asarray(x_older) if x_older is not None else None
        use_bdf2 = transient_order >= 2 and x_older is not None
        
        # In jax we use -1 to denote None for integers
        if j_fixed == -1:
            j_fixed_arg = None
        else:
            j_fixed_arg = jnp.array(j_fixed)
            
        res = self.compiled_residual(
            jnp.asarray(x), z, T_in, Y_in, j_fixed_arg, T_fixed_point, T_prof_dev, solve_energy,
            rdt, mask, x_old_dev, x_older_dev, use_bdf2
        )
        return np.asarray(res)

    def evaluate_jacobian(self, x, problem, rdt=0.0):
        import numpy as np
        from scipy import sparse
        z = jnp.asarray(problem.z)
        T_in = float(problem.T_in)
        Y_in = jnp.asarray(problem.Y_in)
        j_fixed = int(problem.j_fixed) if problem.j_fixed is not None else -1
        T_fixed_point = float(problem.T_fixed_point) if hasattr(problem, 'T_fixed_point') else 0.0
        T_prof_dev = jnp.asarray(problem.T_profile_fixed) if getattr(problem, 'T_profile_fixed', None) is not None else None
        solve_energy = bool(problem.solve_energy)
        
        if j_fixed == -1:
            j_fixed_arg = None
        else:
            j_fixed_arg = jnp.array(j_fixed)
            
        J = self.compiled_jacobian(
            jnp.asarray(x), z, T_in, Y_in, j_fixed_arg, T_fixed_point, T_prof_dev, solve_energy,
            0.0, None, None, None, False
        )
        J_np = np.asarray(J)
        
        if rdt > 0.0:
            mask = build_transient_mask(z.shape[0], self.n_sp, solve_energy=solve_energy)
            np.fill_diagonal(J_np, J_np.diagonal() - mask * rdt)
            
        return sparse.csr_matrix(J_np)


