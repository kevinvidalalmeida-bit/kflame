"""
residual_flame.py – Residual acoplado (T + especies) con u congelado.

Resuelve T y Y simultáneamente con el perfil u(z) del reference fijo.

FIXES:
  * BUG 4 (CRÍTICO): mix_diff_coeffs_mass se obtenía de backend.gas
    (objeto single-state con el ÚLTIMO estado seteado) en vez del
    SolutionArray states → daba Dm solo del último punto para todos.
    Corregido: usa states.mix_diff_coeffs_mass (shape n_pts × n_sp).
  * BUG 5: mdot se calculaba como u_fixed * rho_node[0] (escalar del
    primer nodo) en vez de problem.mdot_fixed. Corregido.
"""
import numpy as np
from state import unpack_temperature_species


def _eval_all_properties_coupled(T, Y, backend, n_pts):
    """
    Evalúa las propiedades termodinámicas en todos los nodos usando
    SolutionArray para eficiencia.

    Retorna
    -------
    rho      : (n_pts,)
    cp       : (n_pts,)
    lambda_  : (n_pts,)
    hk       : (n_species, n_pts)  [J/kg]
    cpk      : (n_species, n_pts)  [J/kg/K]
    Dm       : (n_species, n_pts)  [m²/s]
    omega    : (n_species, n_pts)  [kmol/m³/s]
    """
    import cantera as ct

    if not hasattr(backend, "states_array") or len(backend.states_array) != n_pts:
        backend.states_array = ct.SolutionArray(backend.gas, n_pts)

    states = backend.states_array
    states.TPY = T, backend.problem.case.P, Y.T  # Y.T shape: (n_pts, n_sp)

    rho = states.density           # (n_pts,)
    cp = states.cp_mass            # (n_pts,)
    lambda_ = states.thermal_conductivity  # (n_pts,)

    # Entalpías parciales molares [J/kmol] → dividir por W [kg/kmol] → [J/kg]
    hk_molar = states.partial_molar_enthalpies  # (n_pts, n_sp)
    hk = (hk_molar / backend.W).T              # (n_sp, n_pts)

    # Cp parciales molares [J/kmol/K] → [J/kg/K]
    cpk_molar = states.partial_molar_cp        # (n_pts, n_sp)
    cpk = (cpk_molar / backend.W).T            # (n_sp, n_pts)

    # FIX BUG 4: usar states.mix_diff_coeffs_mass (shape n_pts × n_sp)
    # en vez de backend.gas.mix_diff_coeffs_mass (último estado, shape n_sp)
    Dm_raw = states.mix_diff_coeffs_mass       # (n_pts, n_sp)
    Dm = Dm_raw.T                              # (n_sp, n_pts)

    omega = states.net_production_rates.T      # (n_sp, n_pts) [kmol/m³/s]

    return rho, cp, lambda_, hk, cpk, Dm, omega


def residual_flame(x, problem):
    """
    Calcula el residual acoplado (Energía y Especies) con u(z) congelado.
    x = [T, Y1, ..., Y_{N-1}] para todos los nodos.

    Retorna F = [F_T, F_Y1, ..., F_Y_{N-1}] de tamaño n_pts * n_species.
    """
    n_pts = problem.n_points
    n_sp = problem.n_species
    n_ind = n_sp - 1

    T, Y = unpack_temperature_species(x, n_pts, n_sp)
    Y_ind = Y[:-1, :]

    backend = problem.backend
    W_k = backend.W[:, None]  # (n_sp, 1)

    # Clip Y para evitar crashes en Cantera
    Y_clip = np.clip(Y, 1e-15, 1.0)

    rho_node, cp_node, lambda_node, hk_node, cpk_node, D_node, omega_node = \
        _eval_all_properties_coupled(T, Y_clip, backend, n_pts)

    # FIX BUG 5: usar mdot_fixed directamente en vez de u_fixed * rho_node[0]
    # mdot = rho * u = cte para flujo conservativo
    mdot = problem.mdot_fixed

    z = problem.z
    dz = np.diff(z)

    # ==========================================
    # DIFUSIÓN (caras de los volúmenes de control)
    # ==========================================
    rho_face   = 0.5 * (rho_node[:-1]     + rho_node[1:])
    D_face     = 0.5 * (D_node[:, :-1]    + D_node[:, 1:])     # (n_sp, n_faces)
    lambda_face = 0.5 * (lambda_node[:-1]  + lambda_node[1:])

    dY_dz_face = np.diff(Y, axis=1) / dz      # (n_sp, n_faces)
    dT_dz_face = np.diff(T) / dz              # (n_faces,)

    # Flujos difusivos de especies: J_k = -rho D_k dY_k/dz
    J_face = -rho_face * D_face * dY_dz_face   # (n_sp, n_faces)

    # Corrección masa (conservación)
    sum_J  = np.sum(J_face, axis=0)            # (n_faces,)
    Y_face = 0.5 * (Y[:, :-1] + Y[:, 1:])     # (n_sp, n_faces)
    J_corr = J_face - Y_face * sum_J           # (n_sp, n_faces)

    # Flujo de calor por conducción: q = -lambda dT/dz
    q_face = -lambda_face * dT_dz_face         # (n_faces,)

    # ==========================================
    # NODOS INTERNOS (j = 1 ... N-2)
    # ==========================================
    R_T = np.zeros(n_pts)
    R_Y = np.zeros((n_ind, n_pts))

    # dz_node = (z_{j+1} - z_{j-1}) / 2  para nodos interiores
    dz_node = np.zeros(n_pts)
    dz_node[1:-1] = 0.5 * (z[2:] - z[:-2])

    # Convección de especies (upwind, mdot > 0 supuesto)
    dY_dz_upwind = (Y_ind[:, 1:-1] - Y_ind[:, :-2]) / dz[:-1]
    conv_Y = mdot * dY_dz_upwind                # (n_ind, n_interior)

    # Divergencia difusiva de especies
    diff_Y = (J_corr[:-1, 1:] - J_corr[:-1, :-1]) / dz_node[1:-1]  # (n_ind, n_interior)

    R_Y[:, 1:-1] = (omega_node[:-1, 1:-1] * W_k[:-1] - conv_Y - diff_Y) \
                   / rho_node[1:-1]

    # Convección de energía (upwind)
    dT_dz_upwind = (T[1:-1] - T[:-2]) / dz[:-1]
    conv_T = mdot * cp_node[1:-1] * dT_dz_upwind

    # Conducción: d(lambda dT/dz)/dz  (positivo: divergencia del flujo q)
    # diff_T = (q_{j+1/2} - q_{j-1/2}) / dz_node = d(-lambda dT/dz)/dz
    diff_T = (q_face[1:] - q_face[:-1]) / dz_node[1:-1]

    # Calor de reacción: Σ_k h_k * W_k * omega_k  [J/m³/s]
    q_dot = np.sum(hk_node[:, 1:-1] * omega_node[:, 1:-1] * W_k, axis=0)

    # Transporte de entalpía por difusión: Σ_k J_k * dh_k/dz
    # dh_k/dz ≈ cp_k * dT/dz  (aproximación de temperatura pura)
    dT_dz_node = np.zeros(n_pts)
    dT_dz_node[1:-1] = (T[2:] - T[:-2]) / (z[2:] - z[:-2])
    dh_dz_node = cpk_node[:, 1:-1] * dT_dz_node[1:-1]  # (n_sp, n_interior)
    J_node = 0.5 * (J_corr[:, :-1] + J_corr[:, 1:])     # (n_sp, n_interior)
    enthalpy_diff = np.sum(J_node * dh_dz_node, axis=0)   # (n_interior,)

    # R_T = (-conv_T - diff_T - q_dot - enthalpy_diff) / (rho * cp)
    # Nota: diff_T = d(-lambda dT/dz)/dz = -d(lambda dT/dz)/dz, por eso:
    # -diff_T = +d(lambda dT/dz)/dz  ← término de conducción positivo
    R_T[1:-1] = (-conv_T - diff_T - q_dot - enthalpy_diff) \
                / (rho_node[1:-1] * cp_node[1:-1])

    # ==========================================
    # CONDICIONES DE FRONTERA
    # ==========================================
    # Entrada (Dirichlet) – fijar al perfil de referencia
    R_T[0]    = T[0]       - problem.T_fixed[0]
    R_Y[:, 0] = Y_ind[:, 0] - problem.Y_ref[:-1, 0]

    # Salida (Neumann = gradiente nulo)
    R_T[-1]    = T[-1]        - T[-2]
    R_Y[:, -1] = Y_ind[:, -1] - Y_ind[:, -2]

    # Construir F = [R_T, R_Y1, ..., R_Y_{N-1}]
    F = np.empty((1 + n_ind) * n_pts)
    F[:n_pts]  = R_T
    F[n_pts:] = R_Y.reshape(-1, order="C")
    return F


def residual_flame_transient(xn, xo, problem, rdt):
    """
    Añade el término transitorio (Backward Euler) al residual acoplado.
    F_ts = F_ss - rdt * (xn - xo)   (solo nodos interiores)
    """
    F_ss = residual_flame(xn, problem)

    n_pts = problem.n_points
    n_sp  = problem.n_species
    n_ind = n_sp - 1

    Tn, Yn = unpack_temperature_species(xn, n_pts, n_sp)
    To, Yo = unpack_temperature_species(xo, n_pts, n_sp)

    R_T_ts = F_ss[:n_pts].copy()
    R_T_ts[1:-1] -= rdt * (Tn[1:-1] - To[1:-1])

    R_Y_ss = F_ss[n_pts:].reshape((n_ind, n_pts), order="C")
    R_Y_ts = R_Y_ss.copy()
    R_Y_ts[:, 1:-1] -= rdt * (Yn[:-1, 1:-1] - Yo[:-1, 1:-1])

    # Fronteras no tienen término transitorio
    R_T_ts[0]    = F_ss[0]
    R_T_ts[-1]   = F_ss[n_pts - 1]
    R_Y_ts[:, 0] = R_Y_ss[:, 0]
    R_Y_ts[:, -1] = R_Y_ss[:, -1]

    F_ts = np.empty((1 + n_ind) * n_pts)
    F_ts[:n_pts] = R_T_ts
    F_ts[n_pts:] = R_Y_ts.reshape(-1, order="C")
    return F_ts