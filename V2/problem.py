"""
problem.py – Configuración del problema de llama libre premezclada 1-D.

FreeFlameProblem almacena:
  * Parámetros del caso (termocinética, transporte, malla, tolerancias).
  * Estado del solver (j_fixed, T_fixed_point, solve_energy, etc.).

Se eliminaron `get_transient_jacobian_pairs` y `get_transient_mask_state`
(ya no necesarios con el nuevo layout por-punto: state.build_transient_mask
los reemplaza correctamente).
"""
from __future__ import annotations

import numpy as np
import cantera as ct

from solver import initial_grid
from state import pack_state, C_Y


def _transition_profile(z, width, left, right, locs=(0.0, 0.3, 0.5, 1.0)):
    xi = np.asarray(z, dtype=float) / width
    _, x1, x2, _ = locs
    out = np.where(
        xi <= x1, left,
        np.where(xi >= x2, right,
                 left + (xi - x1) / (x2 - x1) * (right - left))
    )
    return out


class FreeFlameProblem:
    def __init__(self, case, n_points: int = 8, locs=(0.0, 0.3, 0.5, 1.0)):
        self.case = case
        self.locs = tuple(locs)

        # ---- Termodinámica de referencia ----
        gas = ct.Solution(case.mech)
        gas.TP = case.T_in, case.P
        gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)

        self.species_names = list(gas.species_names)
        self.n_species = gas.n_species
        self.jacobian_mode = "coloring"
        self.n_vars_per_point = 2 + self.n_species   # U, T, Y0..YK

        self.P = case.P
        self.T_in = case.T_in
        self.Y_in = gas.Y.copy()

        # Equilibrio adiabático (para conjetura inicial)
        gas_eq = ct.Solution(case.mech)
        gas_eq.TPY = case.T_in, case.P, self.Y_in
        gas_eq.equilibrate("HP")
        self.T_ad = gas_eq.T
        self.Y_eq = gas_eq.Y.copy()
        self.rho_in = gas.density
        self.rho_eq = gas_eq.density

        # ---- Malla ----
        cluster_sigma = None
        if hasattr(case, "cluster_sigma") and float(case.cluster_sigma) > 0.0:
            cluster_sigma = float(case.cluster_sigma)
        self.z = initial_grid(
            case.width, n_points=n_points, locs=locs,
            cantera_seed_grid=bool(getattr(case, "cantera_seed_grid", True)),
            adaptive=bool(getattr(case, "adaptive_grid", True)),
            cluster_strength=float(getattr(case, "cluster_strength", 8.0)),
            cluster_sigma=cluster_sigma,
        )
        self.n_points = int(self.z.size)
        self.width = float(case.width)

        # ---- Modelo de transporte ----
        self.transport_model = str(getattr(case, "transport_model",
                                           "mixture-averaged"))
        self.flux_gradient_basis = str(getattr(case, "flux_gradient_basis",
                                               "molar")).strip().lower()
        self.soret_enabled = bool(getattr(case, "soret_enabled", False))

        # ---- Opciones de refinamiento ----
        self.refine_with_u = bool(getattr(case, "refine_with_u", True))
        self.refine_with_T = bool(getattr(case, "refine_with_T", True))
        self.refine_with_species = bool(getattr(case, "refine_with_species", True))

        # ---- Posición de anclaje de T (free-flame) ----
        self.anchor_z = 0.5 * (locs[1] + locs[2]) * case.width
        self.anchor_T = 0.75 * case.T_in + 0.25 * self.T_ad

        # ---- Estado del solver ----
        self.j_fixed: int | None = None
        self.T_fixed_point: float | None = None
        self.solve_energy: bool = True
        self.T_profile_fixed: np.ndarray | None = None

        # Perfiles de referencia (opcionales, para comparación)
        self.u_fixed: np.ndarray | None = None
        self.T_fixed: np.ndarray | None = None
        self.Y_ref: np.ndarray | None = None
        self.mdot_fixed: float | None = None

        # Backend (se asigna externamente tras construir el problema)
        self.backend = None

        # Diagnóstico
        self.last_residual_error: str | None = None
        self.last_newton_status: int | None = None
        self.last_newton_message: str | None = None

        # ---- Tolerancias (replicando Domain1D defaults) ----
        nv = self.n_vars_per_point
        self.steady_rtol   = 1e-4    # escalar; OneDim::weightedNorm lo usa por componente
        self.steady_atol   = 1e-9
        self.transient_rtol = 1e-4
        self.transient_atol = 1e-11

        # Bounds por componente (C_U=0, C_T=1, C_Y=2..)
        self.T_lower_bound = 200.0
        self.T_upper_bound = 2.0 * float(gas.max_temp)
        self.Y_lower_bound = -1e-7  # Cantera: lo=-1e-7 para todas las especies

        # Gas auxiliar para normalización
        self._reset_gas = ct.Solution(case.mech)
        self._reset_gas.TP = case.T_in, case.P

    # ------------------------------------------------------------------
    #  Saneado de estado (Domain1D::resetBadValues)
    # ------------------------------------------------------------------
    def _normalize_Y_col(self, Y_col: np.ndarray) -> np.ndarray:
        Y_col = np.asarray(Y_col, dtype=float).copy()
        Y_col[~np.isfinite(Y_col)] = 0.0
        Y_col = np.clip(Y_col, 0.0, None)
        s = float(Y_col.sum())
        if s <= 0.0:
            return self.Y_in.copy()
        self._reset_gas.Y = Y_col
        return self._reset_gas.Y.copy()

    def _sanitize_Y(self, Y: np.ndarray) -> np.ndarray:
        """Y (n_sp, n_pts) -> normalizada por columna."""
        Y = np.asarray(Y, dtype=float).copy()
        if Y.ndim != 2:
            return Y

        # Ruta vectorizada: elimina NaN/inf, recorta negativos y normaliza.
        Y[~np.isfinite(Y)] = 0.0
        np.clip(Y, 0.0, None, out=Y)
        s = Y.sum(axis=0)

        bad = s <= 0.0
        if np.any(bad):
            Y[:, bad] = self.Y_in[:, None]
            s[bad] = 1.0

        Y /= s[None, :]
        return Y
    def _sanitize_Y_full(self, Y: np.ndarray) -> np.ndarray:
        """Alias público para compatibilidad con solver_cantera_auto."""
        return self._sanitize_Y(Y)

    def reset_bad_values(self, x: np.ndarray) -> np.ndarray:
        """
        Replica OneDim::resetBadValues + Flow1D::resetBadValues.

        Espera vector de estado en layout por-punto (state.py).
        """
        x = np.asarray(x, dtype=float).copy()
        n_pts = self.n_points
        nv = self.n_vars_per_point
        n_sp = self.n_species

        if x.size != n_pts * nv:
            return x

        x_r = x.reshape(n_pts, nv)

        # Normalizar Y, igual que Flow1D::resetBadValues.
        Y = x_r[:, C_Y:].T.copy()   # (n_sp, n_pts)
        Y = self._sanitize_Y(Y)
        x_r[:, C_Y:] = Y.T

        return x_r.ravel()

    # ------------------------------------------------------------------
    #  Punto fijo de T (free-flame anchor)
    # ------------------------------------------------------------------
    def setup_fixed_temperature(self, T_fixed: float | None = None,
                                z_fixed: float | None = None,
                                T_profile: np.ndarray | None = None):
        """
        Determina j_fixed y T_fixed_point para el anchor de temperatura.
        Reproduce el mecanismo de m_tfixed / m_zfixed en Flow1D.
        """
        T_eff = float(T_fixed) if T_fixed is not None else self.anchor_T

        if z_fixed is None:
            # Buscar el punto de la malla más cercano a T_eff en T_profile
            if T_profile is None:
                T_profile = (self.T_profile_fixed if self.T_profile_fixed is not None
                             else self.T_fixed)
            if T_profile is not None and T_profile.shape == (self.n_points,):
                j_sel = int(np.argmin(np.abs(T_profile - T_eff)))
                T_eff = float(T_profile[j_sel])
            else:
                j_sel = int(np.argmin(np.abs(self.z - self.anchor_z)))
        else:
            j_sel = int(np.argmin(np.abs(self.z - float(z_fixed))))

        # Evitar fronteras
        j_sel = int(np.clip(j_sel, 1, self.n_points - 2))
        self.j_fixed = j_sel
        self.T_fixed_point = T_eff

    # ------------------------------------------------------------------
    #  Conjetura inicial
    # ------------------------------------------------------------------
    def make_initial_guess(self, u_left_guess: float = 0.30) -> np.ndarray:
        mdot = self.rho_in * u_left_guess
        u_right = mdot / self.rho_eq

        u0 = _transition_profile(self.z, self.width, u_left_guess, u_right, self.locs)
        T0 = _transition_profile(self.z, self.width, self.T_in, self.T_ad, self.locs)
        Y0 = np.row_stack([
            _transition_profile(self.z, self.width, self.Y_in[k], self.Y_eq[k], self.locs)
            for k in range(self.n_species)
        ])
        return pack_state(u0, T0, Y0)

    # ------------------------------------------------------------------
    #  Resumen
    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"n_points     = {self.n_points}",
            f"n_species    = {self.n_species}",
            f"width [m]    = {self.width:.6f}",
            f"T_in [K]     = {self.T_in:.3f}",
            f"T_ad [K]     = {self.T_ad:.3f}",
            f"transport    = {self.transport_model}",
            f"flux_basis   = {self.flux_gradient_basis}",
        ]
        if self.j_fixed is not None:
            lines += [
                f"j_fixed      = {self.j_fixed}",
                f"T_fixed_pt   = {self.T_fixed_point:.3f}",
                f"z_fixed [m]  = {self.z[self.j_fixed]:.4e}",
            ]
        return "\n".join(lines)
