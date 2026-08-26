"""
species_backend_native.py – Drop-in replacement for species_backend.py.

Provides the exact same interface as SpeciesBackend but uses:
  - mechanism_data.py   → YAML parser
  - thermo_native.py    → NASA-7 thermodynamics
  - transport_native.py → mixture-averaged transport (Neufeld collision integrals)
  - kinetics_native.py  → full chemical kinetics (Arrhenius/3-body/Troe)

ZERO Cantera dependency. CPU backend via NumPy and Numba.
"""
from __future__ import annotations
import math
import numpy as np

from mechanism_data import MechanismData, load_mechanism, R_UNIV
from thermo_native import NativeThermo
from transport_native import NativeTransport
from kinetics_native import NativeKinetics

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional acceleration
    njit = None
    prange = range

if njit is not None:
    @njit(cache=True)
    def _eval_thermo_sparse_numba_core(
        T_arr, Y, P, nasa_lo, nasa_hi, nasa_tmid, invW,
        rho_out, cp_out, hk_out,
    ):
        """NASA thermo and ideal-gas density, without reaction kinetics."""
        n_sp = Y.shape[0]
        n_pts = Y.shape[1]

        for m in range(n_pts):
            T = T_arr[m]
            inv_wmix = 0.0
            for k in range(n_sp):
                inv_wmix += Y[k, m] * invW[k]
            if inv_wmix < 1.0e-300:
                inv_wmix = 1.0e-300

            rho = P / (R_UNIV * T * inv_wmix)
            rho_out[m] = rho

            cp_mix = 0.0
            for k in range(n_sp):
                c = nasa_hi[k] if T > nasa_tmid[k] else nasa_lo[k]
                cp_R = (
                    c[0]
                    + T * (c[1] + T * (c[2] + T * (c[3] + T * c[4])))
                )
                h_RT = (
                    c[0]
                    + T * (
                        c[1] / 2.0
                        + T * (
                            c[2] / 3.0
                            + T * (c[3] / 4.0 + T * c[4] / 5.0)
                        )
                    )
                    + c[5] / T
                )
                hk_out[k, m] = h_RT * R_UNIV * T
                cp_mix += Y[k, m] * cp_R * R_UNIV * invW[k]

            cp_out[m] = cp_mix

    @njit(cache=True, parallel=True)
    def _eval_thermo_kinetics_sparse_numba_core(
        T_arr, Y, P, nasa_lo, nasa_hi, nasa_tmid, W, invW,
        A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
        is_three_body, is_falloff, is_reversible, has_troe,
        troe_A, troe_T3, troe_T1, troe_T2,
        r_idx, r_nu, r_count, p_idx, p_nu, p_count,
        net_idx, net_nu, net_count, eff_idx, eff_delta, eff_count,
        delta_nu, rho_out, cp_out, omega_out, hk_out,
    ):
        n_sp = Y.shape[0]
        n_pts = Y.shape[1]
        n_rxn = A_hi.shape[0]

        for m in prange(n_pts):
            # Per-node work arrays are intentionally local: this kernel is
            # embarrassingly parallel over grid/perturbation points.
            logC = np.empty(n_sp, dtype=np.float64)
            g_RT = np.empty(n_sp, dtype=np.float64)
            T = T_arr[m]
            inv_RT = 1.0 / (R_UNIV * T)
            inv_wmix = 0.0
            for k in range(n_sp):
                inv_wmix += Y[k, m] * invW[k]
            if inv_wmix < 1.0e-300:
                inv_wmix = 1.0e-300
            Wmix = 1.0 / inv_wmix
            rho = P * Wmix / (R_UNIV * T)
            rho_out[m] = rho

            logT = math.log(T)
            cp_mix = 0.0
            c_total = 0.0
            for k in range(n_sp):
                c = nasa_hi[k] if T > nasa_tmid[k] else nasa_lo[k]
                cp_R = (
                    c[0]
                    + T * (c[1] + T * (c[2] + T * (c[3] + T * c[4])))
                )
                h_RT = (
                    c[0]
                    + T * (
                        c[1] / 2.0
                        + T * (
                            c[2] / 3.0
                            + T * (c[3] / 4.0 + T * c[4] / 5.0)
                        )
                    )
                    + c[5] / T
                )
                s_R = (
                    c[0] * logT
                    + T * (
                        c[1]
                        + T * (
                            c[2] / 2.0
                            + T * (c[3] / 3.0 + T * c[4] / 4.0)
                        )
                    )
                    + c[6]
                )
                hk_out[k, m] = h_RT * R_UNIV * T
                g_RT[k] = h_RT - s_R
                cp_mix += Y[k, m] * cp_R * R_UNIV * invW[k]

                conc = rho * Y[k, m] * invW[k]
                if conc < 0.0:
                    conc = 0.0
                c_total += conc
                if conc < 1.0e-300:
                    conc = 1.0e-300
                logC[k] = math.log(conc)
                omega_out[k, m] = 0.0

            cp_out[m] = cp_mix
            c_factor = 101325.0 / (R_UNIV * T)

            for r in range(n_rxn):
                # T**b * exp(-Ea / RT) as one exponential. ``logT`` is already
                # required by the NASA polynomials above for this grid point.
                kf = A_hi[r] * math.exp(b_hi[r] * logT - Ea_hi[r] * inv_RT)

                rf_exp = 0.0
                for ii in range(r_count[r]):
                    k = r_idx[r, ii]
                    rf_exp += r_nu[r, ii] * logC[k]

                rr_exp = 0.0
                if is_reversible[r]:
                    for ii in range(p_count[r]):
                        k = p_idx[r, ii]
                        rr_exp += p_nu[r, ii] * logC[k]

                delta_g = 0.0
                for ii in range(net_count[r]):
                    k = net_idx[r, ii]
                    delta_g += net_nu[r, ii] * g_RT[k]

                if delta_g > 500.0:
                    delta_g = 500.0
                elif delta_g < -500.0:
                    delta_g = -500.0

                Kc = math.exp(-delta_g) * (c_factor ** delta_nu[r])
                kr = 0.0
                if is_reversible[r]:
                    kr = kf / max(Kc, 1.0e-300)

                Rf = math.exp(rf_exp) * kf
                Rr = 0.0
                if is_reversible[r]:
                    Rr = math.exp(rr_exp) * kr

                M = c_total
                if is_three_body[r] or is_falloff[r]:
                    for ii in range(eff_count[r]):
                        k = eff_idx[r, ii]
                        ck = rho * Y[k, m] * invW[k]
                        if ck < 0.0:
                            ck = 0.0
                        M += eff_delta[r, ii] * ck

                if is_three_body[r]:
                    Rf *= M
                    Rr *= M

                if is_falloff[r]:
                    k0 = A_lo[r] * math.exp(b_lo[r] * logT - Ea_lo[r] * inv_RT)
                    Pr = k0 * M / max(kf, 1.0e-300)
                    F_lind = Pr / (1.0 + Pr)
                    F = 1.0

                    if has_troe[r]:
                        t3 = max(troe_T3[r], 1.0e-300)
                        t1 = max(troe_T1[r], 1.0e-300)
                        Fcent = (
                            (1.0 - troe_A[r]) * math.exp(-T / t3)
                            + troe_A[r] * math.exp(-T / t1)
                            + math.exp(-troe_T2[r] / max(T, 1.0e-300))
                        )
                        Fcent = max(Fcent, 1.0e-300)
                        logFcent = math.log10(Fcent)
                        logPr = math.log10(max(Pr, 1.0e-300))
                        c_troe = -0.4 - 0.67 * logFcent
                        n_troe = 0.75 - 1.27 * logFcent
                        d_troe = 0.14
                        f1 = (logPr + c_troe) / (
                            n_troe - d_troe * (logPr + c_troe)
                        )
                        F = 10.0 ** (logFcent / (1.0 + f1 * f1))

                    kf_falloff = kf * F_lind * F
                    Rf = kf_falloff * (Rf / max(kf, 1.0e-300))
                    if is_reversible[r]:
                        kr_falloff = kf_falloff / max(Kc, 1.0e-300)
                        Rr = kr_falloff * (Rr / max(kr, 1.0e-300))

                q = Rf - Rr
                for ii in range(net_count[r]):
                    k = net_idx[r, ii]
                    omega_out[k, m] += net_nu[r, ii] * q * W[k]
else:
    _eval_thermo_sparse_numba_core = None
    _eval_thermo_kinetics_sparse_numba_core = None


class NativeSpeciesBackend:
    """
    Drop-in replacement for SpeciesBackend.

    Same public methods:
      - eval_node_into(T, Y, omega_out, hk_out)
      - eval_node(T, Y)
      - eval_midpoint(T_left, T_right, Y_left, Y_right)
      - eval_midpoint_full_transport(T_left, T_right, Y_left, Y_right)
      - density(T, Y)

    All thermo, transport, and kinetics are evaluated natively.
    """

    def __init__(self, problem, mech_data: MechanismData | None = None):
        self.problem = problem
        self.backend_kind = "native"
        P = problem.P
        self.xp = np

        # Load mechanism if not provided
        if mech_data is None:
            mech_path = problem.case.mech
            mech_data = load_mechanism(mech_path)

        self.mech = mech_data
        self.mech.pressure = P

        self.thermo    = NativeThermo(mech_data, xp=self.xp)
        self.transport = NativeTransport(mech_data, xp=self.xp)
        self.kinetics  = NativeKinetics(mech_data, xp=self.xp)
        if getattr(self.kinetics, "_numba_available", False):
            self.kinetics._use_numba = bool(getattr(problem, "use_numba_kinetics", False))
        if getattr(self.kinetics, "_sparse_numba_available", False):
            self.kinetics._use_sparse_numba = bool(
                getattr(problem, "use_sparse_numba_kinetics", True)
            )

        self.W = np.asarray(mech_data.molecular_weights, dtype=float)
        self.invW = np.asarray(mech_data.inv_molecular_weights, dtype=float)
        self.W_dev = self.xp.asarray(self.W)
        self.invW_dev = self.xp.asarray(self.invW)
        self.n_species = mech_data.n_species
        self._P_float = float(P)
        self._nb_nasa_lo = np.asarray(self.thermo._lo, dtype=np.float64)
        self._nb_nasa_hi = np.asarray(self.thermo._hi, dtype=np.float64)
        self._nb_nasa_tmid = np.asarray(self.thermo._Tmid, dtype=np.float64)

        # Transport / flux settings from problem
        self.flux_gradient_basis = str(
            getattr(problem, "flux_gradient_basis",
                    getattr(problem.case, "flux_gradient_basis", "molar"))
        ).strip().lower()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _safe_Y(self, Y: np.ndarray) -> np.ndarray:
        """
        Ensure finite composition without renormalizing.

        This mirrors the solver path using Cantera's
        `set_unnormalized_mass_fractions`, which preserves the incoming state.
        """
        Y = self.xp.asarray(Y, dtype=float)
        if bool(getattr(self.problem, "assume_finite_y", False)):
            return Y
        finite = self.xp.all(self.xp.isfinite(Y))
        finite = bool(finite)
        if not finite:
            raise ValueError("Se detectaron fracciones másicas no finitas.")
        return Y

    def _to_host(self, arr):
        return np.asarray(arr)

    def _copy_to_out(self, out: np.ndarray, arr):
        out[:] = arr

    def _Y_to_X(self, Y: np.ndarray) -> np.ndarray:
        """Mass fractions → mole fractions."""
        return self.thermo.Y_to_X(Y)

    def _concentrations(self, T: float, Y: np.ndarray) -> np.ndarray:
        """Species concentrations [kmol/m³]."""
        rho = self.thermo.density(T, self.problem.P, Y)
        if Y.ndim == 1:
            return rho * Y * self.invW_dev
        else:
            return rho[None, :] * Y * self.invW_dev[:, None]

    # ------------------------------------------------------------------
    #  eval_grid_into  (Full Grid Vectorization)
    # ------------------------------------------------------------------
    def eval_grid_native(self, T: np.ndarray, Y: np.ndarray):
        """
        Evaluate all nodal properties on the CPU backend.

        Returns NumPy arrays:
        (rho, Dm, cp, lam, omega_mass, hk).
        """
        P = self.problem.P
        T_dev = self.xp.asarray(T, dtype=float)
        Y_safe = self._safe_Y(Y)

        rho = self.thermo.density(T_dev, P, Y_safe)
        cp = self.thermo.cp_mass(T_dev, Y_safe)
        hk_vals = self.thermo.partial_molar_enthalpies(T_dev)

        cp_R = self.thermo.cp_R(T_dev)
        _mu, lam, Dm, _X = self.transport.eval_all(T_dev, P, Y_safe, cp_R, self.invW_dev)

        C = rho[None, :] * Y_safe * self.invW_dev[:, None]
        g_RT = self.thermo.g_RT(T_dev)
        wdot = self.kinetics.net_production_rates(T_dev, C, g_RT)
        omega_mass = wdot * self.W_dev[:, None]

        return rho, Dm, cp, lam, omega_mass, hk_vals

    def eval_grid_thermo_kinetics_native(self, T: np.ndarray, Y: np.ndarray):
        """
        Evaluate nodal thermo and kinetics on the active array backend.

        Returns NumPy arrays:
        (rho, cp, omega_mass, hk).
        """
        P = self.problem.P
        T_dev = self.xp.asarray(T, dtype=float)
        Y_safe = self._safe_Y(Y)

        inv_wmix = self.xp.sum(Y_safe * self.invW_dev[:, None], axis=0)
        Wmix = 1.0 / self.xp.maximum(inv_wmix, 1e-300)
        rho = P * Wmix / (R_UNIV * T_dev)
        cp, hk_vals, g_RT = self.thermo.cp_mass_hk_g_RT(T_dev, Y_safe)

        C = rho[None, :] * Y_safe * self.invW_dev[:, None]
        wdot = self.kinetics.net_production_rates(T_dev, C, g_RT)
        omega_mass = wdot * self.W_dev[:, None]

        return rho, cp, omega_mass, hk_vals

    def eval_grid_into(self, T: np.ndarray, Y: np.ndarray,
                       omega_out: np.ndarray, hk_out: np.ndarray):
        """
        Evaluate properties at all grid nodes simultaneously.
        T is (N,), Y is (n_sp, N).
        omega_out is (n_sp, N), hk_out is (n_sp, N).
        Returns (rho, Dm, cp, lam).
        """
        rho, Dm, cp, lam, omega_mass, hk_vals = self.eval_grid_native(T, Y)
        self._copy_to_out(hk_out, hk_vals)
        self._copy_to_out(omega_out, omega_mass)
        return self._to_host(rho), self._to_host(Dm), self._to_host(cp), self._to_host(lam)

    def eval_grid_thermo_kinetics_into(self, T: np.ndarray, Y: np.ndarray,
                                       omega_out: np.ndarray, hk_out: np.ndarray):
        """
        Evaluate nodal thermo and kinetics without transport.

        The local Jacobian freezes transport coefficients, matching Cantera's
        Jacobian path. For local perturbations, only density, cp, enthalpy and
        production rates need to be refreshed.
        """
        if (
            _eval_thermo_kinetics_sparse_numba_core is not None
            and bool(getattr(self.problem, "use_fused_numba_thermo_kinetics", True))
            and getattr(self.kinetics, "_sparse_numba_available", False)
        ):
            T_work = np.asarray(T, dtype=np.float64).reshape(-1)
            Y_work = np.ascontiguousarray(self._safe_Y(Y), dtype=np.float64)
            if Y_work.ndim == 1:
                Y_work = np.ascontiguousarray(Y_work[:, None], dtype=np.float64)
            if Y_work.shape[1] != T_work.size:
                raise ValueError("T and Y sizes do not match for thermo/kinetics evaluation.")

            rho = np.empty(T_work.size, dtype=np.float64)
            cp = np.empty(T_work.size, dtype=np.float64)
            _eval_thermo_kinetics_sparse_numba_core(
                T_work,
                Y_work,
                self._P_float,
                self._nb_nasa_lo,
                self._nb_nasa_hi,
                self._nb_nasa_tmid,
                self.W,
                self.invW,
                self.kinetics._nb_A_hi,
                self.kinetics._nb_b_hi,
                self.kinetics._nb_Ea_hi,
                self.kinetics._nb_A_lo,
                self.kinetics._nb_b_lo,
                self.kinetics._nb_Ea_lo,
                self.kinetics._nb_is_three_body,
                self.kinetics._nb_is_falloff,
                self.kinetics._nb_is_reversible,
                self.kinetics._nb_has_troe,
                self.kinetics._nb_troe_A,
                self.kinetics._nb_troe_T3,
                self.kinetics._nb_troe_T1,
                self.kinetics._nb_troe_T2,
                self.kinetics._sp_r_idx,
                self.kinetics._sp_r_nu,
                self.kinetics._sp_r_count,
                self.kinetics._sp_p_idx,
                self.kinetics._sp_p_nu,
                self.kinetics._sp_p_count,
                self.kinetics._sp_net_idx,
                self.kinetics._sp_net_nu,
                self.kinetics._sp_net_count,
                self.kinetics._sp_eff_idx,
                self.kinetics._sp_eff_delta,
                self.kinetics._sp_eff_count,
                self.kinetics._nb_delta_nu,
                rho,
                cp,
                omega_out,
                hk_out,
            )
            return rho, cp

        rho, cp, omega_mass, hk_vals = self.eval_grid_thermo_kinetics_native(T, Y)

        self._copy_to_out(hk_out, hk_vals)
        self._copy_to_out(omega_out, omega_mass)
        return self._to_host(rho), self._to_host(cp)

    def eval_grid_thermo_only_into(self, T: np.ndarray, Y: np.ndarray,
                                   hk_out: np.ndarray):
        """Evaluate density, mixture heat capacity and species enthalpies.

        This is used by the local finite-difference Jacobian once the chemical
        source derivative is supplied analytically.  It deliberately omits
        kinetics, which otherwise dominates the cost of one Jacobian build.
        """
        if _eval_thermo_sparse_numba_core is not None:
            T_work = np.asarray(T, dtype=np.float64).reshape(-1)
            Y_work = np.ascontiguousarray(self._safe_Y(Y), dtype=np.float64)
            if Y_work.ndim == 1:
                Y_work = np.ascontiguousarray(Y_work[:, None], dtype=np.float64)
            if Y_work.shape[1] != T_work.size:
                raise ValueError("T and Y sizes do not match for thermodynamic evaluation.")

            rho = np.empty(T_work.size, dtype=np.float64)
            cp = np.empty(T_work.size, dtype=np.float64)
            _eval_thermo_sparse_numba_core(
                T_work,
                Y_work,
                self._P_float,
                self._nb_nasa_lo,
                self._nb_nasa_hi,
                self._nb_nasa_tmid,
                self.invW,
                rho,
                cp,
                hk_out,
            )
            return rho, cp

        T_dev = self.xp.asarray(T, dtype=float)
        Y_safe = self._safe_Y(Y)
        rho = self.thermo.density(T_dev, self.problem.P, Y_safe)
        cp, hk_vals, _g_rt = self.thermo.cp_mass_hk_g_RT(T_dev, Y_safe)
        self._copy_to_out(hk_out, hk_vals)
        return self._to_host(rho), self._to_host(cp)

    def eval_faces(self, T_face: np.ndarray, Y_face: np.ndarray):
        """
        Evaluate full transport properties at all faces simultaneously.
        T_face is (N-1,), Y_face is (n_sp, N-1).
        Returns (rho_f, Dm_f, lam_f, Wmix_f).
        """
        rho, Dm, lam, Wmix = self.eval_faces_native(T_face, Y_face)
        return self._to_host(rho), self._to_host(Dm), self._to_host(lam), self._to_host(Wmix)

    def eval_faces_native(self, T_face: np.ndarray, Y_face: np.ndarray):
        """
        Evaluate face transport properties on the active array backend.

        Returns NumPy arrays:
        (rho_f, Dm_f, lam_f, Wmix_f).
        """
        P = self.problem.P
        T_dev = self.xp.asarray(T_face, dtype=float)
        Y_safe = self._safe_Y(Y_face)

        if bool(getattr(self.problem, "use_numba_transport", True)):
            fast = self.transport.eval_faces_poly_fast(T_dev, P, Y_safe, self.invW_dev)
            if fast is not None:
                return fast

        inv_wmix = self.xp.sum(Y_safe * self.invW_dev[:, None], axis=0)
        Wmix = 1.0 / self.xp.maximum(inv_wmix, 1e-300)
        rho = P * Wmix / (R_UNIV * T_dev)
        X = Y_safe * Wmix[None, :] * self.invW_dev[:, None]
        if bool(getattr(self.transport, "_has_cond_poly_cpu", np.array([False])).all()):
            cp_R = None
        else:
            cp_R = self.thermo.cp_R(T_dev)
        lam = self.transport.thermal_conductivity(T_dev, X, cp_R)
        Dm = self.transport.mix_diff_coeffs(T_dev, P, X)
        return rho, Dm, lam, Wmix

    # ------------------------------------------------------------------
    #  eval_node_into  (Single point interface)
    # ------------------------------------------------------------------
    def eval_node_into(self, T: float, Y: np.ndarray,
                       omega_out: np.ndarray, hk_out: np.ndarray):
        """Evaluate properties at a grid node. Writes omega and hk in-place."""
        T = float(T)
        P = self.problem.P
        Y = self._safe_Y(Y)

        # Thermodynamics
        rho = self.thermo.density(T, P, Y)
        cp  = self.thermo.cp_mass(T, Y)
        hk_vals = self.thermo.partial_molar_enthalpies(T)
        self._copy_to_out(hk_out, hk_vals)

        # Transport
        cp_R = self.thermo.cp_R(T)
        X = self._Y_to_X(Y)
        _mu, lam, Dm, _X = self.transport.eval_all(T, P, Y, cp_R, self.invW_dev)

        # Kinetics
        C = rho * Y * self.invW_dev
        g_RT = self.thermo.g_RT(T)
        wdot = self.kinetics.net_production_rates(T, C, g_RT)
        omega_mass = wdot * self.W_dev
        self._copy_to_out(omega_out, omega_mass)

        return (
            float(np.asarray(self._to_host(rho))),
            self._to_host(Dm),
            float(np.asarray(self._to_host(cp))),
            float(np.asarray(self._to_host(lam))),
        )

    def eval_node_thermo_kinetics_into(self, T: float, Y: np.ndarray,
                                       omega_out: np.ndarray, hk_out: np.ndarray):
        """Evaluate nodal thermo and kinetics without transport."""
        T = float(T)
        P = self.problem.P
        Y = self._safe_Y(Y)

        rho = self.thermo.density(T, P, Y)
        cp, hk_vals, g_RT = self.thermo.cp_mass_hk_g_RT(T, Y)
        self._copy_to_out(hk_out, hk_vals)

        C = rho * Y * self.invW_dev
        wdot = self.kinetics.net_production_rates(T, C, g_RT)
        omega_mass = wdot * self.W_dev
        self._copy_to_out(omega_out, omega_mass)

        return (
            float(np.asarray(self._to_host(rho))),
            float(np.asarray(self._to_host(cp))),
        )

    # ------------------------------------------------------------------
    #  eval_node  (returns everything)
    # ------------------------------------------------------------------
    def eval_node(self, T: float, Y: np.ndarray):
        """Returns (rho, Dm, omega_mass, cp, lam, hk)."""
        omega = np.empty(self.n_species, dtype=float)
        hk = np.empty(self.n_species, dtype=float)
        rho, Dm, cp, lam = self.eval_node_into(T, Y, omega, hk)
        return rho, Dm, omega, cp, lam, hk

    # ------------------------------------------------------------------
    #  eval_midpoint  (face between j and j+1)
    # ------------------------------------------------------------------
    def eval_midpoint(self, T_left: float, T_right: float,
                      Y_left: np.ndarray, Y_right: np.ndarray):
        """Properties at the face j+½ (arithmetic average T, Y)."""
        Tm = 0.5 * (float(T_left) + float(T_right))
        Ym = 0.5 * (self.xp.asarray(Y_left) + self.xp.asarray(Y_right))
        Ym = self._safe_Y(Ym)
        P = self.problem.P

        rho = self.thermo.density(Tm, P, Ym)
        Dm_arr = self.transport.mix_diff_coeffs(Tm, P, self._Y_to_X(Ym))
        Wmix = self.thermo.mean_molecular_weight(Ym, self.invW_dev)
        return (
            float(np.asarray(self._to_host(rho))),
            self._to_host(Dm_arr),
            float(np.asarray(self._to_host(Wmix))),
        )

    def eval_midpoint_full_transport(self, T_left: float, T_right: float,
                                     Y_left: np.ndarray, Y_right: np.ndarray):
        """Returns (rho, D_mix, lambda, W_mix) at the face."""
        Tm = 0.5 * (float(T_left) + float(T_right))
        Ym = 0.5 * (self.xp.asarray(Y_left) + self.xp.asarray(Y_right))
        Ym = self._safe_Y(Ym)
        P = self.problem.P

        rho = self.thermo.density(Tm, P, Ym)
        cp_R = self.thermo.cp_R(Tm)
        X = self._Y_to_X(Ym)
        lam = self.transport.thermal_conductivity(Tm, X, cp_R)
        Dm  = self.transport.mix_diff_coeffs(Tm, P, X)
        Wmix = self.thermo.mean_molecular_weight(Ym, self.invW_dev)
        return (
            float(np.asarray(self._to_host(rho))),
            self._to_host(Dm),
            float(np.asarray(self._to_host(lam))),
            float(np.asarray(self._to_host(Wmix))),
        )

    def eval_midpoint_full(self, T_left: float, T_right: float,
                           Y_left: np.ndarray, Y_right: np.ndarray):
        """Full properties at face j+½."""
        Tm = 0.5 * (float(T_left) + float(T_right))
        Ym = 0.5 * (self.xp.asarray(Y_left, dtype=float)
                     + self.xp.asarray(Y_right, dtype=float))
        return self.eval_node(Tm, Ym)

    # ------------------------------------------------------------------
    #  density
    # ------------------------------------------------------------------
    def density(self, T: float, Y: np.ndarray) -> float:
        """ρ = P · W_mix / (R · T)."""
        Y = self._safe_Y(Y)
        rho = self.thermo.density(float(T), self.problem.P, Y)
        return float(np.asarray(self._to_host(rho)))
