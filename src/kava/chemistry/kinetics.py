"""
kinetics_native.py - Chemical kinetics for ideal gas mixtures.

Supports:
  - Elementary Arrhenius reactions
  - Three-body reactions with efficiency factors
  - Pressure-dependent falloff reactions (Lindemann and Troe)

Evaluates forward/reverse rate constants and net production rates
for all species simultaneously.

NO Cantera dependency. CPU NumPy/Numba implementation.
"""
from __future__ import annotations
import math
import numpy as np
from kava.chemistry.mechanism import MechanismData, R_UNIV

# R_UNIV = 8314.46261815324 J/(kmol*K)


try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    njit = None


def _negative_mass_action_factor(C, indices, orders, count):
    """Continuation outside the positive simplex, not a physical rate model.

    For elementary molecularity <= 3, retain one negative concentration
    factor (the restoring term), but suppress two or more. General orders
    with a negative participant are suppressed; no fractional power of a
    negative value is evaluated. Matches Cantera 3.2 StoichManager C1/C2/C3
    and C_AnyN conventions. Positive-state log-product arithmetic is unchanged.
    """
    negative_order = 0.0
    total_order = 0.0
    elementary = True
    for i in range(count):
        order = orders[i]
        total_order += order
        elementary = elementary and order >= 0.0 and order == math.floor(order)
        if C[indices[i]] < 0.0 and order != 0.0:
            negative_order += order
    if negative_order == 0.0:
        return 1.0
    if elementary and total_order <= 3.0 and negative_order == 1.0:
        return -1.0
    return 0.0


_negative_mass_action_factor_python = _negative_mass_action_factor
if njit is not None:
    _negative_mass_action_factor = njit(cache=True)(_negative_mass_action_factor)


def _make_mass_action_plan(indices, orders, counts):
    """Classify molecularity once; -1 retains the general-order log product."""
    plan = np.full((len(counts), 4), -1, dtype=np.int64)
    for r, count in enumerate(counts):
        nu = orders[r, :count]
        total = float(nu.sum())
        if total <= 3 and np.all(nu >= 0) and np.all(nu == np.floor(nu)):
            participants = np.repeat(indices[r, :count], nu.astype(np.int64))
            plan[r, 0] = len(participants)
            plan[r, 1:1+len(participants)] = participants
    return plan


def _mass_action_product(C, logC, indices, orders, count, plan, has_negative):
    """Small integer products without reaction-wise exp/log accumulation.

    Keep the original tiny-concentration floor and signed trial-state extension.
    General orders use the unchanged log product. No fastmath is enabled.
    """
    degree = plan[0]
    if degree < 0:
        exponent = 0.0
        for i in range(count):
            exponent += orders[i] * logC[indices[i]]
        product = math.exp(exponent)
        if has_negative:
            product *= _negative_mass_action_factor(C, indices, orders, count)
        return product
    product = 1.0
    negative = 0
    has_large = False
    for i in range(degree):
        value = C[plan[i+1]]
        negative += value < 0.0
        has_large = has_large or abs(value) > 1.0
        product *= max(abs(value), 1.0e-300)
    if negative > 1:
        return 0.0
    # Avoid losing a representable final product to intermediate overflow or
    # underflow for extreme inputs. Ordinary flame concentrations stay on the
    # multiplication path; the guard depends on arithmetic, not pressure.
    if not math.isfinite(product) or (product == 0.0 and has_large):
        exponent = 0.0
        for i in range(count):
            exponent += orders[i] * logC[indices[i]]
        product = math.exp(exponent)
    return -product if negative == 1 else product


_mass_action_product_python = _mass_action_product
if njit is not None:
    _mass_action_product = njit(cache=True, inline='always')(_mass_action_product)


if njit is not None:
    @njit(cache=True)
    def _net_production_rates_numba_core(
        T_arr, C, g_RT, A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
        is_three_body, is_falloff, is_reversible, has_troe,
        troe_A, troe_T3, troe_T1, troe_T2, nu_r, nu_p, nu_net, eff,
        delta_nu,
    ):
        n_sp = C.shape[0]
        n_pts = C.shape[1]
        n_rxn = A_hi.shape[0]
        out = np.zeros((n_sp, n_pts), dtype=np.float64)
        logC = np.empty(n_sp, dtype=np.float64)
        all_indices = np.arange(n_sp)

        for m in range(n_pts):
            T = T_arr[m]
            c_factor = 101325.0 / (R_UNIV * T)
            inv_RT = 1.0 / (R_UNIV * T)
            logT = math.log(T)
            has_negative = False

            for k in range(n_sp):
                c = C[k, m]
                has_negative = has_negative or c < 0.0
                c = abs(c)
                if c < 1.0e-300:
                    c = 1.0e-300
                logC[k] = math.log(c)

            for r in range(n_rxn):
                # T**b * exp(-Ea / RT) as one exponential. This is algebraically
                # identical and avoids the generic floating-point power routine.
                kf = A_hi[r] * math.exp(b_hi[r] * logT - Ea_hi[r] * inv_RT)

                delta_g = 0.0
                rf_exp = 0.0
                rr_exp = 0.0
                M = 0.0
                for k in range(n_sp):
                    delta_g += nu_net[k, r] * g_RT[k, m]
                    rf_exp += nu_r[k, r] * logC[k]
                    rr_exp += nu_p[k, r] * logC[k]
                    ck = C[k, m]
                    M += eff[r, k] * ck

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

                if has_negative:
                    Rf *= _negative_mass_action_factor(C[:, m], all_indices, nu_r[:, r], n_sp)
                    Rr *= _negative_mass_action_factor(C[:, m], all_indices, nu_p[:, r], n_sp)

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
                for k in range(n_sp):
                    out[k, m] += nu_net[k, r] * q

        return out


    @njit(cache=True)
    def _net_production_rates_sparse_numba_core(
        T_arr, C, g_RT, A_hi, b_hi, Ea_hi, A_lo, b_lo, Ea_lo,
        is_three_body, is_falloff, is_reversible, has_troe,
        troe_A, troe_T3, troe_T1, troe_T2,
        r_idx, r_nu, r_count, p_idx, p_nu, p_count,
        net_idx, net_nu, net_count, eff_idx, eff_delta, eff_count,
        delta_nu,
    ):
        n_sp = C.shape[0]
        n_pts = C.shape[1]
        n_rxn = A_hi.shape[0]
        out = np.zeros((n_sp, n_pts), dtype=np.float64)
        logC = np.empty(n_sp, dtype=np.float64)

        for m in range(n_pts):
            T = T_arr[m]
            c_factor = 101325.0 / (R_UNIV * T)
            inv_RT = 1.0 / (R_UNIV * T)
            logT = math.log(T)
            c_total = 0.0
            has_negative = False

            for k in range(n_sp):
                c = C[k, m]
                c_total += c
                has_negative = has_negative or c < 0.0
                c = abs(c)
                if c < 1.0e-300:
                    c = 1.0e-300
                logC[k] = math.log(c)

            for r in range(n_rxn):
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
                    delta_g += net_nu[r, ii] * g_RT[k, m]

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

                if has_negative:
                    Rf *= _negative_mass_action_factor(C[:, m], r_idx[r], r_nu[r], r_count[r])
                    Rr *= _negative_mass_action_factor(C[:, m], p_idx[r], p_nu[r], p_count[r])

                M = c_total
                if is_three_body[r] or is_falloff[r]:
                    for ii in range(eff_count[r]):
                        k = eff_idx[r, ii]
                        ck = C[k, m]
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
                    out[k, m] += net_nu[r, ii] * q

        return out


else:
    _net_production_rates_numba_core = None
    _net_production_rates_sparse_numba_core = None


class NativeKinetics:
    """Evaluate net production rates [kmol/(m^3*s)] for all species."""

    def __init__(self, mech: MechanismData, xp=None):
        self.xp = xp if xp is not None else np
        
        self.mech = mech
        self.n_sp  = mech.n_species
        self.n_rxn = mech.n_reactions
        self.W     = self.xp.asarray(mech.molecular_weights)
        self.invW  = self.xp.asarray(mech.inv_molecular_weights)

        # Dense stoichiometry
        self.nu_r   = self.xp.asarray(mech.nu_reactants)   # (n_sp, n_rxn)
        self.nu_p   = self.xp.asarray(mech.nu_products)     # (n_sp, n_rxn)
        self.nu_net = self.xp.asarray(mech.nu_net)           # (n_sp, n_rxn)

        # Pre-extract reaction parameters into flat arrays for vectorisation
        self._build_reaction_arrays()
        self._numba_available = (
            _net_production_rates_numba_core is not None
            and getattr(self.xp, "__name__", "") == "numpy"
        )
        self._sparse_numba_available = (
            _net_production_rates_sparse_numba_core is not None
            and getattr(self.xp, "__name__", "") == "numpy"
        )
        self._use_numba = False
        self._use_sparse_numba = False
        if self._numba_available or self._sparse_numba_available:
            self._prepare_numba_arrays()

    # ------------------------------------------------------------------
    #  Pre-extract flat arrays from ReactionData list
    # ------------------------------------------------------------------
    def _build_reaction_arrays(self):
        """Pack per-reaction data into contiguous arrays."""
        nr = self.n_rxn
        nsp = self.n_sp
        # Arrhenius high-P
        self.A_hi   = self.xp.asarray([r.A for r in self.mech.reactions])
        self.b_hi   = self.xp.asarray([r.b for r in self.mech.reactions])
        self.Ea_hi  = self.xp.asarray([r.Ea for r in self.mech.reactions])

        # Reaction types
        self.is_elementary = self.xp.asarray([r.rtype == "elementary" for r in self.mech.reactions])
        self.is_three_body = self.xp.asarray([r.rtype == "three-body" for r in self.mech.reactions])
        self.is_falloff    = self.xp.asarray([r.rtype == "falloff" for r in self.mech.reactions])
        self.is_reversible = self.xp.asarray([r.reversible for r in self.mech.reactions])

        # Low-P Arrhenius (for falloff)
        self.A_lo  = self.xp.asarray([r.A_low for r in self.mech.reactions])
        self.b_lo  = self.xp.asarray([r.b_low for r in self.mech.reactions])
        self.Ea_lo = self.xp.asarray([r.Ea_low for r in self.mech.reactions])

        # Troe
        self.has_troe = self.xp.asarray([r.has_troe for r in self.mech.reactions])
        self.troe_A  = self.xp.asarray([r.troe_A for r in self.mech.reactions])
        self.troe_T3 = self.xp.asarray([r.troe_T3 for r in self.mech.reactions])
        self.troe_T1 = self.xp.asarray([r.troe_T1 for r in self.mech.reactions])
        self.troe_T2 = self.xp.asarray([r.troe_T2 for r in self.mech.reactions])

        # Efficiencies matrix (n_rxn, n_sp); 1.0 is the non-3body default.
        eff = np.ones((nr, nsp))
        for j, r in enumerate(self.mech.reactions):
            if r.efficiencies is not None:
                eff[j, :] = r.efficiencies
        self.eff = self.xp.asarray(eff)

    def _prepare_numba_arrays(self):
        """Keep CPU-contiguous mechanism arrays for the optional Numba path."""
        self._nb_A_hi = np.asarray(self.A_hi, dtype=np.float64)
        self._nb_b_hi = np.asarray(self.b_hi, dtype=np.float64)
        self._nb_Ea_hi = np.asarray(self.Ea_hi, dtype=np.float64)
        self._nb_A_lo = np.asarray(self.A_lo, dtype=np.float64)
        self._nb_b_lo = np.asarray(self.b_lo, dtype=np.float64)
        self._nb_Ea_lo = np.asarray(self.Ea_lo, dtype=np.float64)
        self._nb_is_three_body = np.asarray(self.is_three_body, dtype=np.bool_)
        self._nb_is_falloff = np.asarray(self.is_falloff, dtype=np.bool_)
        self._nb_is_reversible = np.asarray(self.is_reversible, dtype=np.bool_)
        self._nb_has_troe = np.asarray(self.has_troe, dtype=np.bool_)
        self._nb_troe_A = np.asarray(self.troe_A, dtype=np.float64)
        self._nb_troe_T3 = np.asarray(self.troe_T3, dtype=np.float64)
        self._nb_troe_T1 = np.asarray(self.troe_T1, dtype=np.float64)
        self._nb_troe_T2 = np.asarray(self.troe_T2, dtype=np.float64)
        self._nb_nu_r = np.ascontiguousarray(self.nu_r, dtype=np.float64)
        self._nb_nu_p = np.ascontiguousarray(self.nu_p, dtype=np.float64)
        self._nb_nu_net = np.ascontiguousarray(self.nu_net, dtype=np.float64)
        self._nb_eff = np.ascontiguousarray(self.eff, dtype=np.float64)
        self._nb_delta_nu = np.asarray(self.nu_net.sum(axis=0), dtype=np.float64)

        nu_r = np.asarray(self.nu_r, dtype=np.float64)
        nu_p = np.asarray(self.nu_p, dtype=np.float64)
        nu_net = np.asarray(self.nu_net, dtype=np.float64)
        eff = np.asarray(self.eff, dtype=np.float64)
        nr = self.n_rxn

        r_lists = [np.nonzero(nu_r[:, j])[0] for j in range(nr)]
        p_lists = [np.nonzero(nu_p[:, j])[0] for j in range(nr)]
        net_lists = [np.nonzero(nu_net[:, j])[0] for j in range(nr)]
        eff_lists = [
            np.nonzero(np.abs(eff[j, :] - 1.0) > 0.0)[0]
            if bool(self._nb_is_three_body[j] or self._nb_is_falloff[j])
            else np.empty(0, dtype=np.int64)
            for j in range(nr)
        ]

        max_r = max(1, max(len(v) for v in r_lists))
        max_p = max(1, max(len(v) for v in p_lists))
        max_net = max(1, max(len(v) for v in net_lists))
        max_eff = max(1, max(len(v) for v in eff_lists))

        self._sp_r_idx = np.zeros((nr, max_r), dtype=np.int64)
        self._sp_r_nu = np.zeros((nr, max_r), dtype=np.float64)
        self._sp_r_count = np.zeros(nr, dtype=np.int64)
        self._sp_p_idx = np.zeros((nr, max_p), dtype=np.int64)
        self._sp_p_nu = np.zeros((nr, max_p), dtype=np.float64)
        self._sp_p_count = np.zeros(nr, dtype=np.int64)
        self._sp_net_idx = np.zeros((nr, max_net), dtype=np.int64)
        self._sp_net_nu = np.zeros((nr, max_net), dtype=np.float64)
        self._sp_net_count = np.zeros(nr, dtype=np.int64)
        self._sp_eff_idx = np.zeros((nr, max_eff), dtype=np.int64)
        self._sp_eff_delta = np.zeros((nr, max_eff), dtype=np.float64)
        self._sp_eff_count = np.zeros(nr, dtype=np.int64)

        for j in range(nr):
            idx = r_lists[j]
            self._sp_r_count[j] = len(idx)
            self._sp_r_idx[j, :len(idx)] = idx
            self._sp_r_nu[j, :len(idx)] = nu_r[idx, j]

            idx = p_lists[j]
            self._sp_p_count[j] = len(idx)
            self._sp_p_idx[j, :len(idx)] = idx
            self._sp_p_nu[j, :len(idx)] = nu_p[idx, j]

            idx = net_lists[j]
            self._sp_net_count[j] = len(idx)
            self._sp_net_idx[j, :len(idx)] = idx
            self._sp_net_nu[j, :len(idx)] = nu_net[idx, j]

            idx = eff_lists[j]
            self._sp_eff_count[j] = len(idx)
            self._sp_eff_idx[j, :len(idx)] = idx
            self._sp_eff_delta[j, :len(idx)] = eff[j, idx] - 1.0

        self._sp_r_plan = _make_mass_action_plan(self._sp_r_idx, self._sp_r_nu, self._sp_r_count)
        self._sp_p_plan = _make_mass_action_plan(self._sp_p_idx, self._sp_p_nu, self._sp_p_count)

    # ------------------------------------------------------------------
    #  Grid Vectorization Helpers
    # ------------------------------------------------------------------
    def _is_grid(self, T):
        return self.xp.asarray(T).ndim > 0

    # ------------------------------------------------------------------
    #  Arrhenius rate constant
    # ------------------------------------------------------------------
    def _arrhenius_vec(self, A: np.ndarray, b: np.ndarray, Ea: np.ndarray, T) -> np.ndarray:
        """Vectorised Arrhenius over reactions."""
        if self._is_grid(T):
            T_arr = self.xp.asarray(T, dtype=float)[None, :]
            return A[:, None] * T_arr**b[:, None] * self.xp.exp(-Ea[:, None] / (R_UNIV * T_arr))
        else:
            return A * T**b * self.xp.exp(-Ea / (R_UNIV * T))

    # ------------------------------------------------------------------
    #  Equilibrium constant via Gibbs (reverse rates)
    # ------------------------------------------------------------------
    def equilibrium_constants(self, g_RT: np.ndarray, T) -> np.ndarray:
        """
        Kc_j for each reaction j.
        """
        # delta_g/RT = sum(nu_net_k * g_RT_k) for each reaction.
        # self.nu_net.T is (n_rxn, n_sp). g_RT is (n_sp,) or (n_sp, N).
        delta_g_RT = self.nu_net.T @ g_RT   # (n_rxn,) or (n_rxn, N)

        # delta_nu = sum of net stoichiometric coefficients per reaction.
        delta_nu = self.nu_net.sum(axis=0)   # (n_rxn,)

        # Clip to avoid overflow
        delta_g_clipped = self.xp.clip(delta_g_RT, -500.0, 500.0)
        Kp = self.xp.exp(-delta_g_clipped)

        P_ref = 101325.0
        if self._is_grid(T):
            T_arr = self.xp.asarray(T, dtype=float)[None, :]
            c_factor = P_ref / (R_UNIV * T_arr)
            Kc = Kp * c_factor ** delta_nu[:, None]
        else:
            c_factor = P_ref / (R_UNIV * T)
            Kc = Kp * c_factor ** delta_nu

        return Kc

    # ------------------------------------------------------------------
    #  Third-body concentration [M]
    # ------------------------------------------------------------------
    def third_body_conc(self, C: np.ndarray) -> np.ndarray:
        """[M]_j for each reaction j. Shape (n_rxn,) or (n_rxn, N)."""
        return self.eff @ C   # (n_rxn, n_sp) @ (n_sp, N) -> (n_rxn, N)

    # ------------------------------------------------------------------
    #  Troe falloff factor
    # ------------------------------------------------------------------
    def _troe_F(self, T, Pr: np.ndarray, j_mask: np.ndarray) -> np.ndarray:
        """Troe falloff factor F for reactions where has_troe is True."""
        A  = self.troe_A[j_mask]
        T3 = self.troe_T3[j_mask]
        T1 = self.troe_T1[j_mask]
        T2 = self.troe_T2[j_mask]

        if self._is_grid(T):
            T_arr = self.xp.asarray(T, dtype=float)[None, :]
            Pr_val = self.xp.maximum(Pr[j_mask, :], 1e-300)
            
            Fcent = ((1 - A[:, None]) * self.xp.exp(-T_arr / self.xp.maximum(T3[:, None], 1e-300))
                     + A[:, None] * self.xp.exp(-T_arr / self.xp.maximum(T1[:, None], 1e-300))
                     + self.xp.exp(-T2[:, None] / self.xp.maximum(T_arr, 1e-300)))
            Fcent = self.xp.maximum(Fcent, 1e-300)
            
            logFcent = self.xp.log10(Fcent)
            logPr = self.xp.log10(Pr_val)

            c = -0.4 - 0.67 * logFcent
            n_troe = 0.75 - 1.27 * logFcent
            d = 0.14
            f1 = (logPr + c) / (n_troe - d * (logPr + c))
            
            F_out = self.xp.ones_like(Pr)
            F_out[j_mask, :] = 10.0 ** (logFcent / (1.0 + f1**2))
            return F_out
        else:
            Pr_val = self.xp.maximum(Pr[j_mask], 1e-300)
            Fcent = ((1 - A) * self.xp.exp(-T / self.xp.maximum(T3, 1e-300))
                     + A * self.xp.exp(-T / self.xp.maximum(T1, 1e-300))
                     + self.xp.exp(-T2 / max(T, 1e-300)))
            Fcent = self.xp.maximum(Fcent, 1e-300)

            logFcent = self.xp.log10(Fcent)
            logPr = self.xp.log10(Pr_val)

            c = -0.4 - 0.67 * logFcent
            n_troe = 0.75 - 1.27 * logFcent
            d = 0.14
            f1 = (logPr + c) / (n_troe - d * (logPr + c))
            
            F_out = self.xp.ones(len(Pr))
            F_out[j_mask] = 10.0 ** (logFcent / (1.0 + f1**2))
            return F_out

    # ------------------------------------------------------------------
    #  Net production rates
    # ------------------------------------------------------------------
    def net_production_rates(self, T, C: np.ndarray, g_RT: np.ndarray) -> np.ndarray:
        """
        Compute net production rates [kmol/(m^3*s)] for each species.

        Parameters
        ----------
        T : temperature [K] (scalar or 1D array)
        C : species concentrations [kmol/m^3], shape (n_sp,) or (n_sp, N)
        g_RT : Gibbs g_k / (R*T), shape (n_sp,) or (n_sp, N)

        Returns
        -------
        wdot : net production rates [kmol/(m^3*s)], shape (n_sp,) or (n_sp, N)
        """
        if self._use_sparse_numba:
            C_arr = np.asarray(C, dtype=np.float64)
            g_arr = np.asarray(g_RT, dtype=np.float64)
            scalar = C_arr.ndim == 1
            if scalar:
                C_work = np.ascontiguousarray(C_arr[:, None])
                g_work = np.ascontiguousarray(g_arr[:, None])
                T_work = np.asarray([float(T)], dtype=np.float64)
            else:
                C_work = np.ascontiguousarray(C_arr)
                g_work = np.ascontiguousarray(g_arr)
                T_work = np.asarray(T, dtype=np.float64).reshape(-1)
                if T_work.size == 1 and C_work.shape[1] != 1:
                    T_work = np.full(C_work.shape[1], float(T_work[0]), dtype=np.float64)

            wdot = _net_production_rates_sparse_numba_core(
                T_work,
                C_work,
                g_work,
                self._nb_A_hi,
                self._nb_b_hi,
                self._nb_Ea_hi,
                self._nb_A_lo,
                self._nb_b_lo,
                self._nb_Ea_lo,
                self._nb_is_three_body,
                self._nb_is_falloff,
                self._nb_is_reversible,
                self._nb_has_troe,
                self._nb_troe_A,
                self._nb_troe_T3,
                self._nb_troe_T1,
                self._nb_troe_T2,
                self._sp_r_idx,
                self._sp_r_nu,
                self._sp_r_count,
                self._sp_p_idx,
                self._sp_p_nu,
                self._sp_p_count,
                self._sp_net_idx,
                self._sp_net_nu,
                self._sp_net_count,
                self._sp_eff_idx,
                self._sp_eff_delta,
                self._sp_eff_count,
                self._nb_delta_nu,
            )
            return wdot[:, 0] if scalar else wdot

        if self._use_numba:
            C_arr = np.asarray(C, dtype=np.float64)
            g_arr = np.asarray(g_RT, dtype=np.float64)
            scalar = C_arr.ndim == 1
            if scalar:
                C_work = np.ascontiguousarray(C_arr[:, None])
                g_work = np.ascontiguousarray(g_arr[:, None])
                T_work = np.asarray([float(T)], dtype=np.float64)
            else:
                C_work = np.ascontiguousarray(C_arr)
                g_work = np.ascontiguousarray(g_arr)
                T_work = np.asarray(T, dtype=np.float64).reshape(-1)
                if T_work.size == 1 and C_work.shape[1] != 1:
                    T_work = np.full(C_work.shape[1], float(T_work[0]), dtype=np.float64)

            wdot = _net_production_rates_numba_core(
                T_work,
                C_work,
                g_work,
                self._nb_A_hi,
                self._nb_b_hi,
                self._nb_Ea_hi,
                self._nb_A_lo,
                self._nb_b_lo,
                self._nb_Ea_lo,
                self._nb_is_three_body,
                self._nb_is_falloff,
                self._nb_is_reversible,
                self._nb_has_troe,
                self._nb_troe_A,
                self._nb_troe_T3,
                self._nb_troe_T1,
                self._nb_troe_T2,
                self._nb_nu_r,
                self._nb_nu_p,
                self._nb_nu_net,
                self._nb_eff,
                self._nb_delta_nu,
            )
            return wdot[:, 0] if scalar else wdot

        is_grid = self._is_grid(T)

        # Forward rate constants
        kf = self._arrhenius_vec(self.A_hi, self.b_hi, self.Ea_hi, T)

        # Reverse rate constants via equilibrium
        Kc = self.equilibrium_constants(g_RT, T)
        
        rev_mask = self.is_reversible[:, None] if is_grid else self.is_reversible
        kr = self.xp.where(rev_mask, kf / self.xp.maximum(Kc, 1e-300), 0.0)

        # Forward and reverse rates of progress
        logC = self.xp.log(self.xp.maximum(self.xp.abs(C), 1e-300))
        
        Rf = self.xp.exp(self.nu_r.T @ logC) * kf
        Rr = self.xp.where(rev_mask, self.xp.exp(self.nu_p.T @ logC) * kr, 0.0)

        if self.xp.any(C < 0.0):
            c_work = C if C.ndim == 2 else C[:, None]
            rf_work = Rf if Rf.ndim == 2 else Rf[:, None]
            rr_work = Rr if Rr.ndim == 2 else Rr[:, None]
            all_indices = range(self.n_sp)
            for m in range(c_work.shape[1]):
                if not self.xp.any(c_work[:, m] < 0.0):
                    continue
                for r in range(self.nu_r.shape[1]):
                    rf_work[r, m] *= _negative_mass_action_factor_python(
                        c_work[:, m], all_indices, self.nu_r[:, r], self.n_sp)
                    rr_work[r, m] *= _negative_mass_action_factor_python(
                        c_work[:, m], all_indices, self.nu_p[:, r], self.n_sp)

        # Three-body enhancement
        M = self.third_body_conc(C)

        tb_mask = self.is_three_body[:, None] if is_grid else self.is_three_body
        Rf = self.xp.where(tb_mask, Rf * M, Rf)
        Rr = self.xp.where(tb_mask, Rr * M, Rr)

        # Falloff
        fo_mask = self.is_falloff[:, None] if is_grid else self.is_falloff
        if self.xp.any(self.is_falloff):
            k0 = self._arrhenius_vec(self.A_lo, self.b_lo, self.Ea_lo, T)
            kinf = kf.copy()

            Pr = self.xp.where(fo_mask,
                          k0 * M / self.xp.maximum(kinf, 1e-300),
                          0.0)

            F_lind = Pr / (1.0 + Pr)

            troe_mask = self.is_falloff & self.has_troe
            if self.xp.any(troe_mask):
                F = self._troe_F(T, Pr, troe_mask)
            else:
                F = self.xp.ones_like(Pr)

            kf_falloff = kinf * F_lind * F
            Rf = self.xp.where(fo_mask, kf_falloff * (Rf / self.xp.maximum(kf, 1e-300)), Rf)

            rev_fo = self.is_reversible & self.is_falloff
            rev_fo_mask = rev_fo[:, None] if is_grid else rev_fo
            if self.xp.any(rev_fo):
                kr_falloff = kf_falloff / self.xp.maximum(Kc, 1e-300)
                Rr = self.xp.where(rev_fo_mask, kr_falloff * (Rr / self.xp.maximum(kr, 1e-300)), Rr)

        # Net rate of progress
        q = Rf - Rr

        # Species production rates
        wdot = self.nu_net @ q   # (n_sp, n_rxn) @ (n_rxn, N) -> (n_sp, N)

        return wdot
