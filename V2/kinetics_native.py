"""
kinetics_native.py – Chemical kinetics for ideal gas mixtures.

Supports:
  - Elementary Arrhenius reactions
  - Three-body reactions with efficiency factors
  - Pressure-dependent falloff reactions (Lindemann and Troe)

Evaluates forward/reverse rate constants and net production rates
for all species simultaneously.

NO Cantera dependency.  GPU-ready (pure NumPy).
"""
from __future__ import annotations
import math
import numpy as np
from mechanism_data import MechanismData, ReactionData, R_UNIV

# R_UNIV = 8314.46261815324 J/(kmol·K)


class NativeKinetics:
    """Evaluate net production rates ẇ_k [kmol/(m³·s)] for all species."""

    def __init__(self, mech: MechanismData, xp=None):
        import numpy as np
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

    # ------------------------------------------------------------------
    #  Pre-extract flat arrays from ReactionData list
    # ------------------------------------------------------------------
    def _build_reaction_arrays(self):
        """Pack per-reaction data into contiguous arrays."""
        nr = self.n_rxn
        nsp = self.n_sp
        import numpy as np # Use local numpy to extract lists, then cast to self.xp
        
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

        # Efficiencies matrix (n_rxn, n_sp) — 1.0 default for non-3body
        eff = np.ones((nr, nsp))
        for j, r in enumerate(self.mech.reactions):
            if r.efficiencies is not None:
                eff[j, :] = r.efficiencies
        self.eff = self.xp.asarray(eff)

    # ------------------------------------------------------------------
    #  Arrhenius rate constant
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  Grid Vectorization Helpers
    # ------------------------------------------------------------------
    def _is_grid(self, T):
        return self.xp.asarray(T).ndim > 0

    # ------------------------------------------------------------------
    #  Arrhenius rate constant
    # ------------------------------------------------------------------
    @staticmethod
    def _arrhenius(A: float, b: float, Ea: float, T: float) -> float:
        """k = A · T^b · exp(-Ea / (R·T))."""
        import math
        return A * T**b * math.exp(-Ea / (R_UNIV * T))

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
        # Δg_RT = Σ ν_net_k · g_RT_k  for each reaction
        # self.nu_net.T is (n_rxn, n_sp). g_RT is (n_sp,) or (n_sp, N).
        delta_g_RT = self.nu_net.T @ g_RT   # (n_rxn,) or (n_rxn, N)

        # Δν = sum of net stoich for each reaction
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
        Compute ẇ_k [kmol/(m³·s)] for each species.

        Parameters
        ----------
        T : temperature [K] (scalar or 1D array)
        C : species concentrations [kmol/m³], shape (n_sp,) or (n_sp, N)
        g_RT : Gibbs g_k / (R·T), shape (n_sp,) or (n_sp, N)

        Returns
        -------
        wdot : net production rates [kmol/(m³·s)], shape (n_sp,) or (n_sp, N)
        """
        Csafe = self.xp.maximum(C, 0.0)
        is_grid = self._is_grid(T)

        # ── Forward rate constants ─────────────────────────────────────
        kf = self._arrhenius_vec(self.A_hi, self.b_hi, self.Ea_hi, T)

        # ── Reverse rate constants via equilibrium ─────────────────────
        Kc = self.equilibrium_constants(g_RT, T)
        
        rev_mask = self.is_reversible[:, None] if is_grid else self.is_reversible
        kr = self.xp.where(rev_mask, kf / self.xp.maximum(Kc, 1e-300), 0.0)

        # ── Forward / reverse rates of progress ────────────────────────
        logC = self.xp.log(self.xp.maximum(Csafe, 1e-300))
        
        Rf = self.xp.exp(self.nu_r.T @ logC) * kf
        Rr = self.xp.where(rev_mask, self.xp.exp(self.nu_p.T @ logC) * kr, 0.0)

        # ── Three-body enhancement ─────────────────────────────────────
        M = self.third_body_conc(Csafe)

        tb_mask = self.is_three_body[:, None] if is_grid else self.is_three_body
        Rf = self.xp.where(tb_mask, Rf * M, Rf)
        Rr = self.xp.where(tb_mask, Rr * M, Rr)

        # ── Falloff ───────────────────────────────────────────────────
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

        # ── Net rate of progress ───────────────────────────────────────
        q = Rf - Rr

        # ── Species production rates ───────────────────────────────────
        wdot = self.nu_net @ q   # (n_sp, n_rxn) @ (n_rxn, N) -> (n_sp, N)

        return wdot
