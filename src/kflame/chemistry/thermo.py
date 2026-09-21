"""
thermo_native.py – NASA-7 polynomial thermodynamics for ideal gas.

Evaluates cp, h, s, and derived quantities for all species simultaneously
using vectorized NumPy on CPU.

NO Cantera dependency.
"""
from __future__ import annotations
import numpy as np
from kflame.chemistry.mechanism import MechanismData, R_UNIV

# R_UNIV = 8314.46261815324 J/(kmol·K)  – same as Cantera


class NativeThermo:
    """
    Evaluates NASA-7 thermodynamic polynomials for ideal gases.
    Supports single-point or vectorized evaluation based on `xp`.
    """

    def __init__(self, mech: MechanismData, xp=None):
        self.xp = xp if xp is not None else np
        
        self.n_sp = mech.n_species
        self.W = self.xp.asarray(mech.molecular_weights)
        self.invW = self.xp.asarray(mech.inv_molecular_weights)

        # NASA-7 coefficients
        self._lo = self.xp.asarray(mech.nasa_low)
        self._hi = self.xp.asarray(mech.nasa_high)
        self._Tmid = self.xp.asarray(mech.nasa_Tmid)

    # ------------------------------------------------------------------
    #  Grid Vectorization Helpers
    # ------------------------------------------------------------------
    def _to_1d(self, T):
        T_arr = self.xp.asarray(T, dtype=float)
        is_scalar = (T_arr.ndim == 0)
        if is_scalar:
            T_arr = T_arr[None]
        return T_arr, is_scalar

    def _get_coefs(self, T_arr: np.ndarray) -> np.ndarray:
        """Returns the correct 7 coefficients per species depending on T_arr (N,)."""
        is_high = T_arr[:, None] > self._Tmid[None, :]  # (N, n_sp)
        # returns (N, n_sp, 7)
        return self.xp.where(is_high[:, :, None], self._hi[None, :, :], self._lo[None, :, :])

    # ------------------------------------------------------------------
    #  Non-dimensional properties (per species)
    # ------------------------------------------------------------------
    def cp_R(self, T) -> np.ndarray:
        """cp_k / R = a0 + a1·T + a2·T² + a3·T³ + a4·T⁴"""
        T_arr, is_scalar = self._to_1d(T)
        c = self._get_coefs(T_arr)
        T_mat = T_arr[:, None]
        res = c[:,:,0] + T_mat * (c[:,:,1] + T_mat * (c[:,:,2] + T_mat * (c[:,:,3] + T_mat * c[:,:,4])))
        return res[0] if is_scalar else res.T  # (n_sp, N)

    def h_RT(self, T) -> np.ndarray:
        """h_k / (R·T) = a0 + a1·T/2 + a2·T²/3 + a3·T³/4 + a4·T⁴/5 + a5/T"""
        T_arr, is_scalar = self._to_1d(T)
        c = self._get_coefs(T_arr)
        T_mat = T_arr[:, None]
        res = c[:,:,0] + T_mat * (c[:,:,1]/2.0 + T_mat * (c[:,:,2]/3.0 + T_mat * (c[:,:,3]/4.0 + T_mat * c[:,:,4]/5.0))) + c[:,:,5]/T_mat
        return res[0] if is_scalar else res.T

    def s_R(self, T) -> np.ndarray:
        """s_k / R = a0·ln(T) + a1·T + a2·T²/2 + a3·T³/3 + a4·T⁴/4 + a6"""
        T_arr, is_scalar = self._to_1d(T)
        c = self._get_coefs(T_arr)
        T_mat = T_arr[:, None]
        res = c[:,:,0]*self.xp.log(T_mat) + T_mat * (c[:,:,1] + T_mat * (c[:,:,2]/2.0 + T_mat * (c[:,:,3]/3.0 + T_mat * c[:,:,4]/4.0))) + c[:,:,6]
        return res[0] if is_scalar else res.T

    def g_RT(self, T) -> np.ndarray:
        """g_k / (R·T) = h_k/(R·T) - s_k/R  for each species."""
        T_arr, is_scalar = self._to_1d(T)
        h = self.h_RT(T_arr)  # (n_sp, N)
        s = self.s_R(T_arr)   # (n_sp, N)
        res = h - s
        return res[:, 0] if is_scalar else res

    def cp_mass_hk_g_RT(self, T, Y: np.ndarray):
        """
        Fused NASA-7 evaluation for cp_mix, h_k and g_k/(R*T).
        """
        T_arr, is_scalar = self._to_1d(T)
        c = self._get_coefs(T_arr)
        T_mat = T_arr[:, None]
        logT = self.xp.log(T_mat)

        cp_R_ns = (
            c[:, :, 0]
            + T_mat * (
                c[:, :, 1]
                + T_mat * (
                    c[:, :, 2]
                    + T_mat * (c[:, :, 3] + T_mat * c[:, :, 4])
                )
            )
        )
        h_RT_ns = (
            c[:, :, 0]
            + T_mat * (
                c[:, :, 1] / 2.0
                + T_mat * (
                    c[:, :, 2] / 3.0
                    + T_mat * (c[:, :, 3] / 4.0 + T_mat * c[:, :, 4] / 5.0)
                )
            )
            + c[:, :, 5] / T_mat
        )
        s_R_ns = (
            c[:, :, 0] * logT
            + T_mat * (
                c[:, :, 1]
                + T_mat * (
                    c[:, :, 2] / 2.0
                    + T_mat * (c[:, :, 3] / 3.0 + T_mat * c[:, :, 4] / 4.0)
                )
            )
            + c[:, :, 6]
        )

        cp_R = cp_R_ns.T
        h_RT = h_RT_ns.T
        g_RT = (h_RT_ns - s_R_ns).T
        hk = h_RT * R_UNIV * T_arr[None, :]

        if Y.ndim == 1:
            cp_mass = self.xp.sum(Y * cp_R[:, 0] * R_UNIV * self.invW)
            return cp_mass, hk[:, 0], g_RT[:, 0]

        cp_mass = self.xp.sum(Y * cp_R * R_UNIV * self.invW[:, None], axis=0)
        return cp_mass, hk, g_RT

    # ------------------------------------------------------------------
    #  Dimensional quantities (per-species, molar basis)
    # ------------------------------------------------------------------
    def partial_molar_cp(self, T) -> np.ndarray:
        """cp_k  [J/(kmol·K)]."""
        return self.cp_R(T) * R_UNIV

    def partial_molar_enthalpies(self, T) -> np.ndarray:
        """h_k  [J/kmol]."""
        T_arr, is_scalar = self._to_1d(T)
        h = self.h_RT(T_arr) * R_UNIV * T_arr[None, :]  # (n_sp, N)
        return h[:, 0] if is_scalar else h

    def partial_molar_entropies(self, T) -> np.ndarray:
        """s_k  [J/(kmol·K)]."""
        return self.s_R(T) * R_UNIV

    def partial_molar_gibbs(self, T) -> np.ndarray:
        """g_k  [J/kmol]."""
        T_arr, is_scalar = self._to_1d(T)
        g = self.g_RT(T_arr) * R_UNIV * T_arr[None, :]
        return g[:, 0] if is_scalar else g

    # ------------------------------------------------------------------
    #  Mixture quantities
    # ------------------------------------------------------------------
    def mean_molecular_weight(self, Y: np.ndarray, invW=None) -> float | np.ndarray:
        """W_mix = 1 / Σ(Y_k / W_k)  [kg/kmol]. Y shape (n_sp,) or (n_sp, N)."""
        if invW is None:
            invW = self.invW
        if Y.ndim == 1:
            inv_Wmix = self.xp.sum(Y * invW)
            return 1.0 / self.xp.maximum(inv_Wmix, 1e-300)
        else:
            inv_Wmix = self.xp.sum(Y * invW[:, None], axis=0)
            return 1.0 / self.xp.maximum(inv_Wmix, 1e-300)

    def density(self, T, P: float, Y: np.ndarray) -> float | np.ndarray:
        """ρ = P · W_mix / (R · T)  [kg/m³]."""
        T_arr, is_scalar = self._to_1d(T)
        Wmix = self.mean_molecular_weight(Y, self.invW)  # (N,) or scalar
        res = P * Wmix / (R_UNIV * T_arr)
        return res[0] if is_scalar else res

    def Y_to_X(self, Y: np.ndarray) -> np.ndarray:
        """Mass fractions → mole fractions."""
        Wmix = self.mean_molecular_weight(Y, self.invW)
        if Y.ndim == 1:
            return Y * Wmix * self.invW
        else:
            return Y * Wmix[None, :] * self.invW[:, None]

    def X_to_Y(self, X: np.ndarray) -> np.ndarray:
        """Mole fractions → mass fractions."""
        XW = X * self.W
        if X.ndim == 1:
            return XW / XW.sum()
        else:
            return XW / XW.sum(axis=0)[None, :]

    def cp_mass(self, T, Y: np.ndarray) -> float | np.ndarray:
        """cp_mix  [J/(kg·K)] = Σ Y_k · cp_k / W_k  (mass-weighted)."""
        cp_k = self.cp_R(T) * R_UNIV  # (n_sp,) or (n_sp, N)
        if Y.ndim == 1:
            return self.xp.sum(Y * cp_k * self.invW)
        else:
            return self.xp.sum(Y * cp_k * self.invW[:, None], axis=0)

    def cp_mole(self, T, X: np.ndarray) -> float | np.ndarray:
        """cp_mix  [J/(kmol·K)] = Σ X_k · cp_k."""
        cp_k = self.cp_R(T) * R_UNIV
        if X.ndim == 1:
            return self.xp.sum(X * cp_k)
        else:
            return self.xp.sum(X * cp_k, axis=0)
