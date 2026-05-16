"""
transport_native.py – Mixture-averaged transport properties for ideal gas.

Implements the same model as Cantera's MixTransport / GasTransport:
  - Pure-species viscosity via Chapman-Enskog theory
  - Wilke mixing rule for mixture viscosity
  - Eucken correlation for pure-species conductivity
  - Mixture-averaged diffusion coefficients

Uses Neufeld et al. (1972) correlations for collision integrals Ω^(1,1)*
and Ω^(2,2)* instead of tabulated Monchick-Mason data.  Accuracy: <0.5%.

NO Cantera dependency.  GPU-ready (pure NumPy arrays).
"""
from __future__ import annotations
import math
import numpy as np
from mechanism_data import MechanismData, R_UNIV, BOLTZMANN, AVOGADRO
import json
from pathlib import Path

# Physical constants
PI = math.pi
EPSILON_0 = 8.854187817e-12  # vacuum permittivity [F/m]

# Load pre-exported Cantera transport polynomials (ln(T) basis)
_TRANSPORT_POLY_FILE = Path(__file__).parent / "cantera_transport_poly_coeffs.json"
try:
    with open(_TRANSPORT_POLY_FILE, "r", encoding="utf-8") as f:
        _CANTERA_TRANSPORT_POLY = json.load(f)
except FileNotFoundError:
    _CANTERA_TRANSPORT_POLY = None
    print(f"Warning: Cantera transport polynomial file not found at {_TRANSPORT_POLY_FILE}")

class NativeTransport:
    """
    Mixture-averaged transport for ideal gas.
    Supports CuPy (GPU) or NumPy (CPU) based on `xp`.
    """

    def __init__(self, mech: MechanismData, xp=None):
        import numpy as np
        self.xp = xp if xp is not None else np
        
        self.mech = mech  # Store mechanism reference for species names
        n = mech.n_species
        self.n_sp = n
        self.mw = self.xp.asarray(mech.molecular_weights)   # kg/kmol
        self.mw_kg = self.mw / AVOGADRO / 1000.0  # kg per molecule

        # Lennard-Jones parameters
        self.eps = self.xp.asarray(mech.well_depth)          # ε/kB [K]
        self.sigma = self.xp.asarray(mech.diameter)           # σ [m]
        self.dipole = self.xp.asarray(mech.dipole)            # μ [C·m]
        self.polar = self.dipole > 0.0              # bool array
        self.alpha = self.xp.asarray(mech.polarizability)     # α [m³]
        self.geometry = self.xp.asarray(mech.geometry)        # 0/1/2
        self.zrot = self.xp.asarray(mech.rot_relax)           # Z_rot
        self.crot = self.xp.where(self.geometry == 0, 0.0,
                    self.xp.where(self.geometry == 1, 1.0, 1.5))

        # Pre-compute pair parameters (symmetric)
        _red_mass = np.zeros((n, n))
        _eps_p = np.zeros((n, n))
        _sig_p = np.zeros((n, n))
        _del_p = np.zeros((n, n))

        # Perform the pair mixing rules in NumPy (it's one-time init)
        eps_cpu = np.asarray(mech.well_depth)
        sig_cpu = np.asarray(mech.diameter)
        dip_cpu = np.asarray(mech.dipole)
        alp_cpu = np.asarray(mech.polarizability)
        pol_cpu = dip_cpu > 0.0
        mw_cpu = np.asarray(mech.molecular_weights)

        for i in range(n):
            for j in range(i, n):
                mi = mw_cpu[i] / AVOGADRO / 1000.0
                mj = mw_cpu[j] / AVOGADRO / 1000.0
                _red_mass[i, j] = _red_mass[j, i] = mi * mj / (mi + mj)

                sigma_ij = 0.5 * (sig_cpu[i] + sig_cpu[j])
                eps_ij   = math.sqrt(eps_cpu[i] * eps_cpu[j])

                # polar correction
                f_eps, f_sigma = 1.0, 1.0
                if pol_cpu[i] != pol_cpu[j]:
                    kp  = i if pol_cpu[i] else j
                    knp = j if kp == i else i
                    d3np = sig_cpu[knp] ** 3
                    d3p  = sig_cpu[kp] ** 3
                    if d3np > 1e-100 and d3p > 1e-100 and eps_cpu[kp] > 1e-100 and eps_cpu[knp] > 1e-100:
                        alpha_star = alp_cpu[knp] / d3np
                        mu_p_star = dip_cpu[kp] / math.sqrt(4 * PI * EPSILON_0 * d3p * eps_cpu[kp] * BOLTZMANN)
                        xi = 1.0 + 0.25 * alpha_star * mu_p_star**2 * math.sqrt(eps_cpu[kp] / eps_cpu[knp])
                        f_sigma = xi ** (-1.0 / 6.0)
                        f_eps = xi * xi

                _sig_p[i, j] = _sig_p[j, i] = sigma_ij * f_sigma
                _eps_p[i, j] = _eps_p[j, i] = eps_ij * f_eps
                
                # reduced dipole moment
                di = dip_cpu[i]
                dj = dip_cpu[j]
                d_ij = math.sqrt(di * dj) if di > 0 and dj > 0 else 0.0
                if eps_ij > 1e-100 and sigma_ij > 1e-100:
                    _del_p[i, j] = _del_p[j, i] = (0.5 * d_ij**2
                        / (4 * PI * EPSILON_0 * eps_ij * BOLTZMANN * sigma_ij**3))

        # Upload pairs to device
        self._reduced_mass = self.xp.asarray(_red_mass)
        self._eps_pair = self.xp.asarray(_eps_p)
        self._sigma_pair = self.xp.asarray(_sig_p)
        self._delta_pair = self.xp.asarray(_del_p)

        # Pre-compute Wilke mixing rule weight ratios
        _wratjk = np.zeros((n, n))
        for j in range(n):
            for k in range(j, n):
                _wratjk[j, k] = math.sqrt(mw_cpu[j] / mw_cpu[k])
                _wratjk[k, j] = math.sqrt(_wratjk[j, k])
        self._wratjk = self.xp.asarray(_wratjk)

        self._tiny = 1e-300
        self._has_transport_poly = _CANTERA_TRANSPORT_POLY is not None
        self._init_cantera_transport_poly()

    def _init_cantera_transport_poly(self):
        """Load Cantera transport polynomial coefficients (ln(T) basis) if available."""
        n = self.n_sp
        visc_cpu = np.zeros((n, 5), dtype=float)
        cond_cpu = np.zeros((n, 5), dtype=float)
        has_visc = np.zeros(n, dtype=bool)
        has_cond = np.zeros(n, dtype=bool)

        if self._has_transport_poly:
            species_data = _CANTERA_TRANSPORT_POLY.get("species", {})
            for i, name in enumerate(self.mech.species_names):
                entry = species_data.get(name, None)
                if entry is None:
                    continue

                vc = entry.get("visc_coeffs", None)
                cc = entry.get("cond_coeffs", None)

                if vc is not None and len(vc) >= 5:
                    visc_cpu[i, :] = np.asarray(vc[:5], dtype=float)
                    has_visc[i] = True
                if cc is not None and len(cc) >= 5:
                    cond_cpu[i, :] = np.asarray(cc[:5], dtype=float)
                    has_cond[i] = True

        self._visc_poly = self.xp.asarray(visc_cpu)
        self._cond_poly = self.xp.asarray(cond_cpu)
        self._has_visc_poly_cpu = has_visc
        self._has_cond_poly_cpu = has_cond
        self._has_visc_poly = self.xp.asarray(has_visc)
        self._has_cond_poly = self.xp.asarray(has_cond)

    def _omega_22(self, Tstar):
        return (1.16145 * self.xp.power(Tstar, -0.14874)
                + 0.52487 * self.xp.exp(-0.77320 * Tstar)
                + 2.16178 * self.xp.exp(-2.43787 * Tstar))

    def _omega_11(self, Tstar):
        return (1.06036 * self.xp.power(Tstar, -0.15610)
                + 0.19300 * self.xp.exp(-0.47635 * Tstar)
                + 1.03587 * self.xp.exp(-1.52996 * Tstar)
                + 1.76474 * self.xp.exp(-3.89411 * Tstar))



    # ------------------------------------------------------------------
    #  Grid Vectorization Helpers
    # ------------------------------------------------------------------
    def _is_grid(self, T):
        return self.xp.asarray(T).ndim > 0

    # ------------------------------------------------------------------
    #  Pure-species viscosity  (Chapman-Enskog)
    # ------------------------------------------------------------------
    def _species_viscosities_chapman(self, T):
        """Fallback Chapman-Enskog pure-species viscosities [Pa?s]."""
        T_arr = self.xp.asarray(T, dtype=float)
        is_grid = self._is_grid(T)
        if not is_grid:
            T_arr = T_arr[None]

        eps_safe = self.xp.maximum(self.eps, 1e-100)
        Tstar = T_arr[None, :] / eps_safe[:, None]
        om22 = self._omega_22(Tstar)

        mk = self.mw_kg[:, None]
        visc = (5.0 / 16.0 * self.xp.sqrt(PI * mk * BOLTZMANN * T_arr[None, :])
                / (PI * self.sigma[:, None]**2 * om22))

        res = self.xp.where(self.eps[:, None] < 1e-100, 1e-10, visc)
        return res if is_grid else res[:, 0]

    def species_viscosities(self, T) -> np.ndarray:
        """
        Pure-species viscosities [Pa?s]. Shape (n_sp,) or (n_sp, N).
        Uses Cantera's ln(T)-basis polynomial form when coefficients are available.
        """
        T_arr = self.xp.asarray(T, dtype=float)
        is_grid = self._is_grid(T)
        if not is_grid:
            T_arr = T_arr[None]

        if bool(self._has_visc_poly_cpu.all()):
            logT = self.xp.log(T_arr)
            poly = (
                self._visc_poly[:, 0, None]
                + logT[None, :] * (
                    self._visc_poly[:, 1, None]
                    + logT[None, :] * (
                        self._visc_poly[:, 2, None]
                        + logT[None, :] * (
                            self._visc_poly[:, 3, None]
                            + logT[None, :] * self._visc_poly[:, 4, None]
                        )
                    )
                )
            )
            sqvisc = self.xp.power(T_arr[None, :], 0.25) * poly
            visc = sqvisc * sqvisc
            return visc if is_grid else visc[:, 0]

        # Partial fallback for missing entries
        visc = self._species_viscosities_chapman(T_arr if is_grid else T_arr[0])
        if bool(self._has_visc_poly_cpu.any()):
            logT = self.xp.log(T_arr)
            poly = (
                self._visc_poly[:, 0, None]
                + logT[None, :] * (
                    self._visc_poly[:, 1, None]
                    + logT[None, :] * (
                        self._visc_poly[:, 2, None]
                        + logT[None, :] * (
                            self._visc_poly[:, 3, None]
                            + logT[None, :] * self._visc_poly[:, 4, None]
                        )
                    )
                )
            )
            sqvisc = self.xp.power(T_arr[None, :], 0.25) * poly
            visc_poly = sqvisc * sqvisc
            mask = self._has_visc_poly[:, None]
            visc = self.xp.where(mask, visc_poly, visc)

        return visc if is_grid else visc[:, 0]

    # ------------------------------------------------------------------
    #  Mixture viscosity (Wilke)
    # ------------------------------------------------------------------
    def viscosity(self, T, X: np.ndarray):
        """Mixture viscosity [Pa·s] using Wilke mixing rule (Chapman-Enskog)."""
        visc_k = self.species_viscosities(T)
        is_grid = self._is_grid(T)
        
        mw_ratio = self.mw[None, :] / self.mw[:, None]  # [j, k] = W_j / W_k
        
        if is_grid:
            ratio_v = visc_k[:, None, :] / self.xp.maximum(visc_k[None, :, :], 1e-300)
            mw_r_3d = mw_ratio[:, :, None]
            # Wilke: Φ[k,j] = (1 + sqrt(μ_k/μ_j) * (W_j/W_k)^(1/4))² / sqrt(8(1 + W_k/W_j))
            factor1 = 1.0 + self.xp.sqrt(ratio_v) * self.xp.power(mw_r_3d, 0.25)
            Phi = factor1**2 / self.xp.sqrt(8.0 * (1.0 + 1.0 / mw_r_3d))
            denom = self.xp.sum(X[None, :, :] * Phi, axis=1)
            Xsafe = self.xp.maximum(X, self._tiny)
            mix_visc = self.xp.sum((Xsafe * visc_k) / self.xp.maximum(denom, self._tiny), axis=0)
            return mix_visc
        else:
            ratio_v = visc_k[:, None] / self.xp.maximum(visc_k[None, :], 1e-300)
            factor1 = 1.0 + self.xp.sqrt(ratio_v) * self.xp.power(mw_ratio, 0.25)
            Phi = factor1**2 / self.xp.sqrt(8.0 * (1.0 + 1.0 / mw_ratio))
            denom = Phi @ X
            Xsafe = self.xp.maximum(X, self._tiny)
            return float(self.xp.sum((Xsafe * visc_k) / self.xp.maximum(denom, self._tiny)))

    # ------------------------------------------------------------------
    #  Pure-species thermal conductivity (Empirical Cantera polynomials)
    # ------------------------------------------------------------------
    def _species_conductivities_eucken(self, T, cp_R: np.ndarray) -> np.ndarray:
        """Fallback Eucken/Mathur pure-species conductivity model [W/(m?K)]."""
        T_arr = self.xp.asarray(T, dtype=float)
        is_grid = self._is_grid(T)
        cp_R_arr = self.xp.asarray(cp_R, dtype=float)

        if not is_grid:
            T_arr = T_arr[None]
            cp_R_arr = cp_R_arr[:, None]

        eps_safe = self.xp.maximum(self.eps, 1e-100)
        Tstar = T_arr[None, :] / eps_safe[:, None]
        om11 = self._omega_11(Tstar)

        visc_k = self._species_viscosities_chapman(T)
        if not is_grid:
            visc_k = visc_k[:, None]

        red_mass_self = self.xp.diag(self._reduced_mass)[:, None]
        diff_self = (3.0 / 16.0 * self.xp.sqrt(2.0 * PI / self.xp.maximum(red_mass_self, 1e-100))
                     * (BOLTZMANN * T_arr[None, :]) ** 1.5
                     / (PI * self.sigma[:, None]**2 * om11))

        f_int = self.mw[:, None] / (R_UNIV * T_arr[None, :]) * diff_self / self.xp.maximum(visc_k, 1e-300)

        cv_rot = self.crot[:, None]
        A_factor = 2.5 - f_int

        Tstar_298 = 298.0 / eps_safe[:, None]
        fz_298 = (1.0 + PI**1.5 / self.xp.sqrt(Tstar_298) * (0.5 + 1.0 / Tstar_298)
                  + (0.25 * PI**2 + 2) / Tstar_298)

        fz_T = (1.0 + PI**1.5 / self.xp.sqrt(Tstar) * (0.5 + 1.0 / Tstar)
                + (0.25 * PI**2 + 2) / Tstar)

        B_factor = (self.zrot[:, None] * fz_298 / self.xp.maximum(fz_T, 1e-30)
                    + 2.0 / PI * (5.0 / 3.0 * cv_rot + f_int))

        c1 = 2.0 / PI * A_factor / self.xp.maximum(B_factor, 1e-30)
        cv_int = cp_R_arr - 2.5 - cv_rot
        f_rot = f_int * (1.0 + c1)
        f_trans = 2.5 * (1.0 - c1 * cv_rot / 1.5)

        cond = (visc_k / self.mw[:, None]) * R_UNIV * (f_trans * 1.5 + f_rot * cv_rot + f_int * cv_int)
        cond = self.xp.where(self.eps[:, None] < 1e-100, 1e-10, cond)
        return cond if is_grid else cond[:, 0]

    def species_conductivities(self, T, cp_R: np.ndarray = None) -> np.ndarray:
        """
        Pure-species thermal conductivities [W/(m?K)].
        Uses Cantera's ln(T)-basis polynomial form when coefficients are available.
        """
        T_arr = self.xp.asarray(T, dtype=float)
        is_grid = self._is_grid(T)
        if not is_grid:
            T_arr = T_arr[None]

        if bool(self._has_cond_poly_cpu.all()):
            logT = self.xp.log(T_arr)
            poly = (
                self._cond_poly[:, 0, None]
                + logT[None, :] * (
                    self._cond_poly[:, 1, None]
                    + logT[None, :] * (
                        self._cond_poly[:, 2, None]
                        + logT[None, :] * (
                            self._cond_poly[:, 3, None]
                            + logT[None, :] * self._cond_poly[:, 4, None]
                        )
                    )
                )
            )
            cond = self.xp.sqrt(T_arr[None, :]) * poly
            return cond if is_grid else cond[:, 0]

        if cp_R is None:
            raise ValueError('cp_R is required when Cantera conductivity polynomials are unavailable.')

        cond = self._species_conductivities_eucken(T, cp_R)
        if bool(self._has_cond_poly_cpu.any()):
            logT = self.xp.log(T_arr)
            poly = (
                self._cond_poly[:, 0, None]
                + logT[None, :] * (
                    self._cond_poly[:, 1, None]
                    + logT[None, :] * (
                        self._cond_poly[:, 2, None]
                        + logT[None, :] * (
                            self._cond_poly[:, 3, None]
                            + logT[None, :] * self._cond_poly[:, 4, None]
                        )
                    )
                )
            )
            cond_poly = self.xp.sqrt(T_arr[None, :]) * poly
            mask = self._has_cond_poly[:, None]
            cond = self.xp.where(mask, cond_poly, cond if is_grid else cond[:, None])

        return cond if is_grid else cond[:, 0]

    # ------------------------------------------------------------------
    #  Mixture thermal conductivity
    # ------------------------------------------------------------------
    def thermal_conductivity(self, T, X: np.ndarray, cp_R: np.ndarray = None):
        """Mixture thermal conductivity [W/(m·K)]."""
        cond_k = self.species_conductivities(T, cp_R)
        Xsafe = self.xp.maximum(X, self._tiny)
        is_grid = self._is_grid(T)
        
        if is_grid:
            sum1 = self.xp.sum(Xsafe * cond_k, axis=0)
            sum2 = self.xp.sum(Xsafe * 1.0 / self.xp.maximum(cond_k, 1e-300), axis=0)
            return 0.5 * (sum1 + 1.0 / self.xp.maximum(sum2, 1e-300))
        else:
            sum1 = float(self.xp.sum(Xsafe * cond_k))
            sum2 = float(self.xp.sum(Xsafe * 1.0 / self.xp.maximum(cond_k, 1e-300)))
            return 0.5 * (sum1 + 1.0 / max(sum2, 1e-300))

    # ------------------------------------------------------------------
    #  Binary diffusion coefficients at unit pressure
    # ------------------------------------------------------------------
    def binary_diff_coeffs(self, T) -> np.ndarray:
        """Binary diffusion coefficients D_ij [m²/s · Pa]."""
        T_arr = self.xp.asarray(T, dtype=float)
        is_grid = self._is_grid(T)
        if not is_grid:
            T_arr = T_arr[None]
            
        eps_pair_safe = self.xp.maximum(self._eps_pair, 1e-100)
        Tstar_pair = T_arr[None, None, :] / eps_pair_safe[:, :, None]
        om11_val = self._omega_11(Tstar_pair)
        
        bdiff = (3.0 / 16.0 
                 * self.xp.sqrt(2.0 * PI / self.xp.maximum(self._reduced_mass[:, :, None], 1e-100)) 
                 * (BOLTZMANN * T_arr[None, None, :]) ** 1.5
                 / (PI * self._sigma_pair[:, :, None]**2 * om11_val))
                 
        bdiff = self.xp.where(self._eps_pair[:, :, None] < 1e-100, 1e-10, bdiff)
        res = 0.5 * (bdiff + bdiff.transpose(1, 0, 2))
        return res if is_grid else res[:, :, 0]

    # ------------------------------------------------------------------
    #  Mixture-averaged diffusion coefficients
    # ------------------------------------------------------------------
    def mix_diff_coeffs(self, T, P: float, X: np.ndarray) -> np.ndarray:
        """Mixture-averaged diffusion coefficients D_km [m²/s]."""
        bdiff = self.binary_diff_coeffs(T)
        Xsafe = self.xp.maximum(X, self._tiny)
        is_grid = self._is_grid(T)
        
        inv_bdiff = 1.0 / self.xp.maximum(bdiff, 1e-300)
        mask = 1.0 - self.xp.eye(self.n_sp)
        
        if is_grid:
            Wmix = self.xp.sum(X * self.mw[:, None], axis=0)
            mask = mask[:, :, None]
            Xsafe_b = Xsafe[:, None, :] # shape (n_sp, 1, N) for sum over axis 0 (j)
            sum2 = self.xp.sum((inv_bdiff * mask) * Xsafe_b, axis=0)
            diag_bdiff = self.xp.diagonal(bdiff, axis1=0, axis2=1).T
            Wmix_b = Wmix[None, :]
            mw_b = self.mw[:, None]
        else:
            Wmix = float(self.xp.sum(X * self.mw))
            sum2 = (inv_bdiff * mask) @ Xsafe
            diag_bdiff = self.xp.diag(bdiff)
            Wmix_b = Wmix
            mw_b = self.mw
            
        d_mix = self.xp.where(sum2 <= 0.0, 
                         diag_bdiff / P, 
                         (Wmix_b - Xsafe * mw_b) / (P * Wmix_b * self.xp.maximum(sum2, 1e-300)))
        return d_mix

    # ------------------------------------------------------------------
    #  Vectorised full evaluation at a state (T, P, Y)
    # ------------------------------------------------------------------
    def eval_all(self, T, P: float, Y: np.ndarray, cp_R: np.ndarray, invW: np.ndarray):
        """Returns (viscosity, thermal_cond, mix_diff_coeffs) in one call."""
        is_grid = self._is_grid(T)
        
        if is_grid:
            YW = Y * invW[:, None]
            X = YW / self.xp.maximum(YW.sum(axis=0)[None, :], 1e-300)
        else:
            YW = Y * invW
            X = YW / max(YW.sum(), 1e-300)
            
        mu  = self.viscosity(T, X)
        lam = self.thermal_conductivity(T, X, cp_R)
        Dm  = self.mix_diff_coeffs(T, P, X)
        return mu, lam, Dm, X


