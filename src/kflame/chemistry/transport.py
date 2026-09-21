"""
transport_native.py – Mixture-averaged transport properties for ideal gas.

Implements the same model as Cantera's MixTransport / GasTransport:
  - Pure-species viscosity via Chapman-Enskog theory
  - Wilke mixing rule for mixture viscosity
  - Eucken correlation for pure-species conductivity
  - Mixture-averaged diffusion coefficients

Fits viscosity, conductivity and binary diffusion natively from molecular
parameters, NASA thermodynamics and local Monchick-Mason collision tables.
No mechanism-specific exported property fits are read. Neufeld correlations
remain only in explicit legacy analytical helper methods.

NO Cantera dependency. CPU NumPy arrays.
"""
from __future__ import annotations
import math
import numpy as np
from kflame.chemistry.mechanism import MechanismData, R_UNIV, BOLTZMANN, AVOGADRO, EPSILON_0
from kflame.chemistry.collision_integrals import native_transport_fits, native_conductivity_fits, _array_key
from collections import OrderedDict

def _mix_soret_coefficients(T, P, Y, X, wmix, mw, viscosity, binary, diffusion, epsilon, cstar):
    n, faces = Y.shape
    result = np.zeros((n, faces))
    for face in range(faces):
        a = np.zeros(n)
        for k in range(n):
            if Y[k, face] < 1e-20:
                continue
            total = 0.0
            for j in range(n):
                if j != k:
                    phi = (1 + math.sqrt(viscosity[k, face] / viscosity[j, face])
                           * (mw[j] / mw[k]) ** .25) ** 2 / math.sqrt(8 * (1 + mw[k] / mw[j]))
                    total += X[j, face] * phi
            a[k] = (15.0 / 4.0) * viscosity[k, face] / mw[k] / (1 + 1.065 * total / X[k, face])
        for k in range(n - 1):
            for j in range(k + 1, n):
                z = math.log(T[face] / epsilon[k, j])
                star = cstar[k, j, 8]
                for power in range(7, -1, -1):
                    star = star * z + cstar[k, j, power]
                scale = (1.2 * star - 1) / (binary[k, j, face] / P) / (mw[k] + mw[j])
                result[k, face] += scale * (Y[k, face] * a[j] - Y[j, face] * a[k])
                result[j, face] += scale * (Y[j, face] * a[k] - Y[k, face] * a[j])
        total = 0.0
        for k in range(n):
            result[k, face] *= diffusion[k, face] * mw[k] * wmix[face]
            total += result[k, face]
        for k in range(n):
            result[k, face] -= Y[k, face] * total
    return result


# Physical constants
PI = math.pi

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    njit = None

if njit is not None:
    _mix_soret_coefficients = njit(cache=True)(_mix_soret_coefficients)


if njit is not None:
    @njit(cache=True)
    def _eval_faces_poly_numba_core(T, Y, P, invW, mw, cond_poly, diff_poly):
        n_sp = Y.shape[0]
        n_faces = Y.shape[1]
        rho = np.empty(n_faces, dtype=np.float64)
        Dm = np.empty((n_sp, n_faces), dtype=np.float64)
        lam = np.empty(n_faces, dtype=np.float64)
        Wmix = np.empty(n_faces, dtype=np.float64)
        X = np.empty(n_sp, dtype=np.float64)
        cond = np.empty(n_sp, dtype=np.float64)

        for m in range(n_faces):
            inv_wmix = 0.0
            for k in range(n_sp):
                inv_wmix += Y[k, m] * invW[k]
            wm = 1.0 / max(inv_wmix, 1.0e-300)
            Wmix[m] = wm
            rho[m] = P * wm / (R_UNIV * T[m])

            for k in range(n_sp):
                X[k] = Y[k, m] * wm * invW[k]

            logT = math.log(T[m])
            sqrtT = math.sqrt(T[m])
            TsqrtT = T[m] * sqrtT

            sum1 = 0.0
            sum2 = 0.0
            for k in range(n_sp):
                poly = (
                    cond_poly[k, 0]
                    + logT * (
                        cond_poly[k, 1]
                        + logT * (
                            cond_poly[k, 2]
                            + logT * (
                                cond_poly[k, 3] + logT * cond_poly[k, 4]
                            )
                        )
                    )
                )
                ck = sqrtT * poly
                cond[k] = ck
                xk = max(X[k], 1.0e-300)
                sum1 += xk * ck
                sum2 += xk / max(ck, 1.0e-300)

            lam[m] = 0.5 * (sum1 + 1.0 / max(sum2, 1.0e-300))

            for k in range(n_sp):
                sumd = 0.0
                for j in range(n_sp):
                    if j == k:
                        continue
                    poly = (
                        diff_poly[k, j, 0]
                        + logT * (
                            diff_poly[k, j, 1]
                            + logT * (
                                diff_poly[k, j, 2]
                                + logT * (
                                    diff_poly[k, j, 3] + logT * diff_poly[k, j, 4]
                                )
                            )
                        )
                    )
                    bdiff = TsqrtT * poly
                    sumd += max(X[j], 1.0e-300) / max(bdiff, 1.0e-300)

                poly_diag = (
                    diff_poly[k, k, 0]
                    + logT * (
                        diff_poly[k, k, 1]
                        + logT * (
                            diff_poly[k, k, 2]
                            + logT * (
                                diff_poly[k, k, 3] + logT * diff_poly[k, k, 4]
                            )
                        )
                    )
                )
                diag_bdiff = TsqrtT * poly_diag
                if sumd <= 0.0:
                    Dm[k, m] = diag_bdiff / P
                else:
                    Dm[k, m] = (
                        wm - max(X[k], 1.0e-300) * mw[k]
                    ) / (P * wm * max(sumd, 1.0e-300))

        return rho, Dm, lam, Wmix
else:
    _eval_faces_poly_numba_core = None


def _host_array(arr):
    """Return a NumPy view/copy for one-time CPU preprocessing."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


_PAIR_CACHE = OrderedDict()


def _build_pair_data(mech):
    """Pressure/temperature-independent molecular data; NumPy CPU only."""
    n = mech.n_species
    # Pre-compute pair parameters (symmetric)
    _red_mass = np.zeros((n, n))
    _eps_p = np.zeros((n, n))
    _sig_p = np.zeros((n, n))
    _del_p = np.zeros((n, n))

    # Perform the pair mixing rules in NumPy (it's one-time init)
    eps_cpu = np.asarray(_host_array(mech.well_depth), dtype=float)
    sig_cpu = np.asarray(_host_array(mech.diameter), dtype=float)
    dip_cpu = np.asarray(_host_array(mech.dipole), dtype=float)
    alp_cpu = np.asarray(_host_array(mech.polarizability), dtype=float)
    pol_cpu = dip_cpu > 0.0
    mw_cpu = np.asarray(_host_array(mech.molecular_weights), dtype=float)

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


    # Pre-compute Wilke mixing rule weight ratios
    _wratjk = np.zeros((n, n))
    for j in range(n):
        for k in range(j, n):
            _wratjk[j, k] = math.sqrt(mw_cpu[j] / mw_cpu[k])
            _wratjk[k, j] = math.sqrt(_wratjk[j, k])

    return _red_mass, _eps_p, _sig_p, _del_p, _wratjk


def _pair_data(mech):
    key = _array_key((mech.molecular_weights, mech.well_depth, mech.diameter,
                      mech.dipole, mech.polarizability))
    if key in _PAIR_CACHE:
        _PAIR_CACHE.move_to_end(key)
        return _PAIR_CACHE[key]
    result = _build_pair_data(mech)
    for array in result:
        array.setflags(write=False)
    _PAIR_CACHE[key] = result
    if len(_PAIR_CACHE) > 8:
        _PAIR_CACHE.popitem(last=False)
    return result


class NativeTransport:
    """
    Mixture-averaged transport for ideal gas.
    Uses NumPy arrays on CPU.
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

        (self._reduced_mass, self._eps_pair, self._sigma_pair,
         self._delta_pair, self._wratjk) = tuple(
            self.xp.asarray(value) for value in _pair_data(mech)
        )

        self._tiny = 1e-300
        self._init_native_transport_poly()

    def _init_native_transport_poly(self):
        """Build all fits from this mechanism, never from a species-name archive."""
        stars, viscosity, diffusion = native_transport_fits(self.mech, self)
        self._cstar_poly = stars[2]
        conductivity = native_conductivity_fits(self.mech, self)
        self._visc_poly = self.xp.asarray(viscosity)
        self._cond_poly = self.xp.asarray(conductivity)
        self._diff_poly = self.xp.asarray(diffusion)
        self._has_visc_poly_cpu = np.ones(self.n_sp, dtype=bool)
        self._has_cond_poly_cpu = np.ones(self.n_sp, dtype=bool)
        self._has_diff_poly_cpu = np.ones((self.n_sp, self.n_sp), dtype=bool)
        self._has_visc_poly = self.xp.asarray(self._has_visc_poly_cpu)
        self._has_cond_poly = self.xp.asarray(self._has_cond_poly_cpu)
        self._has_diff_poly = self.xp.asarray(self._has_diff_poly_cpu)
        self._fast_poly_available = (
            _eval_faces_poly_numba_core is not None
            and getattr(self.xp, "__name__", "") == "numpy"
        )

    def eval_faces_poly_fast(self, T, P: float, Y: np.ndarray, invW: np.ndarray):
        """Fast CPU path for face transport using native polynomial fits."""
        if not self._fast_poly_available:
            return None
        return _eval_faces_poly_numba_core(
            np.asarray(T, dtype=np.float64),
            np.ascontiguousarray(Y, dtype=np.float64),
            float(P),
            np.asarray(invW, dtype=np.float64),
            np.asarray(self.mw, dtype=np.float64),
            np.asarray(self._cond_poly, dtype=np.float64),
            np.asarray(self._diff_poly, dtype=np.float64),
        )

    def thermal_diff_coeffs(self, T, P, Y):
        """Mixture-averaged Soret coefficients [kg/(m s)].

        Adapted from Cantera 3.2 MixTransport::getThermalDiffCoeffs (BSD-3);
        uses native molecular fits and zero-sum mass correction. This is the
        mixture closure, distinct from Dixon--Lewis multicomponent transport.
        """
        T = np.atleast_1d(np.asarray(T, dtype=float))
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        wmix = 1.0 / np.maximum(np.sum(Y / self.mw[:, None], axis=0), 1e-300)
        X = np.maximum(Y * wmix[None, :] / self.mw[:, None], 1e-20)
        return _mix_soret_coefficients(
            T, float(P), Y, X, wmix, self.mw, self.species_viscosities(T),
            self.binary_diff_coeffs(T), self.mix_diff_coeffs(T, P, X),
            self._eps_pair, self._cstar_poly,
        )

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
            return self.xp.sum((Xsafe * visc_k) / self.xp.maximum(denom, self._tiny))

    # ------------------------------------------------------------------
    #  Pure-species thermal conductivity (native molecular/NASA fits)
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
            raise ValueError('cp_R is required for the legacy analytical conductivity helper.')

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
            sum1 = self.xp.sum(Xsafe * cond_k)
            sum2 = self.xp.sum(Xsafe * 1.0 / self.xp.maximum(cond_k, 1e-300))
            return 0.5 * (sum1 + 1.0 / self.xp.maximum(sum2, 1e-300))

    # ------------------------------------------------------------------
    #  Binary diffusion coefficients at unit pressure
    # ------------------------------------------------------------------
    def binary_diff_coeffs(self, T) -> np.ndarray:
        """Binary diffusion coefficients D_ij [m²/s · Pa]."""
        T_arr = self.xp.asarray(T, dtype=float)
        is_grid = self._is_grid(T)
        if not is_grid:
            T_arr = T_arr[None]

        if bool(self._has_diff_poly_cpu.all()):
            logT = self.xp.log(T_arr)
            poly = (
                self._diff_poly[:, :, 0, None]
                + logT[None, None, :] * (
                    self._diff_poly[:, :, 1, None]
                    + logT[None, None, :] * (
                        self._diff_poly[:, :, 2, None]
                        + logT[None, None, :] * (
                            self._diff_poly[:, :, 3, None]
                            + logT[None, None, :] * self._diff_poly[:, :, 4, None]
                        )
                    )
                )
            )
            bdiff = T_arr[None, None, :] * self.xp.sqrt(T_arr)[None, None, :] * poly
            return bdiff if is_grid else bdiff[:, :, 0]

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
            Wmix = self.xp.sum(X * self.mw)
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
            X = YW / self.xp.maximum(YW.sum(), 1e-300)
            
        mu  = self.viscosity(T, X)
        lam = self.thermal_conductivity(T, X, cp_R)
        Dm  = self.mix_diff_coeffs(T, P, X)
        return mu, lam, Dm, X
