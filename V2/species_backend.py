"""
species_backend.py – Interfaz con Cantera para propiedades termoquímicas
y de transporte para free premixed flame 1-D.
"""
import numpy as np
import cantera as ct


class SpeciesBackend:
    def __init__(self, problem):
        self.problem = problem
        self.backend_kind = "cantera"
        self.use_gpu = False
        self.gas = ct.Solution(problem.case.mech)
        self.W = np.asarray(self.gas.molecular_weights, dtype=float)
        self.invW = 1.0 / self.W
        self.n_species = int(self.gas.n_species)
        self._has_set_unnorm = hasattr(self.gas, "set_unnormalized_mass_fractions")

        self.transport_model = str(getattr(problem.case, "transport_model", "mixture-averaged"))
        self.flux_gradient_basis = str(
            getattr(problem, "flux_gradient_basis",
                    getattr(problem.case, "flux_gradient_basis", "molar"))
        ).strip().lower()
        self.soret_enabled = bool(
            getattr(problem, "soret_enabled",
                    getattr(problem.case, "soret_enabled", False))
        )

        if self.flux_gradient_basis not in ("molar", "mole", "mass"):
            raise ValueError(
                "flux_gradient_basis debe ser 'molar' o 'mass'."
            )

        # Resolver una sola vez qué atributo de Cantera leer para D_mix.
        if self.flux_gradient_basis in ("molar", "mole"):
            if hasattr(self.gas, "mix_diff_coeffs"):
                self._mix_diff_mode = 0
            elif hasattr(self.gas, "mix_diff_coeffs_mole"):
                self._mix_diff_mode = 1
            else:
                raise AttributeError(
                    "La instalación de Cantera no expone mix_diff_coeffs / mix_diff_coeffs_mole."
                )
        else:  # mass
            if hasattr(self.gas, "mix_diff_coeffs_mass"):
                self._mix_diff_mode = 2
            else:
                raise AttributeError(
                    "La instalación de Cantera no expone mix_diff_coeffs_mass."
                )

        # Declarado explícitamente: el solver base free-flame se deja
        # en mezcla-averaged / multicomponent SIN Soret, salvo que se extienda
        # el residual con el término térmico correspondiente.
        if self.soret_enabled:
            raise NotImplementedError(
                "soret_enabled=True fue solicitado, pero el término de Soret "
                "todavía no está implementado en residual.py. "
                "Usa soret_enabled=False para la versión actual alineada con "
                "free premixed flame base."
            )

    # ------------------------------------------------------------------
    #  Normalización segura de Y
    # ------------------------------------------------------------------
    def _normalize_Y(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        if not np.all(np.isfinite(Y)):
            raise ValueError("Se detectaron fracciones másicas no finitas.")
        if np.any(Y < -1.0e-3):
            raise ValueError("Se detectaron fracciones másicas negativas.")
        Y = np.clip(Y, 0.0, None)
        s = np.sum(Y)
        if s <= 0.0:
            raise ValueError("La suma de fracciones másicas no es positiva.")
        return Y / s

    def _set_state_unnormalized(self, T: float, Y: np.ndarray) -> None:
        """
        Replica Flow1D::setGas:
          setTemperature(T)
          setMassFractions_NoNorm(Y)
          setPressure(P)
        """
        if isinstance(Y, np.ndarray) and Y.dtype == float:
            Yv = Y
        else:
            Yv = np.asarray(Y, dtype=float)
        if not np.all(np.isfinite(Yv)):
            raise ValueError("Se detectaron fracciones másicas no finitas.")
        gas = self.gas
        gas.TP = float(T), self.problem.P
        if self._has_set_unnorm:
            gas.set_unnormalized_mass_fractions(Yv)
        else:
            gas.Y = Yv
        gas.TP = float(T), self.problem.P

    def _get_mix_diff_coeffs(self) -> np.ndarray:
        """
        Obtiene coeficientes mixture-averaged compatibles con la base del gradiente.
        """
        mode = self._mix_diff_mode
        if mode == 0:
            return np.asarray(self.gas.mix_diff_coeffs, dtype=float)
        if mode == 1:
            return np.asarray(self.gas.mix_diff_coeffs_mole, dtype=float)
        return np.asarray(self.gas.mix_diff_coeffs_mass, dtype=float)

    def eval_node_into(self, T: float, Y: np.ndarray,
                       omega_out: np.ndarray, hk_out: np.ndarray):
        """
        Evalúa propiedades en un nodo y escribe omega/hk en buffers preasignados.
        """
        self._set_state_unnormalized(T, Y)
        gas = self.gas

        rho = float(gas.density)
        Dm = self._get_mix_diff_coeffs()
        np.multiply(np.asarray(gas.net_production_rates, dtype=float),
                    self.W, out=omega_out)
        cp = float(gas.cp_mass)
        lam = float(gas.thermal_conductivity)
        hk_out[:] = np.asarray(gas.partial_molar_enthalpies, dtype=float)
        return rho, Dm, cp, lam

    # ------------------------------------------------------------------
    #  Evaluación en un nodo de malla
    # ------------------------------------------------------------------
    def eval_node(self, T: float, Y: np.ndarray):
        """
        Devuelve (rho, Dm, omega_mass, cp, lam, hk) en un punto.
        """
        omega = np.empty(self.n_species, dtype=float)
        hk = np.empty(self.n_species, dtype=float)
        rho, Dm, cp, lam = self.eval_node_into(T, Y, omega, hk)
        return rho, Dm, omega, cp, lam, hk

    def eval_node_species_only(self, T: float, Y: np.ndarray, return_W_mix=False):
        """
        Versión ligera que solo devuelve (rho, Dm, omega_mass).
        """
        Y = self._normalize_Y(Y)
        self.gas.TPY = float(T), self.problem.P, Y

        rho = float(self.gas.density)
        Dm = self._get_mix_diff_coeffs()
        wdot = np.asarray(self.gas.net_production_rates, dtype=float)
        omega_mass = wdot * self.W

        if return_W_mix:
            return rho, Dm, omega_mass, self.gas.mean_molecular_weight
        return rho, Dm, omega_mass

    # ------------------------------------------------------------------
    #  Evaluación en midpoint (cara entre j y j+1)
    # ------------------------------------------------------------------
    def eval_midpoint(self, T_left: float, T_right: float,
                      Y_left: np.ndarray, Y_right: np.ndarray):
        """
        Propiedades en la cara j+1/2 (promedio aritmético T, Y).
        """
        Tm = 0.5 * (float(T_left) + float(T_right))
        Ym = 0.5 * (Y_left + Y_right)
        self._set_state_unnormalized(Tm, Ym)
        rho = float(self.gas.density)
        Dm = self._get_mix_diff_coeffs()
        return rho, Dm, self.gas.mean_molecular_weight

    def eval_midpoint_full_transport(self, T_left: float, T_right: float,
                                     Y_left: np.ndarray, Y_right: np.ndarray):
        """
        Versión ligera para residual: rho, D_mix, lambda, W_mix en la cara.
        """
        Tm = 0.5 * (float(T_left) + float(T_right))
        Ym = 0.5 * (Y_left + Y_right)
        self._set_state_unnormalized(Tm, Ym)
        gas = self.gas
        rho = float(gas.density)
        Dm = self._get_mix_diff_coeffs()
        lam = float(gas.thermal_conductivity)
        return rho, Dm, lam, gas.mean_molecular_weight

    def eval_midpoint_full(self, T_left: float, T_right: float,
                           Y_left: np.ndarray, Y_right: np.ndarray):
        """
        Propiedades completas en la cara j+1/2.
        """
        Tm = 0.5 * (float(T_left) + float(T_right))
        Ym = 0.5 * (np.asarray(Y_left, dtype=float)
                    + np.asarray(Y_right, dtype=float))
        return self.eval_node(Tm, Ym)

    # ------------------------------------------------------------------
    #  Utilidad: densidad sola (para continuidad)
    # ------------------------------------------------------------------
    def density(self, T: float, Y: np.ndarray) -> float:
        self._set_state_unnormalized(T, Y)
        return float(self.gas.density)
