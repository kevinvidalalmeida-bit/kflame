"""
species_backend_native.py – Drop-in replacement for species_backend.py.

Provides the exact same interface as SpeciesBackend but uses:
  - mechanism_data.py   → YAML parser
  - thermo_native.py    → NASA-7 thermodynamics
  - transport_native.py → mixture-averaged transport (Neufeld collision integrals)
  - kinetics_native.py  → full chemical kinetics (Arrhenius/3-body/Troe)

ZERO Cantera dependency.  GPU-ready via CuPy.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from mechanism_data import MechanismData, load_mechanism, R_UNIV
from thermo_native import NativeThermo
from transport_native import NativeTransport
from kinetics_native import NativeKinetics


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

    def __init__(self, problem, mech_data: MechanismData | None = None, use_gpu: bool = False):
        self.problem = problem
        P = problem.P

        # Load mechanism if not provided
        if mech_data is None:
            mech_path = problem.case.mech
            mech_data = load_mechanism(mech_path)

        # Move to GPU if requested
        mech_data.to_device(use_gpu)

        self.mech = mech_data
        self.mech.pressure = P

        # Initialize sub-modules (they will automatically detect cupy from mech arrays)
        self.thermo    = NativeThermo(mech_data)
        self.transport = NativeTransport(mech_data)
        self.kinetics  = NativeKinetics(mech_data)
        
        self.xp = self.thermo.xp

        # Convenience aliases
        self.W = self.xp.asarray(mech_data.molecular_weights)
        self.invW = self.xp.asarray(mech_data.inv_molecular_weights)
        self.n_species = mech_data.n_species

        # Transport / flux settings from problem
        self.flux_gradient_basis = str(
            getattr(problem, "flux_gradient_basis",
                    getattr(problem.case, "flux_gradient_basis", "molar"))
        ).strip().lower()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _safe_Y(self, Y: np.ndarray) -> np.ndarray:
        """Clip and ensure finite."""
        Y = self.xp.asarray(Y, dtype=float)
        Y = self.xp.clip(Y, 0.0, None)
        if Y.ndim == 1:
            s = Y.sum()
            if s > 0:
                Y = Y / s
        else:
            s = Y.sum(axis=0)
            Y = Y / self.xp.maximum(s, 1e-300)[None, :]
        return Y

    def _Y_to_X(self, Y: np.ndarray) -> np.ndarray:
        """Mass fractions → mole fractions."""
        return self.thermo.Y_to_X(Y)

    def _concentrations(self, T: float, Y: np.ndarray) -> np.ndarray:
        """Species concentrations [kmol/m³]."""
        rho = self.thermo.density(T, self.problem.P, Y)
        if Y.ndim == 1:
            return rho * Y * self.invW
        else:
            return rho[None, :] * Y * self.invW[:, None]

    # ------------------------------------------------------------------
    #  eval_grid_into  (Full Grid Vectorization)
    # ------------------------------------------------------------------
    def eval_grid_into(self, T: np.ndarray, Y: np.ndarray,
                       omega_out: np.ndarray, hk_out: np.ndarray):
        """
        Evaluate properties at all grid nodes simultaneously.
        T is (N,), Y is (n_sp, N).
        omega_out is (n_sp, N), hk_out is (n_sp, N).
        Returns (rho, Dm, cp, lam).
        """
        P = self.problem.P
        Y_safe = self._safe_Y(Y)

        # Thermodynamics
        rho = self.thermo.density(T, P, Y_safe)
        cp  = self.thermo.cp_mass(T, Y_safe)
        hk_out[:] = self.thermo.partial_molar_enthalpies(T)

        # Transport
        cp_R = self.thermo.cp_R(T)
        X = self._Y_to_X(Y_safe)
        _mu, lam, Dm, _X = self.transport.eval_all(T, P, Y_safe, cp_R, self.invW)

        # Kinetics
        C = rho[None, :] * Y_safe * self.invW[:, None]
        g_RT = self.thermo.g_RT(T)
        wdot = self.kinetics.net_production_rates(T, C, g_RT)  # kmol/(m³·s)
        self.xp.multiply(wdot, self.W[:, None], out=omega_out)  # → kg/(m³·s)

        return rho, Dm, cp, lam

    def eval_faces(self, T_face: np.ndarray, Y_face: np.ndarray):
        """
        Evaluate full transport properties at all faces simultaneously.
        T_face is (N-1,), Y_face is (n_sp, N-1).
        Returns (rho_f, Dm_f, lam_f, Wmix_f).
        """
        P = self.problem.P
        Y_safe = self._safe_Y(Y_face)

        rho = self.thermo.density(T_face, P, Y_safe)
        cp_R = self.thermo.cp_R(T_face)
        X = self._Y_to_X(Y_safe)
        lam = self.transport.thermal_conductivity(T_face, X, cp_R)
        Dm  = self.transport.mix_diff_coeffs(T_face, P, X)
        Wmix = self.thermo.mean_molecular_weight(Y_safe, self.invW)
        
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
        hk_out[:] = self.thermo.partial_molar_enthalpies(T)

        # Transport
        cp_R = self.thermo.cp_R(T)
        X = self._Y_to_X(Y)
        _mu, lam, Dm, _X = self.transport.eval_all(T, P, Y, cp_R, self.invW)

        # Kinetics
        C = rho * Y * self.invW
        g_RT = self.thermo.g_RT(T)
        wdot = self.kinetics.net_production_rates(T, C, g_RT)
        self.xp.multiply(wdot, self.W, out=omega_out)

        return rho, Dm, cp, lam

    # ------------------------------------------------------------------
    #  eval_node  (returns everything)
    # ------------------------------------------------------------------
    def eval_node(self, T: float, Y: np.ndarray):
        """Returns (rho, Dm, omega_mass, cp, lam, hk)."""
        omega = self.xp.empty(self.n_species, dtype=float)
        hk = self.xp.empty(self.n_species, dtype=float)
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
        Wmix = self.thermo.mean_molecular_weight(Ym, self.invW)
        return rho, Dm_arr, Wmix

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
        Wmix = self.thermo.mean_molecular_weight(Ym, self.invW)
        return rho, Dm, lam, Wmix

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
        return self.thermo.density(float(T), self.problem.P, Y)
