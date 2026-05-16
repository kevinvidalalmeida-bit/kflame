"""
test_native_vs_cantera.py – Verify native implementation against Cantera.

Compares thermo, transport, and kinetics outputs at several (T, Y) states.
"""
from __future__ import annotations
import numpy as np
import sys
import time

# ── Native modules ─────────────────────────────────────────────────────
from mechanism_data import load_mechanism
from thermo_native import NativeThermo
from transport_native import NativeTransport
from kinetics_native import NativeKinetics

# Path to gri30.yaml
GRI30 = r"c:\Users\lidia\OneDrive\Desktop\TFM\no_esencial\cantera-main\data\gri30.yaml"
P = 101325.0


def main():
    print("=" * 70)
    print("  Verificación: Native vs Cantera")
    print("=" * 70)

    # ── Load mechanism ─────────────────────────────────────────────────
    mech = load_mechanism(GRI30)
    print(f"Loaded: {mech.n_species} species, {mech.n_reactions} reactions")

    thermo = NativeThermo(mech)
    transport = NativeTransport(mech)
    kinetics = NativeKinetics(mech)

    # ── Try to import Cantera ──────────────────────────────────────────
    try:
        import cantera as ct
        has_cantera = True
        gas = ct.Solution("gri30.yaml")
        print(f"Cantera {ct.__version__} loaded for comparison\n")
    except ImportError:
        has_cantera = False
        print("Cantera not available – running native only (no comparison)\n")

    # ── Test states ────────────────────────────────────────────────────
    test_states = []

    # State 1: cold unburned CH4/air
    Y1 = np.zeros(mech.n_species)
    sp = {n: i for i, n in enumerate(mech.species_names)}
    Y1[sp["CH4"]] = 0.055
    Y1[sp["O2"]]  = 0.22
    Y1[sp["N2"]]  = 0.725
    Y1 /= Y1.sum()
    test_states.append(("T=300 unburned", 300.0, Y1))

    # State 2: flame zone
    Y2 = np.zeros(mech.n_species)
    Y2[sp["CH4"]] = 0.01
    Y2[sp["O2"]]  = 0.15
    Y2[sp["N2"]]  = 0.70
    Y2[sp["H2O"]] = 0.08
    Y2[sp["CO2"]] = 0.05
    Y2[sp["OH"]]  = 0.005
    Y2[sp["H"]]   = 0.001
    Y2[sp["CO"]]  = 0.004
    Y2 /= Y2.sum()
    test_states.append(("T=1500 flame", 1500.0, Y2))

    # State 3: hot products
    Y3 = np.zeros(mech.n_species)
    Y3[sp["N2"]]  = 0.72
    Y3[sp["H2O"]] = 0.14
    Y3[sp["CO2"]] = 0.10
    Y3[sp["O2"]]  = 0.03
    Y3[sp["NO"]]  = 0.01
    Y3 /= Y3.sum()
    test_states.append(("T=2200 products", 2200.0, Y3))

    for label, T, Y in test_states:
        print(f"\n{'─' * 60}")
        print(f"  {label}   T = {T} K")
        print(f"{'─' * 60}")

        # ── Native ────────────────────────────────────────────────
        rho_n = thermo.density(T, P, Y)
        cp_n  = thermo.cp_mass(T, Y)
        hk_n  = thermo.partial_molar_enthalpies(T)
        cp_R_n = thermo.cp_R(T)
        X_n = thermo.Y_to_X(Y)
        Wmix_n = thermo.mean_molecular_weight(Y, thermo.invW)

        mu_n, lam_n, Dm_n, _ = transport.eval_all(T, P, Y, cp_R_n, thermo.invW)

        C = rho_n * Y * thermo.invW
        g_RT = thermo.g_RT(T)
        wdot_n = kinetics.net_production_rates(T, C, g_RT)

        print(f"  Native: ρ = {rho_n:.6f} kg/m³")
        print(f"  Native: cp = {cp_n:.4f} J/(kg·K)")
        print(f"  Native: W_mix = {Wmix_n:.6f} kg/kmol")
        print(f"  Native: μ = {mu_n:.6e} Pa·s")
        print(f"  Native: λ = {lam_n:.6e} W/(m·K)")
        print(f"  Native: D_CH4 = {Dm_n[sp['CH4']]:.6e} m²/s")
        print(f"  Native: ẇ_CH4 = {wdot_n[sp['CH4']]:.6e} kmol/(m³·s)")
        print(f"  Native: ẇ_OH  = {wdot_n[sp['OH']]:.6e} kmol/(m³·s)")

        if has_cantera:
            gas.TPY = T, P, Y

            rho_c = gas.density
            cp_c  = gas.cp_mass
            hk_c  = np.asarray(gas.partial_molar_enthalpies)
            mu_c  = gas.viscosity
            lam_c = gas.thermal_conductivity
            Dm_c  = np.asarray(gas.mix_diff_coeffs)
            wdot_c = np.asarray(gas.net_production_rates)  # kmol/(m³·s)

            print(f"\n  Cantera: ρ = {rho_c:.6f}")
            print(f"  Cantera: cp = {cp_c:.4f}")
            print(f"  Cantera: μ = {mu_c:.6e}")
            print(f"  Cantera: λ = {lam_c:.6e}")
            print(f"  Cantera: D_CH4 = {Dm_c[sp['CH4']]:.6e}")
            print(f"  Cantera: ẇ_CH4 = {wdot_c[sp['CH4']]:.6e}")
            print(f"  Cantera: ẇ_OH  = {wdot_c[sp['OH']]:.6e}")

            # Errors
            err_rho = abs(rho_n - rho_c) / max(abs(rho_c), 1e-30)
            err_cp  = abs(cp_n - cp_c)  / max(abs(cp_c), 1e-30)
            err_hk  = np.max(np.abs(hk_n - hk_c) / np.maximum(np.abs(hk_c), 1.0))
            err_mu  = abs(mu_n - mu_c)  / max(abs(mu_c), 1e-30)
            err_lam = abs(lam_n - lam_c) / max(abs(lam_c), 1e-30)

            # Diffusion: relative error on mean
            Dm_c_safe = np.maximum(np.abs(Dm_c), 1e-30)
            err_Dm = float(np.mean(np.abs(Dm_n - Dm_c) / Dm_c_safe))

            # Kinetics: relative error (use log scale for magnitude)
            wdot_c_safe = np.maximum(np.abs(wdot_c), 1e-30)
            # Only compare species with significant rates
            sig = np.abs(wdot_c) > 1e-10
            if sig.sum() > 0:
                err_wdot = float(np.mean(np.abs(wdot_n[sig] - wdot_c[sig]) / wdot_c_safe[sig]))
            else:
                err_wdot = 0.0

            print(f"\n  Errors (relative):")
            print(f"    ρ:    {err_rho:.2e}")
            print(f"    cp:   {err_cp:.2e}")
            print(f"    h_k:  {err_hk:.2e}")
            print(f"    μ:    {err_mu:.2e}")
            print(f"    λ:    {err_lam:.2e}")
            print(f"    D_km: {err_Dm:.2e} (mean)")
            print(f"    ẇ:    {err_wdot:.2e} (mean, significant species)")

    # ── Speed benchmark ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Speed benchmark (1000 evaluations)")
    print(f"{'=' * 60}")

    T_bench, Y_bench = 1500.0, Y2

    t0 = time.perf_counter()
    for _ in range(1000):
        rho = thermo.density(T_bench, P, Y_bench)
        cp = thermo.cp_mass(T_bench, Y_bench)
        hk = thermo.partial_molar_enthalpies(T_bench)
        cp_R = thermo.cp_R(T_bench)
        X = thermo.Y_to_X(Y_bench)
        mu, lam, Dm, _ = transport.eval_all(T_bench, P, Y_bench, cp_R, thermo.invW)
        C = rho * Y_bench * thermo.invW
        g_RT = thermo.g_RT(T_bench)
        wdot = kinetics.net_production_rates(T_bench, C, g_RT)
    t_native = time.perf_counter() - t0
    print(f"  Native: {t_native:.3f} s  ({t_native/1000*1000:.2f} ms/eval)")

    if has_cantera:
        t0 = time.perf_counter()
        for _ in range(1000):
            gas.TPY = T_bench, P, Y_bench
            _ = gas.density
            _ = gas.cp_mass
            _ = gas.partial_molar_enthalpies
            _ = gas.viscosity
            _ = gas.thermal_conductivity
            _ = gas.mix_diff_coeffs
            _ = gas.net_production_rates
        t_cantera = time.perf_counter() - t0
        print(f"  Cantera: {t_cantera:.3f} s  ({t_cantera/1000*1000:.2f} ms/eval)")
        print(f"  Speedup: {t_cantera/t_native:.1f}x")


if __name__ == "__main__":
    main()
