from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import cantera as ct

from config import FlameCase


def build_gas(case: FlameCase) -> ct.Solution:
    gas = ct.Solution(case.mech)
    gas.TP = case.T_in, case.P
    gas.set_equivalence_ratio(case.phi, case.fuel, case.oxidizer)
    return gas


def build_free_flame(case: FlameCase, gas: ct.Solution) -> ct.FreeFlame:
    flame = ct.FreeFlame(gas, width=case.width)
    flame.transport_model = case.transport_model
    flame.set_refine_criteria(
        ratio=case.ratio,
        slope=case.slope,
        curve=case.curve,
        prune=case.prune,
    )
    return flame


def solve_free_flame(case: FlameCase) -> tuple[ct.FreeFlame, dict]:
    gas = build_gas(case)
    flame = build_free_flame(case, gas)

    flame.solve(loglevel=case.loglevel, auto=case.auto)

    data = extract_solution(case, flame)
    return flame, data


def extract_solution(case: FlameCase, flame: ct.FreeFlame) -> dict:
    gas = flame.gas

    z = np.asarray(flame.grid, dtype=float)
    u = np.asarray(flame.velocity, dtype=float)
    T = np.asarray(flame.T, dtype=float)
    Y = np.asarray(flame.Y, dtype=float)   # shape: (n_species, n_points)
    X = np.asarray(flame.X, dtype=float)

    result = {
        "case": asdict(case),
        "species_names": list(gas.species_names),
        "n_species": gas.n_species,
        "n_points": int(len(z)),
        "z": z,
        "u": u,
        "T": T,
        "Y": Y,
        "X": X,
        "Su": float(u[0]),            # velocidad de llama libre
        "rho_in": float(flame.density[0]),
        "mdot_in": float(flame.density[0] * flame.velocity[0]),
    }
    return result


def save_reference_npz(data: dict, outpath: str | Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        outpath,
        species_names=np.array(data["species_names"], dtype=object),
        z=data["z"],
        u=data["u"],
        T=data["T"],
        Y=data["Y"],
        X=data["X"],
        Su=data["Su"],
        rho_in=data["rho_in"],
        mdot_in=data["mdot_in"],
    )


def save_case_json(data: dict, outpath: str | Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "case": data["case"],
        "n_species": data["n_species"],
        "n_points": data["n_points"],
        "Su": data["Su"],
        "rho_in": data["rho_in"],
        "mdot_in": data["mdot_in"],
        "species_names": data["species_names"],
    }

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def summarize_solution(data: dict) -> str:
    return (
        f"n_points   = {data['n_points']}\n"
        f"n_species  = {data['n_species']}\n"
        f"Su [m/s]   = {data['Su']:.8f}\n"
        f"rho_u      = {data['rho_in']:.8f}\n"
        f"mdot_u     = {data['mdot_in']:.8f}\n"
        f"Tmin/Tmax  = {data['T'].min():.3f} / {data['T'].max():.3f}\n"
        f"zmin/zmax  = {data['z'].min():.6e} / {data['z'].max():.6e}\n"
    )