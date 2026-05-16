"""
Export Cantera transport polynomial coefficients for offline native usage.

Output format matches `transport_native.py` expectations:
  - species[name].visc_coeffs: for mu = (T**0.25 * poly(log(T)))**2
  - species[name].cond_coeffs: for lam = (T**0.5 * poly(log(T)))
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cantera as ct


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export Cantera transport polynomial coefficients.")
    p.add_argument(
        "--mechanism",
        default="gri30.yaml",
        help="Cantera mechanism YAML (default: gri30.yaml)",
    )
    p.add_argument(
        "--out",
        default="cantera_transport_poly_coeffs.json",
        help="Output JSON path (default: cantera_transport_poly_coeffs.json)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    gas = ct.Solution(args.mechanism)
    out = {
        "source": "cantera",
        "cantera_version": ct.__version__,
        "mechanism": args.mechanism,
        "transport_model": gas.transport_model,
        "basis": {
            "viscosity": "mu = (T**0.25 * (a0 + a1*lnT + a2*lnT^2 + a3*lnT^3 + a4*lnT^4))^2",
            "conductivity": "lambda = T**0.5 * (b0 + b1*lnT + b2*lnT^2 + b3*lnT^3 + b4*lnT^4)",
        },
        "species": {},
    }

    for i, name in enumerate(gas.species_names):
        out["species"][name] = {
            "index": int(i),
            "visc_coeffs": [float(x) for x in gas.get_viscosity_polynomial(i)],
            "cond_coeffs": [float(x) for x in gas.get_thermal_conductivity_polynomial(i)],
        }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path} ({len(out['species'])} species, Cantera {ct.__version__})")


if __name__ == "__main__":
    main()
