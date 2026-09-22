"""
export_fgm_to_flamelet.py

Lee fgm_table.npz generado por generate_fgm_tables_cantera.py (kflame)
y exporta una tabla FGM en formato FlameMaster (.fla), que es el
formato de texto plano más aceptado por OpenFOAM, Fluent y solvers
académicos como FlameMaster, CHEM1D, etc.

Estructura de salida (un archivo .fla por flamelet / valor de Z):
  - Header global  : metadata, mecanismo, condiciones
  - Sección Body   : c, T, rho, cp, qdot, Su, Y_k para ese Z

También exporta:
  - fgm_table_full.fla  : tabla 2D completa (todos los Z en un solo archivo)
  - summary_ZC.csv      : Z, c, T, qdot en texto plano para verificación

Uso:
  python export_fgm_to_flamelet.py --npz fgm_runs/mi_run/fgm_table.npz
  python export_fgm_to_flamelet.py --npz fgm_runs/mi_run/fgm_table.npz \\
      --species CH4 O2 CO2 H2O CO H2 OH N2 \\
      --out-dir fgm_runs/mi_run/flamelet_export
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    d = dict(data)
    # Convertir object arrays a listas Python donde corresponda
    for key in ("species_names", "used_progress_species", "indicator_species"):
        if key in d:
            d[key] = list(d[key])
    return d


# ---------------------------------------------------------------------------
# Helpers de escritura
# ---------------------------------------------------------------------------

def _fmt_array(arr: np.ndarray, per_line: int = 8, indent: int = 2) -> str:
    """Convierte ndarray 1D a bloque de texto con saltos de línea."""
    prefix = " " * indent
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i : i + per_line]
        lines.append(prefix + "  ".join(f"{v: .6e}" for v in chunk))
    return "\n".join(lines)


def _header_block(
    Z: float,
    Z_st: float,
    pressure: float,
    T_in: float,
    Su: float,
    n_points: int,
    n_species: int,
    mech: str,
    fuel: str,
    oxidizer: str,
    progress_species: list[str],
    created_at: str,
) -> str:
    return (
        f"FLAMELET TABLE\n"
        f"  created_at        = {created_at}\n"
        f"  mechanism         = {mech}\n"
        f"  fuel              = {fuel}\n"
        f"  oxidizer          = {oxidizer}\n"
        f"  pressure          = {pressure:.2f}  [Pa]\n"
        f"  T_unburned        = {T_in:.2f}  [K]\n"
        f"  Z                 = {Z:.8f}  [-]\n"
        f"  Z_st              = {Z_st:.8f}  [-]\n"
        f"  Su                = {Su:.6e}  [m/s]\n"
        f"  numOfPoints       = {n_points}\n"
        f"  numOfSpecies      = {n_species}\n"
        f"  progress_variable = {' + '.join(progress_species)}\n"
        f"BODY\n"
    )


# ---------------------------------------------------------------------------
# Exportar un flamelet individual (un valor de Z)
# ---------------------------------------------------------------------------

def write_single_flamelet(
    path: Path,
    Z: float,
    Z_st: float,
    c_grid: np.ndarray,
    T_row: np.ndarray,
    rho_row: np.ndarray,
    cp_row: np.ndarray,
    conductivity_row: np.ndarray,
    qdot_row: np.ndarray,
    omega_c_row: np.ndarray,
    Su: float,
    Y_rows: dict[str, np.ndarray],     # {nombre_especie: array(n_c)}
    pressure: float,
    T_in: float,
    mech: str,
    fuel: str,
    oxidizer: str,
    progress_species: list[str],
    created_at: str,
) -> None:
    n_c = len(c_grid)
    n_sp = len(Y_rows)

    with path.open("w", encoding="utf-8") as f:
        f.write(_header_block(
            Z=Z, Z_st=Z_st, pressure=pressure, T_in=T_in,
            Su=Su, n_points=n_c, n_species=n_sp,
            mech=mech, fuel=fuel, oxidizer=oxidizer,
            progress_species=progress_species,
            created_at=created_at,
        ))

        # --- ejes ---
        f.write(f"  c [{n_c}]\n")
        f.write(_fmt_array(c_grid) + "\n\n")

        # --- campos termodinámicos ---
        for name, arr in [
            ("T [K]",             T_row),
            ("rho [kg/m3]",       rho_row),
            ("cp_mass [J/kg/K]",  cp_row),
            ("conductivity [W/m/K]", conductivity_row),
            ("qdot [J/m3/s]",     qdot_row),
            ("omega_c [kg/m3/s]", omega_c_row),
        ]:
            f.write(f"  {name} [{n_c}]\n")
            f.write(_fmt_array(arr) + "\n\n")

        # --- fracciones másicas ---
        for sp_name, Y_arr in Y_rows.items():
            f.write(f"  Y_{sp_name} [-] [{n_c}]\n")
            f.write(_fmt_array(Y_arr) + "\n\n")

        f.write("END\n")


# ---------------------------------------------------------------------------
# Exportar tabla 2D completa en un solo archivo
# ---------------------------------------------------------------------------

def write_full_table(
    path: Path,
    table: dict,
    species_to_export: list[str],
    all_species: list[str],
    pressure: float,
    T_in: float,
    mech: str,
    fuel: str,
    oxidizer: str,
    progress_species: list[str],
    created_at: str,
) -> None:
    Z_grid  = np.asarray(table["Z_grid"],  dtype=float)
    c_grid  = np.asarray(table["c_grid"],  dtype=float)
    T_tab   = np.asarray(table["T"],       dtype=float)
    rho_tab = np.asarray(table["rho"],     dtype=float)
    cp_tab  = np.asarray(table["cp_mass"], dtype=float)
    conductivity_tab = np.asarray(table["conductivity"], dtype=float)
    q_tab   = np.asarray(table["qdot"],    dtype=float)
    omega_c_tab = np.asarray(table["omega_c"], dtype=float)
    Y_tab   = np.asarray(table["Y"],       dtype=float)
    Su_arr  = np.asarray(table["Su"],      dtype=float)

    Z_st = float(Z_grid[np.argmin(np.abs(
        np.asarray(table["phi_grid"], dtype=float) - 1.0
    ))]) if "phi_grid" in table else float(Z_grid[len(Z_grid)//2])

    sp_to_idx = {sp: i for i, sp in enumerate(all_species)}
    export_idx = [sp_to_idx[sp] for sp in species_to_export if sp in sp_to_idx]
    export_names = [sp for sp in species_to_export if sp in sp_to_idx]
    skipped = [sp for sp in species_to_export if sp not in sp_to_idx]
    if skipped:
        print(f"  [WARN] Especies no encontradas (omitidas): {skipped}")

    n_Z = len(Z_grid)
    n_c = len(c_grid)

    with path.open("w", encoding="utf-8") as f:
        # ---- cabecera global ----
        f.write("FLAMELET TABLE 2D\n")
        f.write(f"  created_at        = {created_at}\n")
        f.write(f"  mechanism         = {mech}\n")
        f.write(f"  fuel              = {fuel}\n")
        f.write(f"  oxidizer          = {oxidizer}\n")
        f.write(f"  pressure          = {pressure:.2f}  [Pa]\n")
        f.write(f"  T_unburned        = {T_in:.2f}  [K]\n")
        f.write(f"  Z_st              = {Z_st:.8f}  [-]\n")
        f.write(f"  nZ                = {n_Z}\n")
        f.write(f"  nC                = {n_c}\n")
        f.write(f"  numOfSpecies      = {len(export_names)}\n")
        f.write(f"  species           = {' '.join(export_names)}\n")
        f.write(f"  progress_variable = {' + '.join(progress_species)}\n")
        f.write("BODY\n\n")

        # ---- eje Z ----
        f.write(f"Z [-] [{n_Z}]\n")
        f.write(_fmt_array(Z_grid) + "\n\n")

        # ---- eje c ----
        f.write(f"c [-] [{n_c}]\n")
        f.write(_fmt_array(c_grid) + "\n\n")

        # ---- Su(Z) ----
        f.write(f"Su [m/s] [{n_Z}]\n")
        f.write(_fmt_array(Su_arr) + "\n\n")

        # ---- campos 2D: almacenados fila=Z, columna=c ----
        for field_name, tab in [
            ("T [K]",            T_tab),
            ("rho [kg/m3]",      rho_tab),
            ("cp_mass [J/kg/K]", cp_tab),
            ("conductivity [W/m/K]", conductivity_tab),
            ("qdot [J/m3/s]",    q_tab),
            ("omega_c [kg/m3/s]", omega_c_tab),
        ]:
            f.write(f"{field_name} [{n_Z}x{n_c}]\n")
            for i in range(n_Z):
                f.write(f"  // Z = {Z_grid[i]:.6f}\n")
                f.write(_fmt_array(tab[i]) + "\n")
            f.write("\n")

        # ---- fracciones másicas 2D ----
        for sp_name, k in zip(export_names, export_idx):
            f.write(f"Y_{sp_name} [-] [{n_Z}x{n_c}]\n")
            for i in range(n_Z):
                f.write(f"  // Z = {Z_grid[i]:.6f}\n")
                f.write(_fmt_array(Y_tab[i, k]) + "\n")
            f.write("\n")

        f.write("END\n")


# ---------------------------------------------------------------------------
# CSV de verificación rápida
# ---------------------------------------------------------------------------

def write_summary_csv(path: Path, table: dict) -> None:
    Z_grid  = np.asarray(table["Z_grid"], dtype=float)
    c_grid  = np.asarray(table["c_grid"], dtype=float)
    T_tab   = np.asarray(table["T"],      dtype=float)
    q_tab   = np.asarray(table["qdot"],   dtype=float)
    omega_c_tab = np.asarray(table["omega_c"], dtype=float)

    with path.open("w", encoding="utf-8") as f:
        f.write("Z,c,T_K,qdot_Jm3s,omega_c_kgm3s\n")
        for i, Z in enumerate(Z_grid):
            for j, c in enumerate(c_grid):
                f.write(
                    f"{Z:.6f},{c:.6f},{T_tab[i,j]:.4f},"
                    f"{q_tab[i,j]:.4e},{omega_c_tab[i,j]:.4e}\n"
                )

    print(f"  CSV de verificación: {path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exporta fgm_table.npz al formato universal FlameMaster (.fla)."
    )
    p.add_argument(
        "--npz", type=str, required=True,
        help="Ruta al archivo fgm_table.npz generado por KFLAME o su referencia opcional",
    )
    p.add_argument(
        "--out-dir", type=str, default="",
        help="Directorio de salida. Por defecto: misma carpeta que el .npz",
    )
    p.add_argument(
        "--species", nargs="+",
        default=["CH4", "O2", "CO2", "H2O", "CO", "H2", "OH", "N2", "H", "O"],
        help="Especies a exportar en la tabla. Default: CH4 O2 CO2 H2O CO H2 OH N2 H O",
    )
    p.add_argument(
        "--all-species",
        action="store_true",
        help="Exportar todas las especies disponibles en species_names del fgm_table.npz.",
    )
    p.add_argument(
        "--mech", type=str, default="gri30.yaml",
        help="Nombre del mecanismo (solo para el header, no se carga).",
    )
    p.add_argument(
        "--fuel", type=str, default="CH4",
        help="Combustible (para el header).",
    )
    p.add_argument(
        "--oxidizer", type=str, default="O2:1.0, N2:3.76",
        help="Oxidante (para el header).",
    )
    p.add_argument(
        "--pressure", type=float, default=101325.0,
        help="Presión en Pa (para el header).",
    )
    p.add_argument(
        "--T-in", type=float, default=300.0,
        help="Temperatura de entrada en K (para el header).",
    )
    p.add_argument(
        "--single-flamelets", action="store_true",
        help="Además del archivo 2D, exportar un .fla por cada valor de Z.",
    )
    p.add_argument(
        "--no-csv", action="store_true",
        help="No generar el CSV de verificación (puede ser grande).",
    )
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"No se encontró: {npz_path}")

    out_dir = Path(args.out_dir) if args.out_dir.strip() else npz_path.parent / "flamelet_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("FGM TABLE EXPORTER  ->  FlameMaster (.fla) format")
    print("=" * 65)
    print(f"Leyendo : {npz_path.resolve()}")

    table = load_npz(npz_path)

    Z_grid       = np.asarray(table["Z_grid"],  dtype=float)
    c_grid       = np.asarray(table["c_grid"],  dtype=float)
    Su_arr       = np.asarray(table["Su"],       dtype=float)
    all_species  = list(table["species_names"])
    prog_species = list(table.get("used_progress_species", ["CO2", "H2O", "CO", "H2"]))

    if args.all_species:
        species_to_export = all_species
    else:
        species_to_export = args.species

    # Z_st: Z del flamelet más cercano a phi=1
    if "phi_grid" in table:
        phi_grid = np.asarray(table["phi_grid"], dtype=float)
        Z_st = float(Z_grid[np.argmin(np.abs(phi_grid - 1.0))])
    else:
        Z_st = float(Z_grid[len(Z_grid) // 2])

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"  n_Z       = {len(Z_grid)}")
    print(f"  n_c       = {len(c_grid)}")
    print(f"  Z_range   = [{Z_grid.min():.4f}, {Z_grid.max():.4f}]")
    print(f"  Z_st      = {Z_st:.6f}")
    print(f"  n_species = {len(all_species)}")
    print(f"  Exportando especies: {species_to_export}")
    print(f"  Salida    : {out_dir.resolve()}")

    # ----------------------------------------------------------------
    # 1) Tabla 2D completa  →  fgm_table_full.fla
    # ----------------------------------------------------------------
    full_path = out_dir / "fgm_table_full.fla"
    print("\n[1/3] Escribiendo tabla 2D completa ...")
    write_full_table(
        path=full_path,
        table=table,
        species_to_export=species_to_export,
        all_species=all_species,
        pressure=float(args.pressure),
        T_in=float(args.T_in),
        mech=args.mech,
        fuel=args.fuel,
        oxidizer=args.oxidizer,
        progress_species=prog_species,
        created_at=created_at,
    )
    size_mb = full_path.stat().st_size / 1e6
    print(f"  -> {full_path.name}  ({size_mb:.2f} MB)")

    # ----------------------------------------------------------------
    # 2) Flamelets individuales (opcional)  →  flamelet_Z_XXXXX.fla
    # ----------------------------------------------------------------
    if args.single_flamelets:
        print(f"\n[2/3] Escribiendo {len(Z_grid)} flamelets individuales ...")
        sp_to_idx = {sp: i for i, sp in enumerate(all_species)}
        T_tab   = np.asarray(table["T"],       dtype=float)
        rho_tab = np.asarray(table["rho"],     dtype=float)
        cp_tab  = np.asarray(table["cp_mass"], dtype=float)
        conductivity_tab = np.asarray(table["conductivity"], dtype=float)
        q_tab   = np.asarray(table["qdot"],    dtype=float)
        omega_c_tab = np.asarray(table["omega_c"], dtype=float)
        Y_tab   = np.asarray(table["Y"],       dtype=float)

        fl_dir = out_dir / "single_flamelets"
        fl_dir.mkdir(exist_ok=True)

        for i, Z in enumerate(Z_grid):
            tag = f"Z_{Z:.6f}".replace(".", "p")
            fla_path = fl_dir / f"flamelet_{tag}.fla"

            Y_rows = {
                sp: Y_tab[i, sp_to_idx[sp]]
                for sp in species_to_export
                if sp in sp_to_idx
            }

            write_single_flamelet(
                path=fla_path,
                Z=float(Z),
                Z_st=Z_st,
                c_grid=c_grid,
                T_row=T_tab[i],
                rho_row=rho_tab[i],
                cp_row=cp_tab[i],
                conductivity_row=conductivity_tab[i],
                qdot_row=q_tab[i],
                omega_c_row=omega_c_tab[i],
                Su=float(Su_arr[i]),
                Y_rows=Y_rows,
                pressure=float(args.pressure),
                T_in=float(args.T_in),
                mech=args.mech,
                fuel=args.fuel,
                oxidizer=args.oxidizer,
                progress_species=prog_species,
                created_at=created_at,
            )
        print(f"  -> {fl_dir.resolve()}")
    else:
        print("\n[2/3] Flamelets individuales omitidos (usa --single-flamelets).")

    # ----------------------------------------------------------------
    # 3) CSV de verificación
    # ----------------------------------------------------------------
    if not args.no_csv:
        print("\n[3/3] Escribiendo CSV de verificación ...")
        write_summary_csv(out_dir / "summary_ZC.csv", table)
    else:
        print("\n[3/3] CSV omitido (--no-csv).")

    # ----------------------------------------------------------------
    # Resumen final
    # ----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Exportación completada.")
    print(f"  Tabla completa  : {full_path.resolve()}")
    if args.single_flamelets:
        print(f"  Flamelets indiv : {(out_dir / 'single_flamelets').resolve()}")
    print("  Para OpenFOAM   : copia fgm_table_full.fla a constant/")
    print("  Para Fluent     : importar como flamelet table en el panel Species")
    print("-" * 65)


if __name__ == "__main__":
    main()
