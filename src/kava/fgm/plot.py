"""
plot_fgm_figures.py  (v2 – Z-C space)

Post-procesa tablas FGM generadas por KAVA o su referencia opcional
y genera figuras estilo Fig. 8 en espacio (Z, c):

  (a) Y_sp1 vs c por flamelet (líneas coloreadas, rojo = no-monótono)
  (b) Y_sp2 vs c por flamelet
  (c) HRR tabulado en espacio (Z, c) — heatmap
  (d) HRR tabulado en espacio (Z, c) — heatmap (misma tabla, segunda especie)

NOTA: paneles izquierdos requieren raw_profiles/ (--save-raw-profiles en el generador).
      paneles derechos solo requieren fgm_table.npz.

Uso:
  python -m kava plot --run-dir runs/fgm/mi_run
  python -m kava plot --run-dir runs/fgm/mi_run --sp1 CO2 --sp2 OH
  python -m kava plot --run-dir runs/fgm/mi_run --log-scale
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nonmonotonic_segments(c: np.ndarray, Y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lista de (c_seg, Y_seg) donde c NO es estrictamente creciente."""
    dc = np.diff(c)
    bad = np.concatenate(([False], dc <= 0.0))
    segs = []
    in_seg = False
    start = 0
    for i, b in enumerate(bad):
        if b and not in_seg:
            start = max(0, i - 1)
            in_seg = True
        elif not b and in_seg:
            segs.append((c[start:i+1], Y[start:i+1]))
            in_seg = False
    if in_seg:
        segs.append((c[start:], Y[start:]))
    return segs


def load_table(run_dir: Path) -> dict:
    path = run_dir / "fgm_table.npz"
    if not path.exists():
        raise FileNotFoundError(f"No encontré fgm_table.npz en {run_dir}")
    return dict(np.load(path, allow_pickle=True))


def load_raw_profiles(run_dir: Path) -> list[dict]:
    raw_dir = run_dir / "raw_profiles"
    if not raw_dir.exists():
        return []
    recs = []
    for f in sorted(raw_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        recs.append({
            "phi": float(d["phi"][0]),
            "Z":   float(d["Z"][0]) if "Z" in d else None,
            "c":   d["c"],
            "Y":   d["Y"],
            "T":   d["T"],
            "species_names": list(d["species_names"]),
        })
    recs.sort(key=lambda r: r["phi"])
    return recs


def get_sp_idx(sp: str, names: list[str]) -> int:
    for i, s in enumerate(names):
        if s.strip() == sp.strip():
            return i
    raise KeyError(f"Especie '{sp}' no encontrada. Disponibles: {names[:20]}")


# ---------------------------------------------------------------------------
# Panel A / B: Y_sp vs c
# ---------------------------------------------------------------------------

def plot_species_profiles(
    ax: plt.Axes,
    recs: list[dict],
    sp: str,
    label: str = "1",
    n_skip: int = 1,
    cmap_name: str = "Blues",
) -> None:
    if not recs:
        ax.text(0.5, 0.5,
                "raw_profiles/ no disponible\nUsa --save-raw-profiles",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
        return

    cmap = plt.get_cmap(cmap_name)
    x_vals = np.array([r["Z"] if r["Z"] is not None else r["phi"] for r in recs])
    x_min, x_max = x_vals.min(), x_vals.max()

    any_nonmono = False
    missing_phi = []
    for idx, rec in enumerate(recs):
        if idx % max(1, n_skip) != 0 and idx != len(recs) - 1:
            continue
        try:
            k = get_sp_idx(sp, rec["species_names"])
        except KeyError:
            missing_phi.append(rec["phi"])
            continue

        c  = np.asarray(rec["c"], dtype=float)
        Yk = np.asarray(rec["Y"][k], dtype=float)
        x_val = rec["Z"] if rec["Z"] is not None else rec["phi"]

        norm = (x_val - x_min) / max(x_max - x_min, 1e-10)
        color = cmap(0.25 + 0.70 * norm)

        ax.plot(c, Yk, color=color, linewidth=0.9, alpha=0.85)

        segs = nonmonotonic_segments(c, Yk)
        for sc, sy in segs:
            ax.plot(sc, sy, color="red", linewidth=1.1, linestyle="--", alpha=0.95)
            any_nonmono = True

    if missing_phi:
        warnings.warn(
            f"Especie {sp} ausente en {len(missing_phi)} flamelets: phi={missing_phi}",
            stacklevel=2,
        )
    ax.set_xlabel("$c$", fontsize=11)
    ax.set_ylabel(f"$Y_{{{label}}}$  ($\\mathrm{{{sp}}}$)", fontsize=11)
    ax.set_xlim(0, 1)

    leg = [Line2D([0], [0], color=cmap(0.95), lw=1.2, label="Fully burned state")]
    if any_nonmono:
        leg.append(Line2D([0], [0], color="red", lw=1.0, ls="--",
                          label="Non-monotonicity"))
    ax.legend(handles=leg, fontsize=8, framealpha=0.7, loc="upper right")


# ---------------------------------------------------------------------------
# Panel C / D: HRR en espacio (Z, c)
# ---------------------------------------------------------------------------

def plot_hrr_heatmap_ZC(
    ax: plt.Axes,
    table: dict,
    label: str = "1",
    cmap_name: str = "jet",
    log_scale: bool = False,
    normalize_z: bool = False,
) -> None:
    """
    Heatmap de HRR.
    Eje X = Z  (fracción de mezcla de Bilger)
    Eje Y = c  (variable de progreso)
    """
    if "Z_grid" in table:
        y_axis = np.asarray(table["Z_grid"], dtype=float)
        if normalize_z:
            z_min = float(y_axis.min())
            z_span = max(float(y_axis.max() - z_min), 1.0e-300)
            y_axis = (y_axis - z_min) / z_span
            y_label = "$Z_{norm}$"
        else:
            y_label = "$Z$"
    else:
        y_axis = np.asarray(table["phi_grid"], dtype=float)
        y_label = r"$\phi$"
        warnings.warn("Z_grid no encontrado; usando phi_grid. Regenera con generador v2.")

    c_grid = np.asarray(table["c_grid"], dtype=float)
    Q = np.clip(np.asarray(table["qdot"], dtype=float), 0.0, None)

    Z_ax, C_ax = np.meshgrid(y_axis, c_grid)
    Q_plot = Q.T

    if log_scale and np.any(Q_plot > 0):
        norm = mcolors.LogNorm(vmin=max(Q_plot[Q_plot > 0].min(), 1.0), vmax=Q_plot.max())
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=Q_plot.max())

    pcm = ax.pcolormesh(Z_ax, C_ax, Q_plot, cmap=cmap_name, norm=norm, shading="gouraud")
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(r"HRR $[\mathrm{J\,m^{-3}\,s^{-1}}]$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel(y_label, fontsize=11)
    ax.set_ylabel("$c$",   fontsize=11)
    ax.set_xlim(y_axis.min(), y_axis.max())
    ax.set_ylim(c_grid.min(), c_grid.max())


# ---------------------------------------------------------------------------
# Figura 2x2 completa
# ---------------------------------------------------------------------------

def make_figure(
    run_dir: Path,
    sp1: str, sp2: str,
    n_skip: int,
    log_scale: bool,
    out_name: str,
    dpi: int,
    normalize_z: bool = False,
) -> None:
    table = load_table(run_dir)
    recs  = load_raw_profiles(run_dir)

    species_names = list(table["species_names"])
    print(f"Especies  : {len(species_names)} — {species_names[:12]} ...")
    if "Z_grid" in table:
        print(f"Z_grid    : {np.asarray(table['Z_grid'])}")
    else:
        print("Z_grid    : NO ENCONTRADO (tabla generada con versión vieja del generador)")
    print(f"c_grid    : {np.asarray(table['c_grid']).size} puntos")
    print(f"Raw profs : {len(recs)}")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
    fig.subplots_adjust(wspace=0.38, hspace=0.40)

    ax_a, ax_c = axes[0, 0], axes[0, 1]
    ax_b, ax_d = axes[1, 0], axes[1, 1]

    for ax, lbl in zip([ax_a, ax_b, ax_c, ax_d], ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(0.97, 0.97, lbl, transform=ax.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold")

    y_sym = "Z_{norm}" if ("Z_grid" in table and normalize_z) else ("Z" if "Z_grid" in table else r"\phi")

    ax_a.set_title(f"Flamelet profiles — $Y_1$ ($\\mathrm{{{sp1}}}$)", fontsize=9)
    ax_b.set_title(f"Flamelet profiles — $Y_2$ ($\\mathrm{{{sp2}}}$)", fontsize=9)
    ax_c.set_title(f"Tabulated HRR in $({y_sym},c)$ — $\\mathcal{{Y}}_1$", fontsize=9)
    ax_d.set_title(f"Tabulated HRR in $({y_sym},c)$ — $\\mathcal{{Y}}_2$", fontsize=9)

    plot_species_profiles(ax_a, recs, sp1, label="1", n_skip=n_skip, cmap_name="Blues")
    plot_species_profiles(ax_b, recs, sp2, label="2", n_skip=n_skip, cmap_name="Blues")
    plot_hrr_heatmap_ZC(ax_c, table, label="1", cmap_name="jet", log_scale=log_scale,
                        normalize_z=normalize_z)
    plot_hrr_heatmap_ZC(ax_d, table, label="2", cmap_name="jet", log_scale=log_scale,
                        normalize_z=normalize_z)

    out_path = run_dir / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"\nFigura guardada: {out_path.resolve()}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Extra: Su vs Z
# ---------------------------------------------------------------------------

def plot_su_Z(run_dir: Path, dpi: int, normalize_z: bool = False) -> None:
    table = load_table(run_dir)
    if "Z_grid" in table:
        x = np.asarray(table["Z_grid"], dtype=float)
        if normalize_z:
            x = (x - x.min()) / max(float(x.max() - x.min()), 1.0e-300)
            xlabel, fname = "$Z_{norm}$", "Su_Znorm.pdf"
        else:
            xlabel, fname = "$Z$  (Bilger)", "Su_Z.pdf"
    else:
        x, xlabel, fname = table["phi_grid"], r"$\phi$", "Su_phi.pdf"

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(x, np.asarray(table["Su"]) * 100, "o-", color="#1a6faf", lw=1.5, ms=4)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("$S_u$  [cm/s]", fontsize=12)
    ax.set_title("Laminar flame speed", fontsize=10)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.grid(True, lw=0.4, alpha=0.5)
    out = run_dir / fname
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Su guardado: {out.resolve()}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Extra: distribución c_grid
# ---------------------------------------------------------------------------

def plot_c_grid(run_dir: Path, dpi: int) -> None:
    table = load_table(run_dir)
    c_grid    = np.asarray(table["c_grid"])
    c_fine    = np.asarray(table["indicator_fine_c"])
    indicator = np.asarray(table["indicator_fine_value"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax1.plot(c_fine, indicator, color="#c0392b", lw=1.2)
    ax1.set_ylabel("Indicator", fontsize=10)
    ax1.set_title("Adaptive $c$-grid", fontsize=10)
    ax1.grid(True, lw=0.3)

    ax2.eventplot(c_grid, orientation="horizontal", lineoffsets=0.5,
                  linelengths=0.8, linewidths=0.6, color="#1a6faf")
    ax2.set_xlabel("$c$", fontsize=11)
    ax2.set_ylabel("Grid pts", fontsize=10)
    ax2.set_yticks([])
    ax2.grid(True, lw=0.3, axis="x")

    fig.tight_layout()
    out = run_dir / "c_grid_distribution.pdf"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    out_png = run_dir / "c_grid_distribution.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"c_grid guardado: {out.resolve()}")
    print(f"c_grid guardado: {out_png.resolve()}")
    plt.close(fig)


def plot_c_vs_Z_domain(run_dir: Path, dpi: int, normalize_z: bool = False) -> None:
    """Figura simple del espacio tabulado (Z, c) sin variables físicas."""
    table = load_table(run_dir)
    c_grid = np.asarray(table["c_grid"], dtype=float)

    if "Z_grid" in table:
        z_axis = np.asarray(table["Z_grid"], dtype=float)
        z_axis_plot = z_axis
        if normalize_z:
            z_axis_plot = (z_axis - z_axis.min()) / max(float(z_axis.max() - z_axis.min()), 1.0e-300)
            x_label = "$Z_{norm}$"
            out_base = "C_vs_Znorm_domain"
        else:
            x_label = "$Z$  (Bilger)"
            out_base = "C_vs_Z_domain"
    else:
        z_axis = np.asarray(table["phi_grid"], dtype=float)
        z_axis_plot = z_axis
        x_label = r"$\phi$"
        out_base = "C_vs_phi_domain"

    x = np.repeat(z_axis_plot, c_grid.size)
    y = np.tile(c_grid, z_axis.size)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(x, y, s=10, color="#1a6faf", alpha=0.65, edgecolors="none")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("$c$", fontsize=11)
    ax.set_ylim(c_grid.min(), c_grid.max())
    ax.set_xlim(float(z_axis_plot.min()), float(z_axis_plot.max()))
    ax.set_title("Tabla FGM en espacio $(Z,c)$", fontsize=10)
    ax.grid(True, lw=0.35, alpha=0.5)

    txt = (
        f"n_Z = {z_axis.size}\n"
        f"n_c = {c_grid.size}\n"
        f"Z: [{z_axis.min():.4f}, {z_axis.max():.4f}]"
    )
    ax.text(
        0.98, 0.02, txt, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    fig.tight_layout()
    out_pdf = run_dir / f"{out_base}.pdf"
    out_png = run_dir / f"{out_base}.png"
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"C vs Z guardado: {out_pdf.resolve()}")
    print(f"C vs Z guardado: {out_png.resolve()}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera figuras FGM estilo Fig. 8 en espacio (Z, c)."
    )
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--sp1", type=str, default="CO2")
    p.add_argument("--sp2", type=str, default="H2O")
    p.add_argument("--n-skip", type=int, default=1,
                   help="Graficar 1 de cada n_skip perfiles (paneles izquierdos).")
    p.add_argument("--log-scale", action="store_true")
    p.add_argument("--normalize-z", action="store_true",
                   help="Grafica Z normalizado a [0, 1] en lugar de Z Bilger absoluto.")
    p.add_argument("--out", type=str, default="fig8_ZC.pdf")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-extras", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir no existe: {run_dir}")
    make_figure(run_dir, args.sp1, args.sp2, args.n_skip,
                args.log_scale, args.out, args.dpi, normalize_z=args.normalize_z)
    if not args.no_extras:
        plot_su_Z(run_dir, args.dpi, normalize_z=args.normalize_z)
        plot_c_grid(run_dir, args.dpi)
        plot_c_vs_Z_domain(run_dir, args.dpi, normalize_z=args.normalize_z)


if __name__ == "__main__":
    main()
