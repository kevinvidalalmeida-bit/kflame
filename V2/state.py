"""
state.py – Vector de estado con layout POR-PUNTO ENTRELAZADO.

Layout (igual que Cantera Domain1D / Flow1D):
  x = [u0, T0, Y0_0, Y1_0, ..., YK_0,   ← punto 0
       u1, T1, Y0_1, Y1_1, ..., YK_1,   ← punto 1
       ...
       uN, TN, Y0_N, Y1_N, ..., YK_N]   ← punto N-1

Con este layout el Jacobiano transitorio es diagonal exacto:
  J_transient[n, n] = J_steady[n, n] - mask[n] * rdt
idéntico a MultiJac::updateTransient en Cantera.
"""
from __future__ import annotations
import numpy as np

# Offsets de componentes dentro de un bloque de punto (igual que c_offset_* en Flow1D)
C_U = 0  # velocidad axial
C_T = 1  # temperatura
C_Y = 2  # inicio de fracciones másicas


def n_vars(n_species: int) -> int:
    """Número de variables por punto."""
    return 2 + n_species


def state_size(n_points: int, n_species: int) -> int:
    return n_points * n_vars(n_species)


def vidx(j: int, offset: int, nv: int) -> int:
    """Índice escalar: variable `offset` en punto `j`, con `nv` vars/punto."""
    return j * nv + offset


# ---------------------------------------------------------------------------
#  Pack / Unpack
# ---------------------------------------------------------------------------
def pack_state(u: np.ndarray, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    (u, T, Y) → vector 1-D en layout por-punto entrelazado.

    u : (n_pts,)
    T : (n_pts,)
    Y : (n_species, n_pts)
    """
    u = np.asarray(u, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float)

    n_pts = u.size
    if T.shape != (n_pts,):
        raise ValueError(f"T.shape={T.shape} != ({n_pts},)")
    if Y.ndim != 2 or Y.shape[1] != n_pts:
        raise ValueError(f"Y.shape={Y.shape} incompatible con n_pts={n_pts}")

    n_sp = Y.shape[0]
    nv = 2 + n_sp
    x = np.empty(n_pts * nv, dtype=float)
    for j in range(n_pts):
        b = j * nv
        x[b + C_U] = u[j]
        x[b + C_T] = T[j]
        x[b + C_Y : b + C_Y + n_sp] = Y[:, j]
    return x


def unpack_state(x: np.ndarray, n_points: int, n_species: int):
    """
    Vector 1-D → (u, T, Y).

    Retorna
    -------
    u : (n_pts,)
    T : (n_pts,)
    Y : (n_species, n_pts)
    """
    x = np.asarray(x, dtype=float)
    nv = 2 + n_species
    expected = n_points * nv
    if x.size != expected:
        raise ValueError(f"x.size={x.size}, expected={expected}")

    x_r = x.reshape(n_points, nv)
    u = x_r[:, C_U].copy()
    T = x_r[:, C_T].copy()
    Y = x_r[:, C_Y:].T.copy()   # (n_species, n_pts)
    return u, T, Y


# ---------------------------------------------------------------------------
#  Máscara transitoria
# ---------------------------------------------------------------------------
def build_transient_mask(n_points: int, n_species: int,
                         solve_energy: bool = True) -> np.ndarray:
    """
    Máscara transitoria m[n] = 1 para variables diferenciales (en tiempo).

    Reproduce el comportamiento de `diag` en Flow1D::evalEnergy / evalSpecies:
      - U:   siempre 0 (restricción algebraica, incluido en j_fixed)
      - T:   1 en puntos interiores, sólo si solve_energy=True
      - Y_k: 1 en puntos interiores para todos k
      - Fronteras j=0 y j=N-1: todo 0

    El punto j_fixed también deja mask[j_fixed*nv + C_U] = 0 por defecto ya
    que U es siempre algebraico; las ecuaciones de T y Y_k en j_fixed SÍ son
    diferenciales (igual que Cantera), porque allí sólo se reemplaza R[0].
    """
    nv = 2 + n_species
    mask = np.zeros(n_points * nv, dtype=int)
    for j in range(1, n_points - 1):
        b = j * nv
        # C_U: mask = 0 (algebraic)
        if solve_energy:
            mask[b + C_T] = 1
        mask[b + C_Y : b + C_Y + n_species] = 1
    return mask


# ---------------------------------------------------------------------------
#  Interpolación de estado entre mallas
# ---------------------------------------------------------------------------
def interpolate_state(x_old: np.ndarray, z_old: np.ndarray,
                      z_new: np.ndarray, n_species: int) -> np.ndarray:
    """
    Interpola linealmente el estado de z_old a z_new.
    Equivalente a Flow1D::setupGrid + interpolación de Sim1D::refine.
    """
    n_old = z_old.size
    n_new = z_new.size
    u_old, T_old, Y_old = unpack_state(x_old, n_old, n_species)

    u_new = np.interp(z_new, z_old, u_old)
    T_new = np.interp(z_new, z_old, T_old)
    Y_new = np.empty((n_species, n_new), dtype=float)
    for k in range(n_species):
        Y_new[k] = np.interp(z_new, z_old, Y_old[k])

    # Normalizar Y (equivale a Flow1D::resetBadValues post-interpolación)
    Y_new = np.clip(Y_new, 0.0, None)
    s = Y_new.sum(axis=0, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    Y_new /= s

    return pack_state(u_new, T_new, Y_new)
