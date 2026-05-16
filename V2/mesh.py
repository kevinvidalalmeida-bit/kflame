"""
mesh.py – Generación de malla y refinamiento adaptativo tipo Cantera.
"""
import numpy as np


# ---------------------------------------------------------------------------
#  Malla inicial (clustering gaussiano)
# ---------------------------------------------------------------------------
def _adaptive_xi(
    n_points: int,
    locs=(0.0, 0.3, 0.5, 1.0),
    cluster_strength: float = 8.0,
    cluster_sigma: float | None = None,
    n_dense: int = 4001,
) -> np.ndarray:
    if n_points < 2:
        raise ValueError("n_points debe ser >= 2")
    x1 = float(locs[1])
    x2 = float(locs[2])
    if not (0.0 <= x1 < x2 <= 1.0):
        return np.linspace(0.0, 1.0, n_points, dtype=float)

    center = 0.5 * (x1 + x2)
    sigma = float(cluster_sigma) if cluster_sigma else 0.5 * (x2 - x1)
    sigma = max(sigma, 1.0e-3)
    strength = max(0.0, float(cluster_strength))

    xi_dense = np.linspace(0.0, 1.0, n_dense, dtype=float)
    monitor = 1.0 + strength * np.exp(-((xi_dense - center) / sigma) ** 2)
    cdf = np.empty_like(xi_dense)
    cdf[0] = 0.0
    dxi = np.diff(xi_dense)
    cdf[1:] = np.cumsum(0.5 * (monitor[1:] + monitor[:-1]) * dxi)
    total = cdf[-1]
    if total <= 0.0 or not np.isfinite(total):
        return np.linspace(0.0, 1.0, n_points, dtype=float)
    cdf /= total
    xi_target = np.linspace(0.0, 1.0, n_points, dtype=float)
    xi = np.interp(xi_target, cdf, xi_dense)
    xi[0] = 0.0
    for j in range(1, xi.size):
        xi[j] = max(xi[j], xi[j - 1] + 1.0e-14)
    xi /= xi[-1]
    return xi


def initial_grid(
    width: float,
    n_points: int = 8,
    locs=(0.0, 0.3, 0.5, 1.0),
    cantera_seed_grid: bool = False,
    adaptive: bool = True,
    cluster_strength: float = 8.0,
    cluster_sigma: float | None = None,
) -> np.ndarray:
    if n_points < 2:
        raise ValueError("n_points debe ser >= 2")
    if width <= 0.0:
        raise ValueError("width debe ser > 0")
    if cantera_seed_grid and n_points == 8:
        # Match FreeFlame(width=...) default seed used by Cantera.
        return width * np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], dtype=float)
    if not adaptive:
        return np.linspace(0.0, width, n_points, dtype=float)
    xi = _adaptive_xi(
        n_points=n_points,
        locs=locs,
        cluster_strength=cluster_strength,
        cluster_sigma=cluster_sigma,
    )
    return width * xi


def grid_from_reference(npz_path: str, subsample: int | None = None) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    z_ref = np.asarray(data["z"], dtype=float)

    if subsample is not None and subsample > 1:
        idx = np.arange(0, len(z_ref), subsample)
        if idx[-1] != len(z_ref) - 1:
            idx = np.append(idx, len(z_ref) - 1)
        z_ref = z_ref[idx]

    return z_ref


def build_freeflame_refiner_profiles(problem, u: np.ndarray, T: np.ndarray, Y: np.ndarray) -> dict:
    """
    Selección de perfiles para refinamiento en free premixed flame.

    Flow1D activa U, V y T cuando la energía está activa; en este solver
    reducido (sin V) se toma:
      * u si refine_with_u
      * T si solve_energy y refine_with_T
      * Y_k si refine_with_species
    """
    profiles: dict[str, np.ndarray] = {}

    if bool(getattr(problem, "refine_with_u", True)):
        profiles["u"] = np.asarray(u, dtype=float)

    if bool(getattr(problem, "solve_energy", True)) and bool(
        getattr(problem, "refine_with_T", True)
    ):
        profiles["T"] = np.asarray(T, dtype=float)

    if bool(getattr(problem, "refine_with_species", True)):
        names = getattr(problem, "species_names", None)
        for k in range(Y.shape[0]):
            if names is not None and k < len(names):
                key = f"Y_{names[k]}"
            else:
                key = f"Y_{k}"
            profiles[key] = np.asarray(Y[k, :], dtype=float)

    return profiles


class AdaptiveRefiner:
    """
    Refinador de malla adaptativo basado en Cantera refine.cpp.

    Defaults alineados con Refiner::setCriteria:
      ratio=10.0, slope=0.8, curve=0.8, prune=-0.1
    """

    def __init__(self, ratio=10.0, slope=0.8, curve=0.8, prune=-0.1,
                 grid_min=1e-10, max_points=1000):
        if ratio < 2.0:
            raise ValueError("ratio debe ser >= 2.0")
        if not (0.0 <= slope <= 1.0):
            raise ValueError("slope debe estar entre 0 y 1")
        if not (0.0 <= curve <= 1.0):
            raise ValueError("curve debe estar entre 0 y 1")
        if prune > curve or prune > slope:
            raise ValueError("prune debe ser menor que curve y slope")

        self.ratio = float(ratio)
        self.slope = float(slope)
        self.curve = float(curve)
        self.prune = float(prune)
        self.grid_min = float(grid_min)
        self.max_points = int(max_points)
        self._thresh = 1e-14
        self._min_range = 0.01
        self._UNSET = 0
        self._KEEP = 1
        self._REMOVE = -1

    def analyze(self, z: np.ndarray, profiles: dict,
                j_fixed: int | None = None) -> tuple:
        z = np.asarray(z, dtype=float)
        n = int(z.size)
        if n < 2 or n >= self.max_points:
            return set(), set()

        dz = np.diff(z)
        insert_after: set[int] = set()
        keep = np.full(n, self._UNSET, dtype=int)
        keep[0] = self._KEEP
        keep[n - 1] = self._KEEP

        if j_fixed is not None and 0 <= j_fixed < n:
            keep[j_fixed] = self._KEEP

        pruning_enabled = self.prune > 0.0

        for _name, vals in profiles.items():
            vals = np.asarray(vals, dtype=float)
            if vals.shape != (n,):
                continue

            slope_arr = np.diff(vals) / dz

            val_min = float(np.min(vals))
            val_max = float(np.max(vals))
            slp_min = float(np.min(slope_arr))
            slp_max = float(np.max(slope_arr))

            val_mag = max(abs(val_max), abs(val_min))
            slp_mag = max(abs(slp_max), abs(slp_min))

            if (val_max - val_min) > self._min_range * max(val_mag, self._thresh):
                max_change = self.slope * (val_max - val_min)
                for j in range(n - 1):
                    ratio = abs(vals[j + 1] - vals[j]) / (max_change + self._thresh)
                    if ratio > 1.0 and dz[j] >= 2.0 * self.grid_min:
                        insert_after.add(j)
                    if pruning_enabled:
                        if ratio >= self.prune:
                            keep[j] = self._KEEP
                            keep[j + 1] = self._KEEP
                        elif keep[j] == self._UNSET:
                            keep[j] = self._REMOVE

            if (slp_max - slp_min) > self._min_range * max(slp_mag, self._thresh):
                max_change = self.curve * (slp_max - slp_min)
                for j in range(n - 2):
                    ratio = abs(slope_arr[j + 1] - slope_arr[j]) / (
                        max_change + self._thresh / dz[j]
                    )
                    if (
                        ratio > 1.0
                        and dz[j] >= 2.0 * self.grid_min
                        and dz[j + 1] >= 2.0 * self.grid_min
                    ):
                        insert_after.add(j)
                        insert_after.add(j + 1)
                    if pruning_enabled:
                        if ratio >= self.prune:
                            keep[j + 1] = self._KEEP
                        elif keep[j + 1] == self._UNSET:
                            keep[j + 1] = self._REMOVE

        for j in range(1, n - 1):
            if dz[j] > self.ratio * dz[j - 1]:
                insert_after.add(j)
                for jj in (j - 1, j, j + 1, j + 2):
                    if 0 <= jj < n:
                        keep[jj] = self._KEEP

            if dz[j - 1] > self.ratio * dz[j]:
                insert_after.add(j - 1)
                for jj in (j - 2, j - 1, j, j + 1):
                    if 0 <= jj < n:
                        keep[jj] = self._KEEP

            if j > 1 and (z[j + 1] - z[j - 1]) > self.ratio * dz[j - 2]:
                keep[j] = self._KEEP

            if j < n - 2 and (z[j + 1] - z[j - 1]) > self.ratio * dz[j + 1]:
                keep[j] = self._KEEP

        if pruning_enabled:
            for j in range(2, n - 1):
                if keep[j] == self._REMOVE and keep[j - 1] == self._REMOVE:
                    keep[j] = self._KEEP

            remove = {
                int(j) for j in range(1, n - 1)
                if keep[j] == self._REMOVE
            }
        else:
            remove = set()

        return insert_after, remove

    def refine(self, z: np.ndarray, profiles: dict,
               all_Y: np.ndarray | None = None,
               j_fixed: int | None = None) -> tuple:
        z = np.asarray(z, dtype=float)
        insert_after, remove = self.analyze(z, profiles, j_fixed)

        if not insert_after and not remove:
            return z, False, 0, 0

        keep_mask = np.ones(z.size, dtype=bool)
        for j in remove:
            if 0 < j < z.size - 1:
                keep_mask[j] = False

        z_new = []
        n_inserted = 0
        for j in range(z.size - 1):
            if keep_mask[j]:
                z_new.append(float(z[j]))
            if j in insert_after and (len(z_new) + 1) < self.max_points:
                z_new.append(0.5 * (float(z[j]) + float(z[j + 1])))
                n_inserted += 1

        if keep_mask[-1]:
            z_new.append(float(z[-1]))

        z_new = np.asarray(z_new, dtype=float)
        z_new = np.unique(z_new)

        n_removed = int(np.sum(~keep_mask[1:-1]))
        changed = bool(z_new.size != z.size)
        return z_new, changed, n_inserted, n_removed

    def interpolate_solution(self, z_old: np.ndarray, z_new: np.ndarray,
                             *arrays: np.ndarray) -> list:
        result = []
        for arr in arrays:
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                result.append(np.interp(z_new, z_old, arr))
            elif arr.ndim == 2:
                new_arr = np.empty((arr.shape[0], len(z_new)), dtype=float)
                for k in range(arr.shape[0]):
                    new_arr[k, :] = np.interp(z_new, z_old, arr[k, :])
                result.append(new_arr)
            else:
                raise ValueError(f"Array con ndim={arr.ndim} no soportado")
        return result
