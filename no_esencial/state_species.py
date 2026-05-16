import numpy as np


def pack_species(Y_ind: np.ndarray) -> np.ndarray:
    Y_ind = np.asarray(Y_ind, dtype=float)
    if Y_ind.ndim != 2:
        raise ValueError("Y_ind debe tener shape (n_ind_species, n_points)")
    return Y_ind.reshape(-1, order="C")


def unpack_species(x: np.ndarray, n_points: int, n_species: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    n_ind = n_species - 1
    expected = n_ind * n_points
    if x.size != expected:
        raise ValueError(
            f"Tamano incorrecto de x: {x.size}, esperado: {expected}"
        )

    Y_ind = x.reshape((n_ind, n_points), order="C")
    Y_last = 1.0 - np.sum(Y_ind, axis=0, keepdims=True)
    Y = np.vstack([Y_ind, Y_last])
    return Y
