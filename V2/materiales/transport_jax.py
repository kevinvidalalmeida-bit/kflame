from __future__ import annotations
import jax
import jax.numpy as jnp
from mechanism_data import R_UNIV

@jax.jit
def eval_faces_poly_jax(T, Y, P, invW, mw, cond_poly, diff_poly):
    """
    Evaluates transport properties at a single face using JAX.
    T: scalar
    Y: (n_sp,)
    P: scalar
    invW: (n_sp,)
    mw: (n_sp,)
    cond_poly: (n_sp, 5)
    diff_poly: (n_sp, n_sp, 5)
    """
    n_sp = Y.shape[0]
    
    inv_wmix = jnp.dot(Y, invW)
    wm = 1.0 / jnp.maximum(inv_wmix, 1e-300)
    rho = P * wm / (R_UNIV * T)
    
    X = Y * wm * invW
    X_safe = jnp.maximum(X, 1e-300)
    
    T_safe = jnp.maximum(T, 1e-300)
    logT = jnp.log(T_safe)
    sqrtT = jnp.sqrt(T_safe)
    TsqrtT = T_safe * sqrtT
    
    # cond_poly is (n_sp, 5)
    poly = (
        cond_poly[:, 0]
        + logT * (
            cond_poly[:, 1]
            + logT * (
                cond_poly[:, 2]
                + logT * (
                    cond_poly[:, 3] + logT * cond_poly[:, 4]
                )
            )
        )
    )
    cond = sqrtT * poly
    cond_safe = jnp.maximum(cond, 1e-300)
    
    sum1 = jnp.sum(X_safe * cond)
    sum2 = jnp.sum(X_safe / cond_safe)
    lam = 0.5 * (sum1 + 1.0 / jnp.maximum(sum2, 1e-300))
    
    # diff_poly is (n_sp, n_sp, 5)
    poly_d = (
        diff_poly[:, :, 0]
        + logT * (
            diff_poly[:, :, 1]
            + logT * (
                diff_poly[:, :, 2]
                + logT * (
                    diff_poly[:, :, 3] + logT * diff_poly[:, :, 4]
                )
            )
        )
    )
    poly_diag = jnp.diag(poly_d)
    
    bdiff = TsqrtT * poly_d # (n_sp, n_sp)
    bdiff = jnp.maximum(bdiff, 1e-300)
    
    mask = 1.0 - jnp.eye(n_sp)
    inv_bdiff_masked = (1.0 / bdiff) * mask # (n_sp, n_sp)
    
    sumd = jnp.dot(inv_bdiff_masked, X_safe) # (n_sp,)
    sumd_safe = jnp.maximum(sumd, 1e-300)
    
    # Numba equivalent: Dm = (wm - X[k] * mw[k]) / (P * wm * sumd)
    Dm = (wm - X_safe * mw) / (P * wm * sumd_safe)
    
    # Also handle the pure species limit (sumd <= 0.0)
    diag_bdiff = TsqrtT * poly_diag
    pure_Dm = diag_bdiff / P
    Dm = jnp.where(sumd <= 0.0, pure_Dm, Dm)
    
    return rho, Dm, lam, wm, sumd_safe

vmap_eval_faces_poly_jax = jax.vmap(
    eval_faces_poly_jax,
    in_axes=(0, 0, None, None, None, None, None),
    out_axes=(0, 0, 0, 0, 0)
)
