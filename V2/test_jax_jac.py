import jax
import jax.numpy as jnp
import time

n_pts = 34
nv = 55
N = n_pts * nv

@jax.jit
def dummy_residual(x):
    # x is (N,)
    x_r = x.reshape((n_pts, nv))
    T = x_r[:, 1]
    Y = x_r[:, 2:]
    
    # some non-linear dummy physics
    expT = jnp.exp(T / 1000.0)
    omega = jnp.sin(Y) * expT[:, None]
    
    # fake flux (diff between points)
    Y_diff = Y[1:] - Y[:-1]
    
    res = jnp.zeros_like(x_r)
    # just an arbitrary function to test jacfwd performance
    res = res.at[:-1, 2:].set(omega[:-1] + Y_diff)
    res = res.at[-1, 2:].set(omega[-1])
    res = res.at[:, 1].set(T**2)
    return res.flatten()

jac_func = jax.jit(jax.jacfwd(dummy_residual))

# Compile
x = jnp.ones(N)
print("Compiling jacobian...")
t0 = time.time()
J = jac_func(x)
print(f"Compilation took {time.time() - t0:.2f}s")
print(f"J shape: {J.shape}")

# Evaluate
t0 = time.time()
for _ in range(10):
    J = jac_func(x)
J.block_until_ready()
print(f"10 evaluations took {time.time() - t0:.4f}s")
