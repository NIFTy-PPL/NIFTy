from jax.config import config

config.update("jax_enable_x64", True)

import sys
import matplotlib.pyplot as plt
from functools import partial
from jax import numpy as np
from jax import lax, random
from jax import value_and_grad, jit

import jifty1 as jft

seed = 42
key = random.PRNGKey(seed)


# %%
def cartesian_product(arrays, out=None):
    import numpy as onp

    # Generalized N-dimensional products
    arrays = [onp.asarray(x) for x in arrays]
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    if out is None:
        out = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        out[..., i] = a
    return out.reshape(-1, la)


def helper_phi_b(b, x):
    return b * x[0] * np.exp(b * x[1])


# %%
b = 2.

signal_response = partial(helper_phi_b, b)
nll = jft.Gaussian(0., lambda x: x / np.sqrt(1.)) @ signal_response

ham = jft.StandardHamiltonian(nll)
ham = ham.jit()
ham_vg = jit(value_and_grad(ham))

# %%
n_pix_sqrt = 1000
x = np.linspace(-4, 4, n_pix_sqrt)
y = np.linspace(-4, 4, n_pix_sqrt)
xx = cartesian_product((x, y))
ham_everywhere = np.vectorize(ham, signature="(2)->()")(xx).reshape(
    n_pix_sqrt, n_pix_sqrt
)
plt.imshow(
    np.exp(-ham_everywhere.T),
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower"
)
plt.colorbar()
plt.show()

# %%
n_mgvi_iterations = 30
n_samples = [1] * (n_mgvi_iterations - 10) + [2] * 5 + [3, 3, 10, 10, 100]
n_newton_iterations = [7] * (n_mgvi_iterations - 10) + [10] * 6 + 4 * [25]
absdelta = 1e-10

initial_position = np.array([1., 1.])
mkl_pos = 1e-2 * jft.Field(initial_position)

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mkl = jft.MetricKL(
        ham,
        mkl_pos,
        n_samples[i],
        key,
        mirror_samples=True,
        linear_sampling_kwargs={"absdelta": absdelta / 10.},
        hamiltonian_and_gradient=ham_vg
    )

    print("Minimizing...", file=sys.stderr)
    # TODO: Re-introduce a simplified version that works without fields
    opt_state = jft.minimize(
        None,
        x0=mkl_pos,
        method="newton-cg",
        options={
            "fun_and_grad": mkl.energy_and_gradient,
            "hessp": mkl.metric,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "name": "N",
        }
    )
    mkl_pos = opt_state.x
    msg = f"Post MGVI Iteration {i}: Energy {ham(mkl_pos):2.4e}"
    print(msg, file=sys.stderr)

# %%
n_geovi_iterations = 15
n_samples = [1] * (n_geovi_iterations - 10) + [2] * 5 + [3, 3, 5, 5, 100]
n_newton_iterations = [7] * (n_geovi_iterations - 10) + [10] * 6 + [25] * 4
absdelta = 1e-10

initial_position = np.array([1., 1.])
gkl_pos = 1e-2 * jft.Field(initial_position)

for i in range(n_geovi_iterations):
    print(f"geoVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    gkl = jft.GeoMetricKL(
        ham,
        gkl_pos,
        n_samples[i],
        key,
        mirror_samples=True,
        linear_sampling_kwargs={"absdelta": absdelta / 10.},
        non_linear_sampling_kwargs={"maxiter": 20},
        hamiltonian_and_gradient=ham_vg
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=gkl_pos,
        method="newton-cg",
        options={
            "fun_and_grad": gkl.energy_and_gradient,
            "hessp": gkl.metric,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "name": "N",
        }
    )
    gkl_pos = opt_state.x
    msg = f"Post geoVI Iteration {i}: Energy {ham(gkl_pos):2.4e}"
    print(msg, file=sys.stderr)

# %%
b_space_smpls = np.array([(mkl_pos + smpl).val for smpl in mkl.samples])

n_pix_sqrt = 200
x = np.linspace(-4.0, 4.0, n_pix_sqrt, endpoint=True)
y = np.linspace(-4.0, 4.0, n_pix_sqrt, endpoint=True)
X, Y = np.meshgrid(x, y)
XY = np.array([X, Y]).T
xy = XY.reshape((XY.shape[0] * XY.shape[1], 2))
es = np.exp(-lax.map(ham, xy)).reshape(XY.shape[:2]).T

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*b_space_smpls.T)
ax.plot(*mkl_pos, "rx")
plt.show()

# %%
b_space_smpls = np.array([(gkl_pos + smpl).val for smpl in gkl.samples])

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*b_space_smpls.T)
ax.plot(*gkl_pos, "rx")
plt.show()
