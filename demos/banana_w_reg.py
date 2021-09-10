# %%
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


def banana_helper_phi_b(b, x):
    return np.array([x[0], x[1] + b * x[0]**2 - 100 * b])


# %%
b = 0.1

SCALE = 10.

signal_response = lambda s: banana_helper_phi_b(b, SCALE * s)
nll = jft.Gaussian(
    np.zeros(2), lambda x: x / np.array([100., 1.])
) @ signal_response
nll = nll.jit()
nll_vg = jit(value_and_grad(nll))

ham = jft.StandardHamiltonian(nll)
ham = ham.jit()
ham_vg = jit(value_and_grad(ham))


# %%  # MGVI
n_mgvi_iterations = 30
n_samples = [1] * (n_mgvi_iterations - 2) + [2] + [100]
n_newton_iterations = [7] * (n_mgvi_iterations - 10) + [10] * 6 + 4 * [25]
absdelta = 1e-10

initial_position = np.array([1., 1.])
mkl_pos = 1e-2 * initial_position

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mkl = jft.MetricKL(
        ham,
        mkl_pos,
        n_samples=n_samples[i],
        key=subkey,
        mirror_samples=True,
        hamiltonian_and_gradient=ham_vg,
        linear_sampling_kwargs={"miniter": 0}
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=mkl_pos,
        method="newton-cg",
        options={
            "fun_and_grad": mkl.energy_and_gradient,
            "hessp": mkl.metric,
            "energy_reduction_factor": None,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "cg_kwargs": {
                "miniter": 0,
                "name": None
            },
            "name": "N"
        }
    )
    mkl_pos = opt_state.x
    print(
        (
            f"Post MGVI Iteration {i}: Energy {ham(mkl_pos):2.4e}"
            f"; #NaNs {np.isnan(mkl_pos).sum()}"
        ),
        file=sys.stderr
    )

# %%  # geoVI
n_geovi_iterations = 15
n_samples = [1] * (n_geovi_iterations - 2) + [2] + [100]
n_newton_iterations = [7] * (n_geovi_iterations - 10) + [10] * 6 + [25] * 4
absdelta = 1e-10

initial_position = np.array([1., 1.])
gkl_pos = 1e-2 * initial_position

for i in range(n_geovi_iterations):
    print(f"GeoVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    gkl = jft.GeoMetricKL(
        ham,
        gkl_pos,
        n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"miniter": 0},
        non_linear_sampling_kwargs={
            "cg_kwargs": {
                "miniter": 0,
                "absdelta": None,
                "name": None
            },
            "maxiter": 20,
            "name": None
        },
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
            "energy_reduction_factor": None,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "cg_kwargs": {
                "miniter": 0,
                "name": None
            },
            "name": "N",
        }
    )
    gkl_pos = opt_state.x

# %%
absdelta = 1e-10
opt_state = jft.minimize(
    None,
    x0=np.array([1., 1.]),
    method="newton-cg",
    options={
        "fun_and_grad": ham_vg,
        "hessp": ham.metric,
        "energy_reduction_factor": None,
        "absdelta": absdelta,
        "maxiter": 100,
        "cg_kwargs": {
            "miniter": 0,
            "name": None
        },
        "name": "MAP"
    }
)
map_pos = opt_state.x
key, subkey = random.split(key, 2)
map_gkl = jft.GeoMetricKL(
    ham,
    map_pos,
    100,
    key=subkey,
    mirror_samples=True,
    linear_sampling_kwargs={"miniter": 0},
    non_linear_sampling_kwargs={
        "cg_kwargs": {
            "miniter": 0,
            "name": None
        },
        "maxiter": 20,
        "name": None
    },
    hamiltonian_and_gradient=ham_vg
)

# %%

n_pix_sqrt = 1000
x = np.linspace(-30 / SCALE, 30 / SCALE, n_pix_sqrt)
y = np.linspace(-15 / SCALE, 15 / SCALE, n_pix_sqrt)
X, Y = np.meshgrid(x, y)
XY = np.array([X, Y]).T
xy = XY.reshape((XY.shape[0] * XY.shape[1], 2))
es = np.exp(-lax.map(ham, xy)).reshape(XY.shape[:2]).T

fig, axs = plt.subplots(1, 3, figsize=(16, 9))

b_space_smpls = np.array([mkl_pos + smpl for smpl in mkl.samples])
contour = axs[0].contour(X, Y, es)
axs[0].clabel(contour, inline=True, fontsize=10)
axs[0].scatter(*b_space_smpls.T)
axs[0].plot(*mkl_pos, "rx")
axs[0].set_title("MGVI")

b_space_smpls = np.array([gkl_pos + smpl for smpl in gkl.samples])
contour = axs[1].contour(X, Y, es)
axs[1].clabel(contour, inline=True, fontsize=10)
axs[1].scatter(*b_space_smpls.T, alpha=0.7)
axs[1].plot(*gkl_pos, "rx")
axs[1].set_title("GeoVI")

b_space_smpls = np.array([map_pos + smpl for smpl in map_gkl.samples])
contour = axs[2].contour(X, Y, es)
axs[2].clabel(contour, inline=True, fontsize=10)
axs[2].scatter(*b_space_smpls.T, alpha=0.7)
axs[2].plot(*map_pos, "rx")
axs[2].set_title("MAP + GeoVI Samples")

fig.tight_layout()
fig.savefig("banana_vi_w_regularization.png", dpi=400)
plt.close()
