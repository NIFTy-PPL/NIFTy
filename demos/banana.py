from jax.config import config

config.update("jax_enable_x64", True)

import sys
import matplotlib.pyplot as plt
from functools import partial
from jax import numpy as jnp
from jax import lax, random
from jax import value_and_grad, jit

import jifty1 as jft

seed = 42
key = random.PRNGKey(seed)


# %%
def cartesian_product(arrays, out=None):
    import numpy as np

    # Generalized N-dimensional products
    arrays = [np.asarray(x) for x in arrays]
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    if out is None:
        out = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        out[..., i] = a
    return out.reshape(-1, la)


def banana_helper_phi_b(b, x):
    return jnp.array([x[0], x[1] + b * x[0]**2 - 100 * b])


def sample_nonstandard_hamiltonian(
    likelihood, primals, key, cg=jft.cg, cg_kwargs=None
):
    if not isinstance(likelihood, jft.Likelihood):
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)
    from jax.tree_util import Partial

    cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    white_sample = jft.random_like_shapewdtype(
        key, likelihood.left_sqrt_metric_tangents_shape
    )
    met_smpl = likelihood.left_sqrt_metric(primals, white_sample)
    inv_metric_at_p = partial(
        cg, Partial(likelihood.metric, primals), **cg_kwargs
    )
    signal_smpl = inv_metric_at_p(met_smpl)[0]
    return signal_smpl


def NonStandardMetricKL(
    likelihood,
    primals,
    n_samples,
    key,
    mirror_samples: bool = True,
    linear_sampling_cg=jft.cg,
    linear_sampling_kwargs=None,
    hamiltonian_and_gradient=None,
    _samples=None
):
    from jax.tree_util import Partial

    if not isinstance(likelihood, jft.Likelihood):
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)

    if _samples is None:
        _samples = []
        draw = Partial(
            sample_nonstandard_hamiltonian,
            likelihood=likelihood,
            primals=primals,
            cg=linear_sampling_cg,
            cg_kwargs=linear_sampling_kwargs
        )
        subkeys = random.split(key, n_samples)
        samples = tuple(draw(key=k) for k in subkeys)
    else:
        samples = tuple(_samples)

    return jft.kl.SampledKL(
        hamiltonian=likelihood,
        primals=primals,
        samples=samples,
        linearly_mirror_samples=mirror_samples,
        hamiltonian_and_gradient=hamiltonian_and_gradient
    )


# %%
b = 0.1

signal_response = partial(banana_helper_phi_b, b)
nll = jft.Gaussian(
    jnp.zeros(2), lambda x: x / jnp.array([100., 1.])
) @ signal_response

ham = nll
ham = ham.jit()
ham_vg = jit(value_and_grad(ham))

# %%
n_mgvi_iterations = 30
n_samples = [1] * (n_mgvi_iterations - 10) + [2] * 5 + [3, 3, 10, 10, 100]
n_newton_iterations = [7] * (n_mgvi_iterations - 10) + [10] * 6 + 4 * [25]
absdelta = 1e-12

initial_position = jnp.array([1., 1.])
mkl_pos = 1e-2 * initial_position

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mkl = NonStandardMetricKL(
        ham,
        mkl_pos,
        n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"miniter": 0},
        hamiltonian_and_gradient=ham_vg
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
                "miniter": 0
            },
            "name": "N",
        }
    )
    mkl_pos = opt_state.x
    msg = f"Post MGVI Iteration {i}: Energy {ham(mkl_pos):2.4e}"
    print(msg, file=sys.stderr)

# %%
b_space_smpls = jnp.array([mkl_pos + smpl for smpl in mkl.samples])

n_pix_sqrt = 1000
x = jnp.linspace(-10.0, 10.0, n_pix_sqrt, endpoint=True)
y = jnp.linspace(2.0, 17.0, n_pix_sqrt, endpoint=True)
X, Y = jnp.meshgrid(x, y)
XY = jnp.array([X, Y]).T
xy = XY.reshape((XY.shape[0] * XY.shape[1], 2))
es = jnp.exp(-lax.map(ham, xy)).reshape(XY.shape[:2]).T

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*b_space_smpls.T)
ax.plot(*mkl_pos, "rx")
fig.tight_layout()
fig.savefig("banana_mgvi_wo_regularization.png", dpi=400)
plt.close()
