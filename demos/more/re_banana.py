from functools import partial
import sys

from jax import numpy as jnp
from jax import lax, random
from jax import jit
from jax.config import config
import matplotlib.pyplot as plt

import nifty8.re as jft

config.update("jax_enable_x64", True)

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
    likelihood, primals, key, cg=jft.static_cg, cg_name=None, cg_kwargs=None
):
    if not isinstance(likelihood, jft.Likelihood):
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)
    from jax.tree_util import Partial

    cg_kwargs = cg_kwargs if cg_kwargs is not None else {}
    cg_kwargs = {"name": cg_name, **cg_kwargs}

    white_sample = jft.random_like(
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
    linear_sampling_cg=jft.static_cg,
    linear_sampling_name=None,
    linear_sampling_kwargs=None,
):
    from jax.tree_util import Partial

    if not isinstance(likelihood, jft.Likelihood):
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)

    draw = Partial(
        sample_nonstandard_hamiltonian,
        likelihood=likelihood,
        primals=primals,
        cg=linear_sampling_cg,
        cg_name=linear_sampling_name,
        cg_kwargs=linear_sampling_kwargs,
    )
    subkeys = random.split(key, n_samples)
    samples_stack = lax.map(lambda k: draw(key=k), subkeys)

    return jft.kl.SampleIter(
        mean=primals,
        samples=jft.unstack(samples_stack),
        linearly_mirror_samples=mirror_samples
    )


# %%
b = 0.1

signal_response = partial(banana_helper_phi_b, b)
nll = jft.Gaussian(
    jnp.zeros(2), lambda x: x / jnp.array([100., 1.])
) @ signal_response

ham = nll
ham = ham.jit()
ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))

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
    samples = NonStandardMetricKL(
        ham,
        mkl_pos,
        n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"miniter": 0},
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=mkl_pos,
        method="newton-cg",
        options={
            "fun_and_grad": partial(ham_vg, primals_samples=samples),
            "hessp": partial(ham_metric, primals_samples=samples),
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
    msg = f"Post MGVI Iteration {i}: Energy {samples.at(mkl_pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)

# %%
b_space_smpls = jnp.array(tuple(samples.at(mkl_pos)))

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
