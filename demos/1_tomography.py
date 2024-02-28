# %%
import dataclasses
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from jax import numpy as jnp
from jax import random
from jax.scipy.ndimage import map_coordinates

import nifty8.re as jft

# %%
dims = (64, 64, 64)
distances = tuple(1.0 / d for d in dims)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(offset_mean=2, offset_std=(1e-1, 3e-2))
cfm.add_fluctuations(  # Axis over which the kernle is defined
    dims,
    distances=distances,
    fluctuations=(1.0, 5e-1),
    loglogavgslope=(-5.0, 2e-1),
    flexibility=(1e0, 2e-1),
    asperity=(5e-1, 5e-2),
    prefix="ax1",
    non_parametric_kind="power",
)
correlated_field = cfm.finalize()  # forward model for a GP prior


# %%
def _los(x, /, start, end, *, distances, shape, n_sampling_points, order=1):
    from jax.scipy.ndimage import map_coordinates

    l2i = ((shape - 1) / shape) / distances
    start_iloc = start * l2i
    end_iloc = end * l2i
    ddi = (end_iloc - start_iloc) / n_sampling_points
    adi = jnp.arange(0, n_sampling_points) + 0.5
    dist = jnp.linalg.norm(end - start)
    pp = start_iloc[:, jnp.newaxis] + ddi[:, jnp.newaxis] * adi[jnp.newaxis]
    return map_coordinates(x, pp, order=order, cval=jnp.nan).sum() * (
        dist / n_sampling_points
    )


class LOS(jft.Model):
    start: jax.Array = dataclasses.field(metadata=dict(static=False))
    end: jax.Array = dataclasses.field(metadata=dict(static=False))
    distances: jax.Array = dataclasses.field(metadata=dict(static=False))

    def __init__(
        self,
        start,
        end,
        distances,
        shape,
        n_sampling_points=500,
        interpolation_order=1,
    ):
        # We assume that `start` and `end` are of shape (n_points, n_dimensions)
        self.start = jnp.array(start)
        self.end = jnp.array(end)
        self.distances = jnp.array(distances)
        self.shape = jnp.array(shape)
        self._los = partial(
            _los,
            n_sampling_points=n_sampling_points,
            order=interpolation_order,
            distances=self.distances,
            shape=self.shape,
        )

    def __call__(self, x):
        in_axes = (None, 0, 0)
        if self.start.ndim < self.end.ndim:
            in_axes = (None, None, 0)
        elif self.start.ndim > self.end.ndim:
            in_axes = (None, 0, None)
        return jax.vmap(self._los, in_axes=in_axes)(x, self.start, self.end)


test_key = random.PRNGKey(42)
for test_shape in ((10,), (25,), (12, 12), (6, 6, 6)):
    n_test_points = 1_000
    test_los = LOS(
        np.zeros((len(test_shape),)),
        np.ones((len(test_shape),))[np.newaxis],
        distances=tuple(1.0 / s for s in test_shape),
        shape=test_shape,
        n_sampling_points=n_test_points,
    )
    for test_x in (jnp.ones(test_shape), random.normal(test_key, test_shape)):
        desired = map_coordinates(
            test_x,
            [
                np.linspace(0, s - 1, num=n_test_points, endpoint=False)
                for s in test_x.shape
            ],
            order=1,
            cval=np.nan,
        ).mean()
        desired *= np.linalg.norm(np.ones((len(test_shape),)))
        np.testing.assert_allclose(
            test_los(test_x).sum(),
            desired,
            rtol=1e-2,
        )


# %%
class Forward(jft.Model):

    def __init__(self, log_density, los):
        self.log_density = log_density
        self.los = los
        # Track a method with which a random input for the model. This is not
        # strictly required but is usually handy when building deep models.
        super().__init__(init=log_density.init)

    def density(self, x):
        return jnp.exp(self.log_density(x))

    def __call__(self, x):
        return self.los(self.density(x))


key = random.PRNGKey(42)

start = (0.5, 0.5, 0.5)
# NOTE, synthetic end
n_synth_points = 100
key, sk = random.split(key)
end = random.uniform(
    sk,
    (n_synth_points, np.shape(start)[-1]),
    minval=0.05,
    maxval=0.95,
)

n_sampling_points = 256
los = LOS(
    start,
    end,
    distances=distances,
    shape=correlated_field.target.shape,
    n_sampling_points=n_sampling_points,
)
forward = Forward(correlated_field, los)

# %%
key, sk = random.split(key)
synth_pos = forward.init(sk)
synth_truth = forward(synth_pos)
key, sk = random.split(key)
noise_scl = 0.15
synth_noise = random.normal(sk, synth_truth.shape, synth_truth.dtype)
synth_noise = synth_noise * noise_scl * synth_truth
synth_data = synth_truth + synth_noise

# %%
lh = jft.Gaussian(synth_data, noise_cov_inv=1.0 / synth_noise**2).amend(forward)

# %%
n_vi_iterations = 6
delta = 1e-4
n_samples = 4
odir = "los_playground"

key, sk = random.split(key)
pos_init = jft.Vector(lh.init(sk))
# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
samples, state = jft.optimize_kl(
    lh,
    pos_init,
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    # Source for the stochasticity for sampling
    key=key,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples from
    # an implicit covariance matrix
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(pos_init) / 10.0, maxiter=100),
    ),
    # Arguements for the minimizer in the nonlinear updating of the samples
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name="MCG"), maxiter=35
        )
    ),
    sample_mode="linear_resample",
    odir=odir,
    resume=False,
)

# %%
synth_density = forward.density(synth_pos)
post_density = jax.vmap(forward.density)(samples.samples)

# %%
extent = (0, 1, 0, 1)
fig, axs = plt.subplots(3, 3, dpi=500)
for i, (ax_t, ax_p, ax_ps) in enumerate(axs):
    ax_t.imshow(synth_density.sum(axis=i), extent=extent)
    # not_i = tuple(set.difference({0, 1, 2}, {i}))
    # ax_p.plot(
    #     *end[:, not_i].T,
    #     "+",
    #     markersize=0.5,
    #     color="red",
    # )
    ax_p.imshow(post_density.mean(axis=0).sum(axis=i), extent=extent)
    ax_ps.imshow(post_density.std(axis=0).sum(axis=i), extent=extent)
plt.show()

# %%
post_density_mean = post_density.mean(axis=0)

X, Y, Z = np.mgrid[
    tuple(slice(0, 1, sz * 1j) for sz in forward.log_density.target.shape)
]
n_highest_points = 10_000
q_pm = np.quantile(post_density_mean, 1 - n_highest_points / post_density_mean.size)
q_s = np.quantile(synth_density, 1 - n_highest_points / synth_density.size)
q = max(q_pm, q_s)
ss_pm = post_density_mean >= q
ss_s = synth_density >= q
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=X[ss_pm].flatten(),
            y=Y[ss_pm].flatten(),
            z=Z[ss_pm].flatten(),
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.1),
            name="Posterior Mean",
        ),
        go.Scatter3d(
            x=X[ss_s].flatten(),
            y=Y[ss_s].flatten(),
            z=Z[ss_s].flatten(),
            mode="markers",
            marker=dict(size=3, color="gray", opacity=0.1),
            name="Truth",
        ),
        # go.Scatter3d(
        #     x=end[:, 0],
        #     y=end[:, 1],
        #     z=end[:, 2],
        #     mode="markers",
        #     marker=dict(size=1),
        # )
    ]
)
axis_range = list(zip(end.min(axis=0), end.max(axis=0)))
fig.update_layout(
    showlegend=True,
    template="plotly_white",
    scene=dict(
        aspectmode="data",
        xaxis=dict(nticks=4, range=axis_range[0]),
        yaxis=dict(nticks=4, range=axis_range[1]),
        zaxis=dict(nticks=4, range=axis_range[2]),
    ),
    width=720,
    height=480,
)
fig.show()
