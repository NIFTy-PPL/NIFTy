#!/usr/bin/env python3

# %%
import jax

jax.config.update("jax_enable_x64", True)

# %%
from nifty8 import re as jft

dims = (128, 128)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(offset_mean=2, offset_std=(1e-1, 3e-2))
cfm.add_fluctuations(  # Axis over which the kernle is defined
    dims,
    distances=tuple(1.0 / d for d in dims),
    fluctuations=(1.0, 5e-1),
    loglogavgslope=(-3.0, 2e-1),
    flexibility=(1e0, 2e-1),
    asperity=(5e-1, 5e-2),
    prefix="ax1",
    non_parametric_kind="power",
)
correlated_field = cfm.finalize()  # forward model for a GP prior

# %%
from jax import numpy as jnp


class Forward(jft.Model):
    def __init__(self, correlated_field):
        self._cf = correlated_field
        # Track a method with which a random input for the model. This is not
        # strictly required but is usually handy when building deep models.
        super().__init__(init=correlated_field.init)

    def __call__(self, x):
        # NOTE, any kind of masking of the output, non-linear and linear
        # transformation could be carried out here. Models can also combined and
        # nested in any way and form.
        return jnp.exp(self._cf(x))


forward = Forward(correlated_field)

# REMOVE STARTING HERE
from jax import random
from matplotlib import pyplot as plt

truth = forward(forward.init(random.PRNGKey(3141)))
data = random.poisson(random.PRNGKey(2718), truth)
jnp.save("data.npy", data)
plt.imshow(data)
plt.colorbar()
# REMOVE STOP HERE

data = jnp.load("data.npy")
lh = jft.Poissonian(data).amend(forward)

# %%
from jax import random

key = random.PRNGKey(42)
key, sk = random.split(key, 2)
# NIFTy is agnostic w.r.t. the type of input it gets as long as it supports core
# airthmetics properties. Tell NIFTy to treat our parameter dictionary as a
# vector.
samples = jft.Samples(pos=jft.Vector(lh.init(sk)), samples=None, keys=None)

delta = 1e-4
absdelta = delta * jft.size(samples.pos)

opt_vi = jft.OptimizeVI(lh, n_total_iterations=25)
opt_vi_st = opt_vi.init_state(
    key,
    # Typically on the order of 2-12
    n_samples=lambda i: 1 if i < 2 else (2 if i < 4 else 6),
    # Arguments for the conjugate gradient method used to drawing samples from
    # an implicit covariance matrix
    draw_linear_kwargs=dict(
        cg_name="SL", cg_kwargs=dict(absdelta=absdelta / 10.0, maxiter=100)
    ),
    # Arguements for the minimizer in the nonlinear updating of the samples
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN", xtol=delta, cg_kwargs=dict(name=None), maxiter=5
        )
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(minimize_kwargs=dict(name="M", xtol=delta, maxiter=35)),
    sample_mode=lambda i: "nonlinear_resample" if i < 3 else "nonlinear_update",
)
for i in range(opt_vi.n_total_iterations):
    print(f"Iteration {i+1:04d}")
    # Continuously updates the samples of the approximate posterior distribution
    samples, opt_vi_st = opt_vi.update(samples, opt_vi_st)
    print(opt_vi.get_status_message(samples, opt_vi_st))

from functools import partial

# %%
import jax
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pm = jax.tree_map(partial(jnp.mean, axis=0), jax.vmap(forward)(samples.samples))
ps = jax.tree_map(partial(jnp.std, axis=0), jax.vmap(forward)(samples.samples))

fig = make_subplots(
    rows=2,
    cols=2,
    horizontal_spacing=0.12,
    vertical_spacing=0.2,
    subplot_titles=("truth", "post. mean", "|truth - post. mean|", "post. std."),
)
x, y = tuple(np.mgrid[slice(0.0, 1.0, d * 1j)] for d in dims)
fig.add_trace(go.Heatmap(x=x, y=y, z=truth.T, coloraxis="coloraxis"), 1, 1)
fig.add_trace(go.Heatmap(x=x, y=y, z=pm.T, coloraxis="coloraxis"), 1, 2)
fig.add_trace(
    go.Heatmap(x=x, y=y, z=np.abs(truth - pm).T, coloraxis="coloraxis3"), 2, 1
)
fig.add_trace(go.Heatmap(x=x, y=y, z=ps.T, coloraxis="coloraxis3"), 2, 2)
for i in range(1, 4 + 1):
    ya = dict(title="y") if i in (1, 3) else dict()
    xa = dict(title="x") if i in (3, 4) else dict()
    fig.update_layout(
        {
            f"xaxis{i}": dict(range=(0, 1)) | xa,
            f"yaxis{i}": dict(range=(0, 1), scaleanchor=f"x{i}", scaleratio=1.0) | ya,
        }
    )
fig.update_layout(
    template="plotly_white",
    width=550,
    height=550,
    coloraxis=dict(
        colorbar_x=0.425,
        colorbar_y=+0.79,
        colorbar_len=0.4,
        colorscale="viridis",
        colorbar_thickness=15,
    ),
    coloraxis3=dict(
        colorbar_x=0.425,
        colorbar_y=+0.21,
        colorbar_len=0.4,
        colorscale="gray",
        colorbar_thickness=15,
    ),
    margin=dict(t=40, b=10, l=10, r=10),
)
fig.show()
fig.write_html("minimal_reconstruction_truth_mean_diff_std.html")
fig.write_image("minimal_reconstruction_truth_mean_diff_std.png", scale=4)

# %%
fig = make_subplots(
    rows=1,
    cols=3,
    horizontal_spacing=0.12,
    subplot_titles=("data", "posterior mean", "posterior std."),
)
x, y = tuple(np.mgrid[slice(0.0, 1.0, d * 1j)] for d in dims)
fig.add_trace(go.Heatmap(x=x, y=y, z=data.T, coloraxis="coloraxis"), 1, 1)
fig.add_trace(go.Heatmap(x=x, y=y, z=pm.T, coloraxis="coloraxis"), 1, 2)
fig.add_trace(go.Heatmap(x=x, y=y, z=ps.T, coloraxis="coloraxis3"), 1, 3)
for i in range(1, 4 + 1):
    ya = dict(title="y") if i in (1,) else dict()
    xa = dict(title="x") if i in (1, 2, 3) else dict()
    fig.update_layout(
        {
            f"xaxis{i}": dict(range=(0, 1)) | xa,
            f"yaxis{i}": dict(range=(0, 1), scaleanchor=f"x{i}", scaleratio=1.0) | ya,
        }
    )
fig.update_layout(
    template="plotly_white",
    width=720,
    height=220,
    coloraxis=dict(colorbar_x=0.26, colorscale="viridis", colorbar_thickness=15),
    coloraxis3=dict(colorbar_x=1.0075, colorscale="gray", colorbar_thickness=15),
    margin=dict(t=20, b=5, l=5, r=5),
)
fig.show()
fig.write_html("minimal_reconstruction_data_mean_std.html")
fig.write_image("minimal_reconstruction_data_mean_std.png", scale=4)

# %%
import nifty8 as ift

dims = (128, 128)
dom = ift.RGSpace(dims, distances=tuple(1.0 / d for d in dims))
cfm = ift.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(offset_mean=2, offset_std=(1e-1, 3e-2))
cfm.add_fluctuations(  # Axis over which the kernle is defined
    dom,
    fluctuations=(1.0, 5e-1),
    loglogavgslope=(-3.0, 2e-1),
    flexibility=(1e0, 2e-1),
    asperity=(5e-1, 5e-2),
    prefix="ax1",
)
correlated_field_ift = cfm.finalize()
forward_ift = correlated_field_ift.exp()

lh_ift = ift.PoissonianEnergy(ift.Field(ift.DomainTuple.make(dom), data)) @ forward_ift


# %%
@jax.jit
def generic_lh_metric(lh, p, t):
    return lh.metric(p, t)


pos = samples.pos
lh_met = partial(generic_lh_metric, lh)

pos_ift = {
    k: ift.Field(d, v) if not k.endswith("spectrum") else ift.Field(d, v.T)
    for (k, d), v in zip(lh_ift.domain.items(), samples.pos.tree.values())
}
pos_ift = ift.MultiField.from_dict(pos_ift)

lh_metric_ift = lh_ift(ift.Linearization.make_var(pos_ift, want_metric=True)).metric

# %%
from upy.detective import timeit

# Warm-up and consistency test
out = lh_met(pos, pos).tree
out_ift = lh_metric_ift(pos_ift).val
assert out.keys() == out_ift.keys()
aallclose = partial(np.testing.assert_allclose, atol=1e-10, rtol=1e-11)
for k in out.keys():
    aallclose(out[k], out_ift[k] if not k.endswith("spectrum") else out_ift[k].T)

t = timeit(lambda: jax.block_until_ready(lh_met(pos, pos)))
t_ift = timeit(lambda: lh_metric_ift(pos_ift))
print(f"{t=}, {t_ift=}")
