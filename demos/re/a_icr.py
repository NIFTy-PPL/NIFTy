# %%

import jax
import matplotlib.pyplot as plt
import nifty.re as jft
from jax import random

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

# %% [markdown]
# ## ICR Correlated Field Model
#
# Create a correlated field on an arbitrary grid. There are a few prespecfieid grids
# but it is trivial to define a new grid on top of e.g. `SimpleOpenGrid`. NIFTy
# provides grids for common spaces such as a HEALPIx grid, a logarithmic grid, and a
# `HPLogRGrid` (HEALPix times log radius grid).

# %%
shape = (32, 32)

grid = jft.SimpleOpenGrid(min_shape=shape)
cf = jft.ICRField(
    grid,
    kernel=dict(
        kind="experimental_matern",
        scale=(1.0, 0.2),
        cutoff=(1e-1, 1e-2),
        loglogslope=(-3.0, 0.5),
    ),
)


# %% [markdown]
# ## The likelihood

# %%
signal_response = signal = cf
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, signal_response.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = (
    (noise_cov(jft.ones_like(signal_response.target))) ** 0.5
) * jft.random_like(key, signal_response.target)
data = signal_response_truth + noise_truth

lh = jft.Gaussian(data, noise_cov_inv).amend(signal_response)

# %% [markdown]
# ## The inference

# %%
n_vi_iterations = 6
delta = 1e-4
n_samples = 2

key, k_i, k_o = random.split(key, 3)
# NOTE, changing the number of samples always triggers a resampling
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # point_estimates=("mgcfmcutoff",),
    # Arguments for the conjugate gradient method used to drawing samples
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="linear_resample",
    odir="results_icr",
    resume=False,
)

# %%
post_sr_mean = jft.mean(tuple(signal(s) for s in samples))
to_plot = [
    ("Signal", signal(pos_truth), "im"),
    ("Noise", noise_truth, "im"),
    ("Data", data, "im"),
    ("Reconstruction", post_sr_mean, "im"),
]
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for ax, v in zip(axs.flat, to_plot):
    title, field, tp, *labels = v
    ax.set_title(title)
    if tp == "im":
        end = grid.at(-1).shape  # grid can be arbitrarily deformed
        im = ax.imshow(field.T, cmap="inferno", extent=(0.0, end[0], 0.0, end[1]))
        plt.colorbar(im, ax=ax, orientation="horizontal")
    else:
        ax_plot = ax.loglog if tp == "loglog" else ax.plot
        x = field[0]
        for f in field[1:]:
            ax_plot(x, f, alpha=0.7)
for ax in axs.flat[len(to_plot) :]:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig("results_icr_full_reconstruction.png", dpi=400)
plt.show()

# %%
