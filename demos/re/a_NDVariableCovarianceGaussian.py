# %%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

import nifty.re as jft
from nifty.re.tree_math.util import sqrtm

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.key(seed)

# %% [markdown]
# ## N-dimensional, non-diagonal, variable-covariance Gaussian likelihood
#
# This example demonstrates how to use the `NDVariableCovarianceGaussian` likelihood.
# You can choose between a parametrisation using the covariance matrix and its inverse,
# the precision matrix. We strongly recommend parametrising the matrix in its eigenspace
# and build the rotated covariance as needed (see below for an example).

# %% [markdown]
# ## Create synthetic data

# %%

N = 100
mean = jnp.array([1.0, -1.0])
std = [1.0, 0.5]
corr = 0.75
cov_true = jnp.array(
    [
        [std[0] * std[0], std[0] * std[1] * corr],
        [std[0] * std[1] * corr, std[1] * std[1]],
    ]
)
key, key1 = random.split(key, 2)
random_numbers = random.normal(key1, shape=(N, 2))
data = jnp.einsum("ij,kj->ki", sqrtm(cov_true), random_numbers)
data += mean[None, :]
assert data.shape == (N, 2)

# %% [markdown]
# ## Build likelihood and forward-model

# %%


class Forward(jft.Model):
    def __init__(self):
        self.mean = jft.NormalPrior(0.0, 5.0, name="mean", shape=(1, 2))
        self.std_0 = jft.LogNormalPrior(1.0, 1.0, name="std_0", shape=())
        self.std_1 = jft.LogNormalPrior(1.0, 1.0, name="std_1", shape=())
        self.angle = jft.UniformPrior(0.0, jnp.pi / 2, name="angle", shape=())
        super().__init__(
            init=self.mean.init | self.std_0.init | self.std_1.init | self.angle.init
        )

    def __call__(self, xi):
        mean = self.mean(xi)
        mean = jnp.broadcast_to(mean, (N, 2))
        std_0 = self.std_0(xi)
        std_1 = self.std_1(xi)
        angle = self.angle(xi)
        diag = jnp.array([[std_0, 0], [0, std_1]])
        rot = jnp.array(
            [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
        )
        cov = rot @ diag @ rot.T
        cov = jnp.broadcast_to(cov[None, ...], (N, 2, 2))
        return (mean, cov)


fwd_model = Forward()
lh = jft.NDVariableCovarianceGaussian(data).amend(fwd_model)

# %% [markdown]
# ## Run inference

# %%
key, key2, key3 = random.split(key, 3)
samps, state = jft.optimize_kl(
    lh,
    1.0e-2 * jft.Vector(lh.init(key2)),
    key=key3,
    n_total_iterations=5,
    n_samples=10,
    sample_mode="nonlinear_resample",
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN", xtol=1.0e-4, cg_kwargs=dict(name="SNCG"), maxiter=10
        ),
    ),
)

# %% [markdown]
# ## Plot results

# %%


def plot_mean_cov(mean, cov, lbl, color, alpha, *, ax):
    v, U = jnp.linalg.eigh(cov)
    theta = jnp.arctan2(U[1, 1], U[0, 1])
    v = jnp.sqrt(v)
    ell = plt.matplotlib.patches.Ellipse(
        mean,
        3.0 * v[1],
        3.0 * v[0],
        angle=theta * 180 / jnp.pi,
        edgecolor=color,
        fill=False,
        alpha=alpha,
        label=lbl,
    )
    ax.add_patch(ell)
    ax.plot(mean[0], mean[1], "o", color=color, markersize=2, alpha=alpha)


fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], marker=".", s=5, label="data")
# Plot samples
for i, samp in enumerate(samps):
    mean = fwd_model(samp)[0][0]
    cov = fwd_model(samp)[1][0]
    lbl = "Samples 3 sigma" if i == 0 else ""
    plot_mean_cov(mean, cov, lbl, "red", alpha=0.5, ax=ax)
plot_mean_cov(mean, cov_true, "Truth 3 sigma", "k", 1.0, ax=ax)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
fig.savefig("results_NDVCG.png", dpi=400)
plt.show()
