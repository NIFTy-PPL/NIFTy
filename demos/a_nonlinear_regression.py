#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %% [markdown]
# # Demonstration of a non-linear regression using NIFTy.re

# %%
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
import numpy as np
from jax import random
from functools import partial
import operator as op

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

# %%
seed = 42
key = random.PRNGKey(seed)


class NonLinearRegression(jft.Model):
    def __init__(self, slope_mean, slope_std, intercept_min, intercept_max, x):
        self.slope = jft.LogNormalPrior(slope_mean, slope_std, name="slope")
        self.intercept = jft.UniformPrior(
            intercept_min, intercept_max, name="intercept"
        )
        self.x = x
        super().__init__(init=self.slope.init | self.intercept.init)

    def __call__(self, xi, *, x=None):
        x = x if x is not None else self.x
        return x * self.slope(xi) + self.intercept(xi)


key, sk = random.split(key)
x = random.uniform(sk, (50,), float, -4.0, 4.0)
nlr = NonLinearRegression(3.0, 2.0, -5.0, 5.0, x)

# %%
noise_std = 5

key_tr, key_n = random.split(random.PRNGKey(31415))
y = nlr(nlr.init(key_tr)) + noise_std * random.normal(key_n, x.shape, x.dtype)

# %%
fig, ax = plt.subplots()
ax.plot(x, y, color="dodgerblue", linestyle="None", marker=".", markersize=8)
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()
fig.savefig("nonlinear_regression_data.png")
plt.show()

# %%
lh = jft.Gaussian(y, noise_std_inv=partial(op.mul, 1.0 / noise_std)).amend(nlr)

# %%
key, ki, ko = random.split(key, 3)

delta = 1e-4
samples_opt, st = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(ki)),
    key=ko,
    n_total_iterations=5,
    n_samples=12,
    draw_linear_kwargs=dict(
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0),
    ),
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(xtol=delta, cg_kwargs=dict(name=None))
    ),
    kl_kwargs=dict(
        minimize_kwargs=dict(name="M", xtol=delta, cg_kwargs=dict(name=None))
    ),
    sample_mode="nonlinear_resample",
)

# %%
x_p = np.linspace(x.min(), x.max(), 500)
y_unnoised_samples = jax.vmap(partial(nlr, x=x_p))(samples_opt.samples)

fig, ax = plt.subplots()
ax.plot(x, y, color="dodgerblue", linestyle="None", marker=".", markersize=8)
ax.plot(x_p, y_unnoised_samples.mean(axis=0), color="black")
qs = (0.16, 0.84)
ax.fill_between(
    x_p, *np.quantile(y_unnoised_samples, qs, axis=0), color="gray", alpha=0.3
)
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()
fig.savefig("nonlinear_regression_posterior.png", dpi=400)
plt.show()
