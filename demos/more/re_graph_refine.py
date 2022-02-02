#!/usr/bin/env python3

import jax
from jax import numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np

import nifty8.re as jft


def get_kernel(layer_weights, depth, n_samples=100):
    xi = {"offset": 0., "layer_weights": layer_weights}
    kernel = np.zeros(2**depth)
    for _ in range(n_samples):
        xi["excitations"] = jnp.array(rng.normal(size=(2**depth, )))
        r = fwd(xi)
        for i in range(r.size):
            kernel[i] += np.mean(r * np.roll(r, i))
    kernel /= len(n_samples)
    return kernel


def fwd(xi):
    offset = xi["offset"]
    excitations = xi["excitations"]
    layer_wgt = xi["layer_weights"]

    kernel = jnp.array([1., 2., 1.])
    kernel /= kernel.sum()
    layers = [excitations]
    while layers[-1].size > 1:
        lvl = layers[-1]
        if layers[-1].size > 2:
            lvl = jnp.convolve(lvl, kernel, mode="same")
        layers += [0.5 * lvl.reshape(-1, 2).sum(axis=1)]
    if len(layers) != len(layer_wgt):
        raise ValueError()

    field = offset
    for d, (wgt, lvl) in enumerate(zip(layer_wgt, layers)):
        field += wgt * jnp.repeat(lvl, 2**d)

    return field


# %%
rng = np.random.default_rng(42)
depth = 8

for _ in range(10):
    layer_weights = jnp.array(rng.normal(size=(depth + 1, )))
    layer_weights = jnp.exp(0.1 * layer_weights)  #lognomral
    kernel = get_kernel(layer_weights, depth, n_samples=30)
    plt.plot(kernel)
plt.show()

# %%
spec = np.fft.fft(kernel)
plt.plot(spec)
plt.yscale("log")
plt.xscale("log")
plt.show()

# %%

rng = np.random.default_rng(42)
depth = 12
xi = {
    "offset": 0.,
    "excitations": jnp.array(rng.normal(size=(2**depth, ))),
    # "layer_weights": jnp.exp(+jnp.array(rng.normal(size=(depth + 1, )))),
    "layer_weights": jnp.exp(0.1 * jnp.arange(depth + 1, dtype=float)),
    # "layer_weights": jnp.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,]),
}

plt.plot(xi["excitations"], label="excitations", alpha=0.6)
plt.plot(fwd(xi), label="Forward Model", alpha=0.6)
plt.legend()
plt.show()

# %%
cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
cf_fl = {
    "fluctuations": (1e-1, 5e-3),
    "loglogavgslope": (-1.5, 1e-2),
    "flexibility": (5e-1, 1e-1),
    "asperity": (5e-1, 5e-2),
    "harmonic_domain_type": "Fourier"
}
dims = jnp.array([2**depth])

cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(dims, distances=1. / dims.shape[0], **cf_fl, prefix="ax1")
correlated_field, ptree = cfm.finalize()
key = random.PRNGKey(42)

pos_truth = jft.random_like(key, ptree)
plt.plot(correlated_field(pos_truth))
plt.show()

# %%
d = correlated_field(pos_truth)
lh = jax.jit(
    lambda x: ((d - fwd(x))**2).sum() +
    sum([(el**2).sum() for el in jax.tree_util.tree_leaves(x)])
)
print(lh(xi))

# %%
opt_state = jft.minimize(
    lh,
    jft.Field(xi),
    method="newton-cg",
    options={
        "name": "N",
        "absdelta": 0.1,
        "maxiter": 30
    }
)

# %%
plt.plot(correlated_field(pos_truth), label="truth")
plt.plot(fwd(opt_state.x), label="reconstruction")
plt.legend()
plt.show()

# %%
pos_rec = opt_state.x.val.copy()
pos_rec["layer_weights"] = pos_rec["layer_weights"].at[:-8].set(0.)

pos_truth = jft.random_like(key, ptree)
plt.plot(correlated_field(pos_truth), alpha=0.7, label="truth")
plt.plot(fwd(opt_state.x), alpha=0.7, label="reconstruction")
plt.plot(fwd(pos_rec), alpha=0.7, label="reconstruction coarse")
plt.legend()
plt.show()
