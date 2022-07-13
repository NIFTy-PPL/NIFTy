#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %% [markdown]
# ## What Is This All About?
#
# * Short introduction in how to port code from NIFTy to JAX + NIFTY (jifty)
#   * How to get the JAX expression for a NIFTy operator
#   * How to minimize in jifty
# * Benchmark NIFTy vs jifty

# %%
from collections import namedtuple
from functools import partial
import sys

from jax import jit, value_and_grad
from jax import random
from jax import numpy as jnp
from jax.config import config as jax_config
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
import numpy as np

import nifty8 as ift
import nifty8.re as jft

jax_config.update("jax_enable_x64", True)
# jax_config.update('jax_log_compiles', True)

# %%
filename = "getting_started_nifty2jax{}.png"

position_space = ift.RGSpace([512, 512])
cfm_kwargs = {
    'offset_mean': -2.,
    'offset_std': (1e-5, 1e-6),
    'fluctuations': (2., 0.2),  # Amplitude of field fluctuations
    'loglogavgslope': (-4., 1),  # Exponent of power law power spectrum
    # Amplitude of integrated Wiener process on top of power law power spectrum
    'flexibility': (8e-1, 1e-1),
    'asperity': (3e-1, 1e-3)  # Ragged-ness of integrated Wiener process
}

correlated_field_nft = ift.SimpleCorrelatedField(position_space, **cfm_kwargs)
pow_spec_nft = correlated_field_nft.power_spectrum

signal_nft = correlated_field_nft.exp()
response_nft = ift.GeometryRemover(signal_nft.target)
signal_response_nft = response_nft(signal_nft)

# %% [markdown]
# ## From NIFTy to JAX + NIFTy
#
# By now, we built a beautiful and very complicated forward model. However,
# instead of using vanilla NumPy (i.e. using plain NIFTy), we want to compile
# the forward pass with JAX.

# Note, JAX + NIFTy does not have the concept of domains. Though, it still
# needs to know how large the parameter space is. This can either be provided
# via an initializer or via a pytree containing the shapes and dtypes. Thus, in
# addition to extracting the JAX call, we also need to extract the parameter
# space on which this call should act.

# %%
pow_spec = pow_spec_nft.jax_expr
signal = signal_nft.jax_expr
# Convenience method to get JAX expression and domain
signal_response: jft.Model = ift.nifty2jax.convert(signal_response_nft, float)

noise_cov = 0.5**2

# %%
key = random.PRNGKey(42)

key, sk = random.split(key)
synth_pos = jft.Field(signal_response.init(sk))
data = synth_signal_response = signal_response(synth_pos)
data += jnp.sqrt(noise_cov) * random.normal(sk, shape=data.shape)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
im = axs.flat[0].imshow(synth_signal_response)
fig.colorbar(im, ax=axs.flat[0])
im = axs.flat[1].imshow(data)
fig.colorbar(im, ax=axs.flat[1])
fig.tight_layout()
plt.show()

# %%
lh = jft.Gaussian(data, noise_cov_inv=lambda x: x / noise_cov) @ signal_response
ham = jft.StandardHamiltonian(likelihood=lh).jit()
ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))
MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name")
)

key, subkey = random.split(key)
pos = pos_init = 1e-2 * jft.Field(signal_response.init(subkey))

# %% [markdown]
# Let's do a simple MGVI minimization. Note, while this might look very similar
# to plain NIFTy, the convergence criteria and various implementation details
# are very different. Thus, timing the minimization and comparing it to NIFTy
# most probably leads to very screwed results. It is best to only compare a
# single value-and-gradient call in both implementations for the purpose of
# creating a benchmark.

# %%
n_mgvi_iterations = 10
n_samples = 2
absdelta = 0.1
n_newton_iterations = 15

# Minimize the potential
key, *sk = random.split(key, 1 + n_mgvi_iterations)
for i, subkey in enumerate(sk):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    mg_samples = MetricKL(
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=True,
        linear_sampling_name=None,
        linear_sampling_kwargs={"absdelta": absdelta / 10.}
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        pos,
        method="newton-cg",
        options={
            "fun_and_grad": partial(ham_vg, primals_samples=mg_samples),
            "hessp": partial(ham_metric, primals_samples=mg_samples),
            "absdelta": absdelta,
            "maxiter": n_newton_iterations
        }
    )
    pos = opt_state.x
    msg = f"Post MGVI Iteration {i}: Energy {mg_samples.at(pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)

# %%
# The minimization is done now and we can have a look at the result.
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
im = axs.flat[0].imshow(synth_signal_response)
fig.colorbar(im, ax=axs.flat[0])
im = axs.flat[1].imshow(data)
fig.colorbar(im, ax=axs.flat[1])
sr_pm = mg_samples.at(pos).mean(signal_response)
im = axs.flat[2].imshow(sr_pm)
fig.colorbar(im, ax=axs.flat[2])
fig.tight_layout()
plt.show()

# %% [markdown]
# Awesome! We have seen now how a model can be translated to JAX. By doing so
# we were able to use such convenient transformation like `jit` and
# `value_and_grad` from JAX. Thus, we can start using higher order derivatives
# and other useful JAX features like `vmap` and `pmap`. Last but certainly not
# least, we can now also let our code run on the GPU.

# %% [markdown]
# ## Performance
#
# The driving force behind all of this is of course speed! So let's validate
# that translating the model to JAX actually is faster.

# %%
Timed = namedtuple("Timed", ("time", "number"), rename=True)


def timeit(stmt, setup=lambda: None, number=None):
    import timeit

    if number is None:
        number, _ = timeit.Timer(stmt).autorange()

    setup()
    t = timeit.timeit(stmt, number=number) / number
    return Timed(time=t, number=number)


r = jft.Field(signal_response.init(random.PRNGKey(54)))

r_nft = ift.makeField(signal_response_nft.domain, r.val)
data_nft = ift.makeField(signal_response_nft.target, data)
lh_nft = ift.GaussianEnergy(
    data_nft,
    inverse_covariance=ift.ScalingOperator(data_nft.domain, 1. / noise_cov)
) @ signal_response_nft
ham_nft = ift.StandardHamiltonian(lh_nft)

_ = ham(r)  # Warm-Up
t = timeit(lambda: ham(r).block_until_ready())
t_nft = timeit(lambda: ham_nft(r_nft))

print(f"W/  JAX :: {t}")
print(f"W/O JAX :: {t_nft}")

# %%
# For about 2e+5 #parameters the FFT starts to dominate in the computation and
# NumPy-based NIFTy is about as fast as JAX-based NIFTy. Thus, we should not
# have expected to gain much performance for our model at hand.

# So far so good but are we really sure that this is doing the same thing. To
# validate the result of our model in JAX, let's transfer our synthetic
# position to plain NIFTy and run the model there again.

sp = ift.makeField(signal_response_nft.domain, synth_pos.val)
np.testing.assert_allclose(
    signal_response_nft(sp).val, signal_response(synth_pos)
)

# %% [markdown]
# For smaller models or models where the FFT does not dominate JAX-based NIFTy
# should always have an edge over NumPy based NIFTy. The difference in
# performance can range from only a couple of double digit percentages for
# \approx 1e+5 #parameters to many orders of magnitudes. For example with 65536
# #parameters JAX-based NIFTy should be 2-3 times faster.

# We can show this more explicitly with a proper benchmark. In the following we
# will instantiate models of various shapes and time the JAX version against
# the NumPy version. Instead of testing solely a single forward pass, we will
# compare a full evaluation of the model and its gradient.


# %%
def get_lognormal_model(shapes, cfm_kwargs, data_key, noise_cov=0.5**2):
    import warnings

    position_space = ift.RGSpace(shapes)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", category=UserWarning, message="no JAX"
        )
        correlated_field_nft = ift.SimpleCorrelatedField(
            position_space, **cfm_kwargs
        )
        signal_nft = correlated_field_nft.exp()
        response_nft = ift.GeometryRemover(signal_nft.target)
        signal_response_nft = response_nft(signal_nft)

    signal_response: jft.Model = ift.nifty2jax.convert(
        signal_response_nft, float
    )

    sk_signal, sk_noise = random.split(data_key)
    synth_pos = jft.Field(signal_response.init(sk_signal))
    data = signal_response(synth_pos)
    data += jnp.sqrt(noise_cov) * random.normal(sk_noise, shape=data.shape)

    noise_cov_inv = 1. / noise_cov
    noise_std_inv = jnp.sqrt(noise_cov_inv)
    lh = jft.Gaussian(
        data,
        noise_cov_inv=lambda x: noise_cov_inv * x,
        noise_std_inv=lambda x: noise_std_inv * x
    ) @ signal_response
    ham = jft.StandardHamiltonian(likelihood=lh)
    ham_vg = value_and_grad(ham)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", category=UserWarning, message="no JAX"
        )
        data_nft = ift.makeField(signal_response_nft.target, data)
        noise_cov_inv_nft = ift.ScalingOperator(data_nft.domain, 1. / noise_cov)
        lh_nft = ift.GaussianEnergy(
            data_nft, inverse_covariance=noise_cov_inv_nft
        ) @ signal_response_nft
        ham_nft = ift.StandardHamiltonian(lh_nft)

    def ham_vg_nft(x):
        x = x.val if isinstance(x, jft.Field) else x
        x = ift.makeField(ham_nft.domain, x)
        x = ift.Linearization.make_var(x)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", category=UserWarning, message="no JAX"
            )
            res = ham_nft(x)
        one_nft = ift.Field(ift.DomainTuple.make(()), 1.)
        bwd = res.jac.adjoint_times(one_nft)
        return (res.val.val, bwd.val)

    aux = {
        "synthetic_position": synth_pos,
        "hamiltonian_nft": ham_nft,
        "hamiltonian": ham,
        "signal_response_nft": signal_response_nft,
        "signal_response": signal_response,
    }
    return ham_vg, ham_vg_nft, aux


get_ln_mod = partial(
    get_lognormal_model, cfm_kwargs=cfm_kwargs, data_key=key, noise_cov=0.5**2
)

dimensions_to_test = [
    (256, ), (512, ), (1024, ), (256**2, ), (512**2, ), (128, 128), (256, 256),
    (512, 512), (1024, 1024), (2048, 2048)
]
for dims in dimensions_to_test:
    h, h_nft, aux = get_ln_mod(dims)
    r = aux["synthetic_position"]
    h = jit(h)
    _ = h(r)  # Warm-Up

    np.testing.assert_allclose(h(r)[0], h_nft(r)[0])
    ift.myassert(all(tree_map(np.allclose, h(r)[1].val, h_nft(r)[1]).values()))
    ti = timeit(lambda: h(r)[0].block_until_ready())
    ti_n = timeit(lambda: h_nft(r))

    print(
        f"Shape {str(dims):>16s}"
        f" :: JAX {ti.time:4.2e}"
        f" :: NIFTy {ti_n.time:4.2e}"
        f" ;; ({ti.number:6d}, {ti_n.number:<6d} loops respectively)"
    )

# %% [markdown]
# | Shape                  | JAX          | NIFTy          | Loops respectively                  |
# |:-----------------------|:-------------|:---------------| -----------------------------------:|
# | Shape           (256,) | JAX 2.58e-05 | NIFTy 6.96e-03 | ( 10000, 50     loops respectively) |
# | Shape           (512,) | JAX 3.90e-05 | NIFTy 7.14e-03 | ( 10000, 50     loops respectively) |
# | Shape          (1024,) | JAX 6.33e-05 | NIFTy 6.97e-03 | (  5000, 50     loops respectively) |
# | Shape         (65536,) | JAX 5.41e-03 | NIFTy 1.42e-02 | (    50, 20     loops respectively) |
# | Shape        (262144,) | JAX 2.72e-02 | NIFTy 4.41e-02 | (    10, 5      loops respectively) |
# | Shape       (128, 128) | JAX 5.07e-04 | NIFTy 7.00e-03 | (   500, 50     loops respectively) |
# | Shape       (256, 256) | JAX 3.74e-03 | NIFTy 1.01e-02 | (   100, 20     loops respectively) |
# | Shape       (512, 512) | JAX 1.53e-02 | NIFTy 2.33e-02 | (    20, 10     loops respectively) |
# | Shape     (1024, 1024) | JAX 7.80e-02 | NIFTy 7.72e-02 | (     5, 5      loops respectively) |
# | Shape     (2048, 2048) | JAX 3.21e-01 | NIFTy 3.52e-01 | (     1, 1      loops respectively) |

# For small problems JAX-based NIFTy is significantly faster than the NumPy
# based one. For really small problems it is more than 200 times faster. This
# is because the overhead from python can be significantly reduced with JAX
# since most of the heavy-lifting happens without going back to python.

# Notice, how above a certain threshold, here 2e+5, the NumPy-based NIFTy and
# JAX-bassed NIFTy start to perform similarly well because the performance of
# the FFT is the sole bottle neck.
