#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

#%%
from jax import numpy as jnp
from jax import lax, random
import jax
from jax.config import config
import matplotlib
import matplotlib.pyplot as plt

import nifty8.re as jft

config.update("jax_enable_x64", True)

matplotlib.rcParams['figure.figsize'] = (10, 7)

#%%
dims = (512, )
#datadims = (4,)
loglogslope = 2.
power_spectrum = lambda k: 1. / (k**loglogslope + 1.)
modes = jnp.arange((dims[0] / 2) + 1., dtype=float)
harmonic_power = power_spectrum(modes)
harmonic_power = jnp.concatenate((harmonic_power, harmonic_power[-2:0:-1]))

#%%
correlated_field = lambda x: jft.correlated_field.hartley(
    # x is a signal in fourier space
    # each modes amplitude gets multiplied by it's harmonic_power
    # and the whole signal is transformed back
    harmonic_power * x
)

# %% [markdown]
# signal_response = lambda x: jnp.exp(1. + correlated_field(x))
signal_response = lambda x: correlated_field(x)
# The signal response is $ \vec{d} = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix} \cdot s + \vec{n} $ where $s \in \mathbb{R}$ and $\vec{n} \sim \mathcal{G}(0, N)$
# signal_response = lambda x: jnp.ones(shape=datadims) * x
# ???
noise_cov_inv_sqrt = lambda x: 1.**-1 * x

#%%
# create synthetic data
seed = 43
key = random.PRNGKey(seed)
key, subkey = random.split(key)
# normal random fourier amplitude
pos_truth = random.normal(shape=dims, key=subkey)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
# 1. / noise_cov_inv_sqrt(jnp.ones(dims)) becomes the standard deviation of the noise gaussian
noise_truth = 1. / noise_cov_inv_sqrt(jnp.ones(dims)
                                     ) * random.normal(shape=dims, key=subkey)
data = signal_response_truth + noise_truth

#%%
plt.plot(signal_response_truth, label='signal response')
#plt.plot(noise_truth, label='noise', linewidth=0.5)
plt.plot(data, 'k.', label='noisy data', markersize=4.)
plt.xlabel('real space domain')
plt.ylabel('field value')
plt.legend()
plt.title("signal and data")
plt.show()


#%%
def Gaussian(data, noise_cov_inv_sqrt):
    # Simple but not very generic Gaussian energy
    # primals
    def hamiltonian(primals):
        p_res = primals - data
        # TODO: is this the weighting with noies amplitude thing again?
        l_res = noise_cov_inv_sqrt(p_res)
        return 0.5 * jnp.sum(l_res**2)

    return jft.Likelihood(hamiltonian, )


# negative log likelihood
nll = Gaussian(data, noise_cov_inv_sqrt) @ signal_response

#%%
ham = jft.StandardHamiltonian(likelihood=nll)
ham_gradient = jax.grad(ham)


# %% [markdown]
def plot_mean_and_stddev(ax, samples, mean_of_r=None, truth=False, **kwargs):
    signal_response_of_samples = lax.map(signal_response, samples)
    if mean_of_r == None:
        mean_of_signal_response = jnp.mean(signal_response_of_samples, axis=0)
    else:
        mean_of_signal_response = mean_of_r
    mean_label = kwargs.pop('mean_label', 'sample mean of signal response')
    ax.plot(mean_of_signal_response, label=mean_label)
    std_dev_of_signal_response = jnp.std(signal_response_of_samples, axis=0)
    if truth:
        ax.plot(signal_response_truth, label="truth")
    ax.fill_between(
        jnp.arange(len(mean_of_signal_response)),
        y1=mean_of_signal_response - std_dev_of_signal_response,
        y2=mean_of_signal_response + std_dev_of_signal_response,
        color='grey',
        alpha=0.5
    )
    title = kwargs.pop('title', 'position samples')
    if title is not None:
        ax.set_title(title)
    xlabel = kwargs.pop('xlabel', 'position')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ylabel = kwargs.pop('ylabel', 'signal response')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend(loc='lower right', fontsize=8)


#%%
key, subkey = random.split(key)
initial_position = random.uniform(key=subkey, shape=pos_truth.shape)

sampler = jft.HMCChain(
    potential_energy=ham,
    inverse_mass_matrix=1.,
    position_proto=initial_position,
    step_size=0.05,
    num_steps=128,
)

chain, _ = sampler.generate_n_samples(
    42, initial_position, num_samples=30, save_intermediates=True
)
print(f"acceptance ratio: {chain.acceptance}")

# %%
plot_mean_and_stddev(plt.gca(), chain.samples, truth=True)
plt.title("HMC position samples")
plt.show()

# %% [markdown]
# # NUTS
jft.hmc._DEBUG_STORE = []

sampler = jft.NUTSChain(
    position_proto=initial_position,
    potential_energy=ham,
    inverse_mass_matrix=1.,
    # 0.9193 # integrates to ~3-7, very smooth sample mean
    # 0.8193 # integrates to depth ~22, very noisy sample mean
    step_size=0.05,
    max_tree_depth=17,
)

chain, _ = sampler.generate_n_samples(
    42, initial_position, num_samples=30, save_intermediates=True
)
plt.hist(chain.depths, bins=jnp.arange(sampler.max_tree_depth + 2))
plt.title('NUTS tree depth histogram')
plt.xlabel('tree depth')
plt.ylabel('count')
plt.show()

# %%
plot_mean_and_stddev(plt.gca(), chain.samples, truth=True)
plt.title("NUTS position samples")
plt.show()

# %%
if jft.hmc._DEBUG_FLAG:
    debug_pos = jnp.array(jft.hmc._DEBUG_STORE)[:, 0, :]

    for idx, dbgp in enumerate(debug_pos):
        plt.plot(signal_response(dbgp), label=f'{idx}', alpha=0.1)
    #plt.legend()

    # %%
    debug_pos_x = jnp.array(jft.hmc._DEBUG_STORE)[:, 0, 0]
    debug_pos_y = jnp.array(jft.hmc._DEBUG_STORE)[:, 0, 1]
    for idx, dbgp in enumerate(debug_pos):
        plt.scatter(debug_pos_x, debug_pos_y, s=0.1, color='k')
    #plt.legend()
    plt.show()

# %%[markdown]
# # 1D position and momentum time series
if chain.samples[0].shape == (1, ):
    plt.plot(chain.samples, label='position')
    #plt.plot(momentum_samples, label='momentum', linewidth=0.2)
    #plt.plot(unintegrated_momenta, label='unintegrated momentum', linewidth=0.2)
    plt.title('position and momentum time series')
    plt.xlabel('time')
    plt.ylabel('position, momentum')
    plt.legend()
    plt.show()

# %% [markdown]
# # energy time series
potential_energies = lax.map(ham, chain.samples)
kinetic_energies = jnp.sum(chain.trees.proposal_candidate.momentum**2, axis=1)
#rejected_potential_energies = lax.map(ham, rejected_position_samples)
#rejected_kinetic_energies = jnp.sum(rejected_momentum_samples**2, axis=1)
plt.plot(potential_energies, label='pot')
plt.plot(kinetic_energies, label='kin', linewidth=1)
plt.plot(kinetic_energies + potential_energies, label='total', linewidth=1)
#plt.plot(rejected_potential_energies , label='rejected_pot')
#plt.plot(rejected_kinetic_energies , label='rejected_kin', linewidth=2)
#plt.plot(rejected_kinetic_energies + rejected_potential_energies, label='rejected_total', linewidth=0.2)
plt.title('NUTS energy time series')
plt.xlabel('time')
plt.ylabel('energy')
plt.yscale('log')
plt.legend()
plt.show()

# %% [markdown]
# # Wiener Filter

# jax.linear_transpose for R^\dagger
# square noise_sqrt_inv ... for N^-1
# S is unit due to StandardHamiltonian
# jax.scipy.sparse.linalg.cg for D

# signal_response(s) is only needed for shape of data space
_impl_signal_response_dagger = jax.linear_transpose(signal_response, pos_truth)
signal_response_dagger = lambda d: _impl_signal_response_dagger(d)[0]
# noise_cov_inv_sqrt is diagonal
noise_cov_inv = lambda d: noise_cov_inv_sqrt(noise_cov_inv_sqrt(d))

# signal prior covariance S is assumed to be unit (see StandardHamiltonian)
# the tranposed function wierdly returns a (1,)-tuple which we unpack right here
D_inv = lambda s: s + signal_response_dagger(noise_cov_inv(signal_response(s)))

j = signal_response_dagger(noise_cov_inv(data))

m, _ = jax.scipy.sparse.linalg.cg(D_inv, j)

# %%

# TODO fix labels
plt.plot(signal_response(m), label='signal response of mean')
plt.plot(signal_response_truth, label='true signal response')
plt.legend()
plt.title('Wiener Filter')
plt.show()


# %%
def sample_from_d_inv(key):
    s_inv_key, rnr_key = random.split(key)
    s_inv_smpl = random.normal(s_inv_key, pos_truth.shape)
    # random.normal sample from dataspace and then R^\dagger \sqrt{N^{-1}}
    # jax.eval_shape(signal_response, pos_truth)
    rnr_smpl = signal_response_dagger(
        noise_cov_inv_sqrt(random.normal(rnr_key, signal_response_truth.shape))
    )
    return s_inv_smpl + rnr_smpl


def sample_from_d(key):
    d_inv_smpl = sample_from_d_inv(key)
    # TODO: what to do here?
    smpl, _ = jft.cg(D_inv, d_inv_smpl, maxiter=32)
    return smpl


wiener_samples = jnp.array(
    list(map(lambda key: sample_from_d(key) + m, random.split(key, 30)))
)

# %%
subplots = (3, 1)
fig_height_pt = 541  # pt
#fig_width_pt = 360 # pt
inches_per_pt = 1 / 72.27
fig_height_in = 1. * fig_height_pt * inches_per_pt
fig_width_in = fig_height_in / 0.618 * (subplots[1] / subplots[0])
fig_dims = (fig_width_in, fig_height_in)

fig, (ax_raw, ax_nuts, ax_wiener) = plt.subplots(
    subplots[0], subplots[1], sharex=True, sharey=False, figsize=fig_dims
)

ax_raw.plot(signal_response_truth, label='true signal response')
ax_raw.plot(data, 'k.', label='noisy data', markersize=2.)
#ax_raw.set_xlabel('position')
ax_raw.set_ylabel('signal response')
ax_raw.set_title("signal and data")
ax_raw.legend(fontsize=8)

plot_mean_and_stddev(
    ax_nuts,
    chain.samples,
    truth=True,
    title="NUTS",
    xlabel=None,
    mean_label='sample mean'
)
plot_mean_and_stddev(
    ax_wiener,
    wiener_samples,
    mean_of_r=signal_response(m),
    truth=True,
    title="Wiener Filter",
    mean_label='exact posterior mean'
)

fig.tight_layout()

plt.savefig('wiener.pdf', bbox_inches='tight')
print("final plot saved as wiener.pdf")
