# %% [markdown]
# # Wiener Filter
#
# NIFTy primarily uses Variational Inference to approximate posterior
# distributions. However, for linear models with Gaussian likelihoods,
# variational inference is not required, as the posterior can be computed
# directly for this special case. The formula for directly computing the
# posterior of a linear model is called the Wiener Filter. NIFTy also
# implements the Wiener Filter, and this notebook demonstrates how to use it.
# It provides a brief introduction to the mathematical background
# of the Wiener Filter. For a complete derivation, see, for example,
# [this script](https://wwwmpa.mpa-garching.mpg.de/~ensslin/lectures/Files/ScriptIT\&IFT.pdf).
#

# %% [markdown]
# Before we start with the Wiener filter implementation in `NIFTy.re`, let us
# quickly review the mathematical background.
#
# The Wiener Filter makes the following assumptions:
# * A linear measurement response $R$ relating the signal $s$ to the
# measured data $d = Rs+n$ and the noise $n$.
# * A Gaussian prior on the signal $s$. For the simple Wiener Filter, the
# Gaussian needs to have mean 0. The covariance of the Gaussian prior is
# denoted by $S$, i.e., $\mathcal{P}(s) = \mathcal{G}(s,S)$.
# * Analogously, the noise $n$ also needs to be Gaussian distributed:
# $\mathcal{P}(n) = \mathcal{G}(n,N)$.
#
# As a consequence of the Gaussian noise, the likelihood is also a Gaussian
# distribution:
#
# $$ \mathcal{P}(d|s) = \mathcal{G}(d - Rs,N). $$
#
# The joint distribution $\mathcal{P}(s,d)$, which is the product of the
# likelihood and the prior, can be written as:
#
# $$ \mathcal{P}(s,d) = \mathcal{P}(d|s) \, \mathcal{P}(s) = \mathcal{G}(d - Rs,N) \mathcal{G}(s,S). $$
#
# The product of two Gaussian distributions remains Gaussian, so the
# posterior is also a Gaussian distribution. Via direct calculation, it can be
# shown that the posterior distribution can be written as:
#
# $$ \mathcal{P}(s|d) = \frac{\mathcal{P}(s,d)}{\mathcal{P}(d)} = \mathcal{G}(s-m,D), $$
#
# with the posterior covariance being:
#
# $$ D^{-1} = S^{-1} + R^\dagger N^{-1} R, $$
#
# and the mean being:
#
# $$ m = DR^\dagger N^{-1} d. $$
#
# In some applications, $D$ is called the information propagator and
# $j = R^\dagger N^{-1} d$ the information source.

# %% [markdown]
# To summarize, under the assumptions of the Wiener filter, the posterior is a
# Gaussian distribution, and the posterior mean $m$ can be computed via
# $m = F_W d$ with:
#
# $$ F_W = D R^\dagger N^{-1} = \big( S^{-1} + R^\dagger N^{-1} R \big)^{-1} R^\dagger N^{-1}. $$
#
# Through a short calculation, it can be shown that the Wiener Filter is equivalent
# to the optimal linear filter:
#
# $$  F_L = S R^\dagger \big( R S R^\dagger + N \big)^{-1}, $$
#
# which is designed to minimize the expected root mean square between the signal
# $s$ and the posterior mean $m$.

# %% [markdown]
# As introduced in previous notebooks, NIFTy builds on generative models mapping
# from a standard normal distribution to the desired prior signal distribution.
# For this reason, the prior covariance $S$ is (in `NIFTy.re`) always given by
# the unit matrix. The generative model mapping the latent parameters to the
# desired signal prior distribution becomes part of the response $R$. Thus, the
# Wiener filter for NIFTy reads:
#
# $$ F_W = \left( \mathbb{1} + R^\dagger N^{-1} R \right)^{-1} R^\dagger N^{-1}. $$

# %% [markdown]
# ## Wiener Filter in NIFTy

# %% [markdown]
# To demonstrate the Wiener filter, we will continue the Gaussian process example
# from the [previous notebook](2_gaussian_processes). To recapitulate, in
# the previous notebook we coded a generative Gaussian process model of the form:
#
# $$ s = \text{HT} \, A \xi, $$
#
# with $\text{HT}$ being the Hartley transform and $A$ the amplitude spectrum.
# Inserting this generative model into the measurement equation, we get:
#
# $$ d = R_{I} s + n = R_{I}\text{HT} \,  A \xi + n = R \xi + n, $$
#
# with $R_{I}$ being the response of the instrument and $R$ the combination of
# generative model and instrument response. The following code block contains the
# generative model from the previous notebook.

# %%
import nifty.re as jft

import jax
import jax.numpy as jnp
import jax.random as random

# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100

jax.config.update("jax_enable_x64", True)
seed = 42
key = random.PRNGKey(seed)


dims = 100
distances = 0.01
grid = jft.correlated_field.make_grid(
    dims, distances=distances, harmonic_type="fourier"
)

axis = jnp.arange(0, dims * distances, distances)


def amplitude_spectrum(k):
    return 2.5 / (5 + k**2)


k_lengths = grid.harmonic_grid.mode_lengths
amplitudes = amplitude_spectrum(k_lengths)
sqrt_hamonic_cov = amplitudes[grid.harmonic_grid.power_distributor]


class FixedPowerCorrelatedField(jft.Model):
    def __init__(self, sqrt_hamonic_cov, grid):
        self.sqrt_hamonic_cov = sqrt_hamonic_cov
        self.ht = jft.correlated_field.hartley
        self.harmonic_dvol = 1 / grid.total_volume

        super().__init__(
            domain=jax.ShapeDtypeStruct(shape=grid.shape, dtype=jnp.float64)
        )

    def __call__(self, x):
        return self.harmonic_dvol * self.ht(self.sqrt_hamonic_cov * x)


signal = FixedPowerCorrelatedField(sqrt_hamonic_cov, grid)

# %% [markdown]
# For an initial example, we will assume a perfect instrument, meaning that we set
# $R_\text{I} = \mathbb{1}$ to the unit matrix. In later examples, we will
# consider a more complicated response function. Thus, for now, the signal
# response $Rs$ is equal to $s$.

# %%
signal_response = signal

# %% [markdown]
# ### Synthetic data generation

# %% [markdown]
# In the previous notebooks, we loaded a data file as you would do in an
# application to real data. In this notebook, we will generate synthetic data by
# drawing prior samples with the following procedure. First, we draw a standard
# normal sample on the input domain of our generative model. We name this sample
# `pos_truth` as it will be the ground truth. We pass this latent space sample
# through our model to compute the corresponding signal response.

# %%
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, signal_response.domain)
signal_response_truth = signal_response(pos_truth)

# %% [markdown]
# To generate synthetic data consistent with the measurement equation
# $d = Rs + n$, we draw a random noise sample. For the example in this notebook,
# we will assume that the noise is uncorrelated between the data points and has a
# standard deviation of $0.3$.

# %%
noise_std = 0.3
noise_cov = lambda x: noise_std**2 * x
noise_cov_inv = lambda x: noise_std**-2 * x
key, subkey = random.split(key)
noise_truth = noise_std * jft.random_like(key, signal_response.target)

# %% [markdown]
# Having now random realizations for $s$ and $n$, we can construct the synthetic
# data $d = Rs + n$.

# %%
data = signal_response_truth + noise_truth

# %% [markdown]
# Let's plot the synthetic data together with the ground truth.

# %%
plt.plot(axis, data, ".", color="tab:blue", label="Data")
plt.plot(axis, signal(pos_truth), color="tab:orange", label="Ground truth")
plt.legend()
plt.show()

# %% [markdown]
# `wiener_filter_posterior` does not directly take data as an input parameter,
# but rather the likelihood connected to the data. As we assume a linear
# measurement model with Gaussian signal and noise, the likelihood is a
# Gaussian of the following form:
#
# $$  \mathcal{P}(d|s) = \mathcal{G}(d - Rs,N) $$

# %%
lh = jft.Gaussian(data, noise_cov_inv).amend(signal_response)

# %% [markdown]
# The following parameters are most important for the `wiener_filter_posterior`.
# Additional parameters for further options are documented in the API reference.
# * `likelihood`: The NIFTy likelihood of the inference problem contains the
# data, the noise covariance, and the signal response.
# * `position`: This parameter is optional and is only important for non-linear
# models or non-Gaussian likelihoods. If the model is non-linear, the model
# gets linearized around this position to apply the Wiener filter formulas.
# However, for non-linear models, the Wiener filter is no longer exact, and you
# should consider using variational inference discussed in the
# [this notebook](1_inference).
# * `key`: JAX random number generation key used to generate keys for drawing
# posterior samples.
# * `n_samples`: Number of posterior samples to draw.
# * `draw_linear_kwargs`: Specifies the parameters used for the conjugate
# gradient scheme to obtain the Wiener filter solution. For numerical
# reasons, `NIFTy` does not directly invert $D^{-1}$, but instead applies $D$
# via conjugate gradient minimization. For more information on how to set the
# parameters of the conjugate gradient, see the
# [inference notebook](1_inference).
# * `optimize_for_linear`: Enables numerical optimization for linear models.
#
# The `wiener_filter_posterior` method returns a set of samples from the
# posterior distribution. The mean of the samples equals the posterior mean
# computed via the Wiener filter formulas, as the samples are drawn
# synthetically around the mean.

# %%
# The Wiener filter

delta = 1e-6
key, k_w = random.split(key)
samples, _ = jft.wiener_filter_posterior(
    likelihood = lh,
    key=k_w,
    n_samples=20,
    draw_linear_kwargs=dict(
        cg_name=None,
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
)

# %% [markdown]
# With the drawn `samples` around the Wiener filter solution, we can now
# calculate the posterior statistics, its mean and standard deviation of the
# `signal`, and compare it to the ground truth.

# %%
post_mean, post_std = jft.mean_and_std(tuple(signal(s) for s in samples))

# %%
plt.plot(axis, data, ".", color="tab:blue", label="Data")
plt.plot(axis, signal(pos_truth), color="tab:orange", label="Ground truth")
plt.plot(axis, post_mean, label="Posterior mean", color="tab:green")
plt.fill_between(
    axis,
    post_mean - post_std,
    post_mean + post_std,
    color="tab:green",
    alpha=0.4,
    label=r"Posterior std (1$\sigma$)",
)
plt.legend()
plt.show()

# %% [markdown]
# The plot above shows, besides the data and the ground truth signal, the
# Wiener filter reconstruction of the `signal` together with its $1\sigma$
# band. The Wiener filter solution mostly captures the ground truth signal of the
# synthetic data.

# %% [markdown]
# ## Wiener Filter on incomplete data
#
# As a second example, let's change the signal response to a more complicated
# function. The signal response is no longer a unit matrix, but a point-wise
# multiplication with a sensitivity. For a given slice of the signal array,
# we set the sensitivity to 0, meaning that the data carries no information about
# the signal in this region.
#
# In the NIFTy code, we define a class `SignalResponse` to multiply the signal
# with the `sensitivity` array.


# %%
class SignalResponse(jft.Model):
    def __init__(self, signal, sensitivity):
        self.signal = signal
        self.sensitivity = sensitivity
        super().__init__(domain=signal.domain)

    def __call__(self, x):
        return self.signal(x) * self.sensitivity


sensitivity = jnp.ones(shape=dims)
sensitivity = sensitivity.at[25:80].set(0.0)
signal_response = SignalResponse(signal, sensitivity)

# %% [markdown]
# As before, we generate synthetic data consistent with the measurement equation
# $d = Rs+n$.

# %%
noise_std = 0.1
noise_cov = lambda x: noise_std**2 * x
noise_cov_inv = lambda x: noise_std**-2 * x

signal_truth = signal(pos_truth)

signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_true = noise_std * jft.random_like(subkey, signal_response.domain)
data = signal_response_truth + noise_true

# %% [markdown]
# Again, we assume Gaussian noise to construct the likelihood for the problem
# and call the `wiener_filter_posterior` function. We compute the posterior mean
# and standard deviation for the reconstructed `signal`.

# %%
lh = jft.Gaussian(data, noise_cov_inv).amend(signal_response)

key, subkey = random.split(key)
samples, info = jft.wiener_filter_posterior(
    lh,
    key=subkey,
    n_samples=20,
    draw_linear_kwargs=dict(cg_name=None, cg_kwargs=dict(absdelta=delta, maxiter=100)),
)

post_mean, post_std = jft.mean_and_std(tuple(signal(s) for s in samples))

# %% [markdown]
# As before, we plot the reconstruction alongside the ground truth of the signal
# and the data. Furthermore, we shade the area where the `sensitivity` is zero
# in grey.

# %%
plt.axvspan(25, 80, color="gray", alpha=0.25, label="Missing data area")
plt.plot(jnp.arange(len(data)), signal_truth, color="tab:orange", label="Ground Truth")
plt.plot(jnp.arange(len(data)), post_mean, color="tab:green", label="Posterior mean")
plt.fill_between(
    range(post_mean.size),
    (post_mean - post_std),
    (post_mean + post_std),
    color="tab:green",
    alpha=0.4,
    label=r"Posterior stddev (1$\sigma$)",
)

data = data.at[25:80].set(jnp.nan)  # Remove masked data from plotting
plt.scatter(jnp.arange(len(data)), data, label="Noisy Data", s=15, color="tab:blue")
plt.legend()
plt.show()

# %% [markdown]
# Again, we can see that the Wiener filter solution recovers the signal in
# regions where data is available. We can see that it interpolates in the area where
# we have no data, given the fixed covariance structure. Furthermore, the Wiener
# filter solution assigns greater posterior uncertainty to the regions
# unconstrained by the data.

# %% [markdown]
# ## Summary
#
# This notebook introduced the Wiener filter and showed how to apply the Wiener
# filter formulas in `NIFTy.re` using the `wiener_filter_posterior` function.
# Furthermore, the limitations of the Wiener filter for nonlinear models were
# highlighted. Additionally, the notebook introduced the idea of generating
# synthetic data from the prior models. Reconstructing synthetically generated
# data is not only handy for introductory examples but is also a good test for
# new models and inference algorithms.
