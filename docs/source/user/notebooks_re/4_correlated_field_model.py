# %% [markdown]
# # Gaussian Processes with a variable kernel
#
# In this notebook we explain the implementation of Gaussian processes with
# variable power spectra in `NIFTy`. In the previous Gaussian process
# [notebook](2_gaussian_processes) we implemented a generative model
# for Gaussian processes with a fixed correlation kernel, or fixed power
# spectrum assuming that it is known. However, for many inference setups, this
# is not the case and the correlation structures are a priori unknown.
#
# To overcome this limitation, we introduce a Gaussian process model where the
# exact power spectrum can be a priori unknown. The basic underlying idea is to
# set up a generative model for power spectra and make this power spectrum model
# part of the overall generative model for the Gaussian process.
#
# Let us begin with a brief recap of the generative Gaussian process model with
# a fixed power spectrum from the previous notebook.

# %% [markdown]
# ## Fixed Power Spectrum Model
#
# In `NIFTy`, we leverage the Wiener-Khinchin theorem to build our Gaussian
# process models. The Wiener-Khinchin theorem states that the correlation
# structure for a statistically homogeneous and isotropic signal $s$ can be
# expressed in harmonic space as a diagonal matrix. This means that the
# covariance matrix for the signal, $S$, can be written in harmonic space as:
#
# $$ \tilde{S}_{kk'} = \left< s_k s_{k'}^{\dagger} \right> = 2\pi \delta(k - k') P(|k|) $$
#
# where $k$ and $k'$ are the Fourier coefficients, $\tilde{S}_{kk'}$ is the
# harmonic space covariance matrix, and $P(|k|)$ is the power spectrum.
#

# %% [markdown]
# With the Wiener-Khinchin theorem, modeling the correlation structure of an
# $n$-dimensional system is reduced from scaling as $\mathcal{O}(n^2)$ to
# scaling as $\mathcal{O}(n)$.
#
# A signal $s$ is then modeled starting in harmonic space, as:
#
# $$ s = \mathrm{HT}(A \xi) $$
#
# where $\xi$ is a standard normal vector, $\xi \sim \mathcal{N}(0,1)$, $A$ is
# the amplitude spectrum related to $\tilde{S}$ as $A = \sqrt{\tilde{S}}$, and
# $\mathrm{HT}$ is the harmonic transform.

# %% [markdown]
# Since $A$ is fixed in this case, we can only define Gaussian processes with a
# fixed power spectrum. This narrows the range of the signal realizations we can
# capture with the model.

# %% [markdown]
# To circumvent this constraint and to capture a wider span of signals, we make
# the amplitude spectrum $A$ a part of the model. This means that we build a
# generative model for $A$, which is then applied to the previously introduced
# latent standard normal random vector.
#
# Expressed as a formula, the idea for a generative Gaussian process model with
# variable power spectra is:
#
# $$ s(\xi_0,\xi_1) = \mathrm{HT}\left(A(\xi_0)\xi_1\right) $$
#
# where now $A$ is a generative model for the amplitude spectrum, which
# transforms the standard normally distributed latent parameters $\xi_0$ to
# possible amplitude spectra. The result $A(\xi_0)$ is then applied to the
# latent parameters $\xi_1$, introduced in the previous notebooks. Thus, the
# variable power spectrum model takes two sets of latent parameters as input,
# $\xi_0$ and $\xi_1$. From $\xi_0$ an amplitude spectrum is generated, which is
# then applied to $\xi_1$.
#
# There are many potential generative models for amplitude spectra $A(\xi_0)$.
# NIFTy implements two options: first, to parameterize the amplitude spectrum
# with a Matérn kernel with learnable parameters, and second, to use a
# non-parametric model for the amplitude spectrum. In the following, we will
# showcase the non-parametric model.

# %% [markdown]
# ## Non-parametric correlated field model
#
# Many physical processes lead to power-law-like power spectra. For this reason,
# the non-parametric amplitude spectrum model of NIFTy builds upon a power law
# (with variable slope) to which multiplicative non-parametric deviations are
# added. A set of hyperparameters allows steering the prior distributions on the
# slope of the power law and other components of the model in order to tailor
# this general power spectrum model toward specific applications. These
# hyperparameters are named `loglogavgslope`, `fluctuations`, `offset_mean`,
# `offset_std`, `flexibility`, and `asperity`.
#
# In this notebook, we provide an intuition for these hyperparameters and
# instructions on how to set them in your applications. For complete
# mathematical derivations, we refer to
# [this publication](https://www.nature.com/articles/s41550-021-01548-0).
#

# %% [markdown]
# Let us begin by importing the required modules and initializing our space to
# build our prior models on. For this notebook, we will work with a
# 1-dimensional space with 256 points, and the distance between those points is
# given as 1, in the appropriate units.
#

# %%
import nifty.re as jft
from jax import random
from jax import numpy as jnp
import numpy as np
import jax

# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100

jax.config.update("jax_enable_x64", True)

shape = 256
distances = 1
seed = 50
key = random.PRNGKey(seed)


# %% [markdown]
# ### `CorrelatedFieldMaker` overview

# %% [markdown]
# The Gaussian process models in NIFTy with variable amplitude spectra are
# constructed using the `jft.CorrelatedFieldMaker` helper class. In most NIFTy
# applications, the `CorrelatedFieldMaker` is directly used to generate a
# Gaussian process model. For this notebook, we will pack it inside a helper
# function which we call `fieldmaker`, as we will instantiate several Gaussian
# process models to visualize how the model works. In the code below you find
# this helper function. The following list gives a high-level overview of how
# the `jft.CorrelatedFieldMaker` works before going into the details further
# below in this notebook.
#
# * First, we instantiate an instance `cfm` of the `jft.CorrelatedFieldMaker`.
# The initialization method of the correlated field maker gets an argument
# called `prefix`. This string is used as a key in the dictionary with the input
# for the Gaussian process model.
# * Second, we call the function `set_amplitude_total_offset` of the `cfm`
# object. This function takes as arguments `offset_mean` and `offset_std` and
# sets the mean of the Gaussian process. We will discuss the exact meaning of
# the two arguments of this function further below.
# * As a third step, a model for the fluctuations of the Gaussian process around
# its mean value is added with `cfm.add_fluctuations`. The `add_fluctuations`
# function gets several arguments:
# * `shape`: This is the shape of the Gaussian process we want to generate. Here
# we want to generate a one-dimensional Gaussian process with 256 pixels. To
# generate a Gaussian process with multiple dimensions, `shape` should be a
# tuple containing the number of pixels for each axis.
# * `distances`: This should be the distances between the individual pixels in
# whatever units your code is using. For this demo we will set distances to 1.
# * `**args`: The dictionary `args` contains the arguments parametrizing the
# model for the amplitude spectrum of the Gaussian process. We will discuss
# these arguments in detail below.
# * As a final step, `cfm.finalize()` is executed, returning the final Gaussian
# process model.
#
# The arguments described above are only a subset of the features of the
# `CorrelatedFieldMaker`. For additional options please see the
# [API reference](https://ift.pages.mpcdf.de/nifty/mod/nifty.re.correlated_field.html).


# %%
def fieldmaker(shape, distances, prefix, **args):
    cfm = jft.CorrelatedFieldMaker(prefix=f"{prefix}")
    cfm.set_amplitude_total_offset(
        offset_mean=args["offset_mean"], offset_std=args["offset_std"]
    )
    args.pop("offset_mean")
    args.pop("offset_std")
    cfm.add_fluctuations(
        shape=shape,
        distances=distances,
        **args,
    )
    cf_model = cfm.finalize()

    return cf_model, cfm.power_spectrum


# %% [markdown]
# ### `CorrelatedFieldMaker` parameters

# %% [markdown]
# The overview of the `CorrelatedFieldMaker` above left out detailed
# explanations about the `offset_mean` and `offset_std` arguments of the
# `set_amplitude_total_offset` method as well as the additional keyword
# arguments of the `add_fluctuations` method
# (which are `fluctuations`, `loglogavgslope`, `flexibility`, and `asperity`).
# These explanations should follow here.
#
# The arguments `offset_mean` and `offset_std` passed to the
# `set_amplitude_total_offset` method control the prior distribution for the
# average value of the Gaussian process.
# * `offset_mean` should be a float and is the mean value of the prior
# distribution on the average of the Gaussian process. Thus in the example of
# this [previous notebook](2_gaussian_processes) where we want to model the
# temperature in a room as a function of time with a Gaussian process and
# believe that the average temperature is $21$ degrees, we would set
# `offset_mean = 21`.
# * In many applications we don't know a priori the exact average value of the
# Gaussian process. In our example the actual average temperature might also be
# $22.7$ degrees. With the argument `offset_std` we can specify how much we a
# priori believe that the actual mean might deviate from `offset_mean`. Thereby
# `offset_std` should be a tuple of two positive floats. With the first entry of
# the tuple we specify the standard deviation of the prior for the average
# value. Thus in our example if we believe that the average temperature is $21$
# degrees, but think that the actual average temperature might be $21 \pm 1$
# degrees, we would set the first entry in the tuple for `offset_std` to $1.0$.
# The second entry in the tuple for `offset_std` specifies an uncertainty on the
# value set for the standard deviation (thus the first element). For many
# applications this functionality is not necessary and one can just insert a
# small positive float for the second entry in the tuple. In our example we
# could set `offset_mean = 21.` and `offset_std=(1, 1e-3)`, meaning that we
# effectively impose a Gaussian prior on the average temperature with mean 21
# and standard deviation 1. If we are not sure about what standard deviation we
# want to impose we could set something larger than $10^{-3}$ in the second
# entry of the `offset_std` tuple which will be used as an uncertainty of the
# standard deviation. However, in most applications of the correlated field
# model this functionality is not needed.
#
# The arguments `fluctuations`, `loglogavgslope`, `flexibility`, and `asperity`
# steer the prior model for the amplitude spectrum of the Gaussian process.
# Similar to `offset_std`, all of these arguments need to be tuples of floats.
# In the following we will explain the meaning of each of these parameters.
# * The `fluctuations` parameter sets the standard deviation of the Gaussian
# process around its average value. The first entry in the `fluctuations` tuple
# is the mean value for the strength of the fluctuations and the second entry
# the uncertainty on it. In our example if we would estimate that the
# temperature fluctuates over time probably by $2$ degrees, but are not sure and
# could imagine that the actual fluctuations might also be only by $1$ degree or
# even by $3$ degrees, we could set `fluctuation = (2., 1.)`.
# * The `loglogavgslope` parameter sets the prior on the slope of the power law
# modeling the amplitude spectrum. The first entry in the tuple sets the prior
# mean on the slope of the power law, the second entry the uncertainty.
# * The parameter `flexibility` sets the prior on how much the amplitude
# spectrum can deviate from a pure power law. Thereby the deviations from the
# power law are smooth functions themselves. As for the other parameters, the
# `flexibility` needs to be a tuple with the first entry being the prior mean of
# the strength of the deviations from a power law and the second entry being the
# standard deviation. If you don't want to allow for deviations from a pure
# power law amplitude spectrum, this part of the model can be disabled by
# setting `flexibility = None`.
# * The `asperity` parameter is similar to `flexibility`. The difference to
# `flexibility` is that `asperity` models small scale deviations from a power
# law. For many applications `asperity` can be disabled by setting it to `None`,
# as small scale features in the amplitude spectrum correspond to oscillatory
# patterns (with a precise length scale) in the position space. However, for
# applications where oscillatory patterns are expected, as for example day/night
# temperature fluctuations, enabling `asperity` makes sense.

# %% [markdown]
# To visualize the Gaussian process model we initialize its parameters. We set
# the prior mean of the average value to $21$ with a standard deviation of $1$.
# Furthermore, we set the mean on the fluctuations of the Gaussian process to
# $2$ with a standard deviation of $1$. We set the average slope of the
# amplitude spectrum to $-2$ (with a std of $0.3$) and activate the flexibility
# and asperity components to model deviations from a power law.

# %%
cf_kwargs = {
    "offset_mean": 21.0,
    "offset_std": (1.0, 1e-3),
    "fluctuations": (2.0, 1.0),
    "loglogavgslope": (-2.0, 0.3),
    "flexibility": (2.0, 1.0),
    "asperity": (1.0, 0.5),
    "prefix": "",
}

# %% [markdown]
# Using this set of parameters and the helper function from above we initialize
# a Gaussian process model. `model` will be the Gaussian process model itself
# and `ps` the power spectrum model.

# %%
model, ps = fieldmaker(shape=shape, distances=distances, **cf_kwargs)

# %% [markdown]
# Next we want to look at random samples from this generative Gaussian process
# model. To do so we first initialize latent space standard normal samples on
# the input domain of the Gaussian process model.

# %%
key, *subkeys = random.split(key, 5)
xi_samples = [jft.random_like(sk, model.domain) for sk in subkeys]

# %% [markdown]
# Now we can plot for each latent space sample the corresponding Gaussian
# process sample and amplitude spectrum sample. In the left column are the
# Gaussian process samples and in the right the corresponding amplitude spectra.
# Note that the model implies periodic boundary conditions.

# %%
x_vec = model.target_grids[0].distances[0] * np.arange(
    shape
)  # sampling points of Gaussian process
k_vec = model.target_grids[
    0
].harmonic_grid.mode_lengths  # fourier modes of amplitude spectrum

fig, axes = plt.subplots(len(xi_samples), 2, figsize=(15, 15))
for i, xi in enumerate(xi_samples):
    axes[i, 0].plot(x_vec, model(xi))
    axes[i, 1].loglog(k_vec, ps(xi))
plt.show()

# %% [markdown]
# ### Parameter visualization

# %% [markdown]
# The remaining part of the notebook visualizes prior samples of the Gaussian
# process model to give an intuitive understanding of how to set the parameters.
# To do so, we define below a base choice of parameters and will then vary the
# parameters one by one to visualize the impact of each of them.

# %%
cf_kwargs = {
    "offset_mean": 0.0,
    "offset_std": (0.5, 0.2),
    "fluctuations": (1.0, 0.2),
    "loglogavgslope": (-2.0, 0.3),
    "flexibility": None,
    "asperity": None,
    "prefix": "",
}

# %% [markdown]
# Now, with a model generator `fieldmaker` and the grid in place, let us look at
# how varying each of these parameters changes the output power spectrum. For
# this, we build a function `vary_parameter` that takes a hyperparameter of the
# correlated field model, an array of values it takes, and outputs plots of the
# power spectra as given by the parameter it inputs.

# %%
realisations = 5


def vary_parameter(parameter, values, **args):
    global key
    for i, j in enumerate(values):
        syn_data = np.zeros(shape=(shape, realisations))
        syn_pow = np.zeros(shape=(int(shape / 2 + 1), realisations))
        args[parameter] = j
        fig = plt.figure(tight_layout=True, figsize=(10, 3))
        fig.suptitle(f"{parameter} = {j}")
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("Field Realizations")
        ax1.set_ylim(
            -4.0,
            4,
        )
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_title("Power Spectra")
        # Plotting different realisations for each field.
        for k in range(realisations):
            cf_model, pow_cf = fieldmaker(shape, distances, **args)
            key, signalk = jax.random.split(key, num=2)
            syn_signal = jft.random_like(signalk, cf_model.domain)
            syn_data[:, k] = cf_model(syn_signal)
            syn_pow[:, k] = pow_cf(syn_signal)
            ax1.plot(x_vec, syn_data[:, k], linewidth=1)
            ax2.plot(
                k_vec,
                np.sqrt(syn_pow[:, k]),
                linewidth=1,
            )


# %% [markdown]
# ### `loglogavgslope`
#
# First, let us look at a hyperparameter that governs the power law nature of
# the power spectrum. Here, it is input as a tuple of (mean, std).
#
# The `loglogavgslope` determines the steepness of the slope of the power
# spectrum in double logarithmic coordinates. In signal space, this is
# equivalent to determining the *smoothness* of the signal.
#
# The `loglogavgslope` $m$ is modeled with a Gaussian distribution, as it can
# take both negative and positive values. A negative slope indicates a falling
# power spectrum, and a positive slope indicates a rising power spectrum. The
# mean and standard deviation of the Gaussian prior are parameterized by $\mu_m$
# and $\sigma_m$.
#
# Let us look at how varying $\mu_m$ and $\sigma_m$ affects the signal and power
# spectrum.
#
# First, we vary $\mu_m$, while keeping $\sigma_m$ a low constant value. This
# denotes that we have very high confidence in the a priori value set for
# $\mu_m$, but such a low value is used here only for demonstration purposes and
# does not reflect a practical choice of $\sigma_m$.

# %%
vary_parameter(
    "loglogavgslope",
    [(-6.0, 1e-16), (-2.0, 1e-16), (2.0, 1e-16)],
    **cf_kwargs,
)

# %% [markdown]
# We notice here that the steeper the falling slope, the smoother the signal
# realisations. This is intuitively understood as, the higher the power to the
# lowest modes (i.e. the largest scales) the more larger scale structures are
# present in the signal, which lends to the signal appearing smooth.
#
# Now let us look at the influence of the variation of $\sigma_m$ on the signal
# realisations and the power spectra.

# %%
vary_parameter("loglogavgslope", [(-2.0, 0.02), (-2.0, 0.2), (-2.0, 2.0)], **cf_kwargs)
cf_kwargs["loglogavgslope"] = (-2.0, 1e-16)

# %% [markdown]
# Here we vary the relative absolute deviation of the `loglogavgslope` by $1\%$,
# $10\%$ and $100\%$. This translates in the first case to the signal
# realisations being similarly smooth, in the second case to the signals varying
# in smoothness, and in the third case to the signals varying highly in
# smoothness. A higher value of the standard deviation of the `loglogavgslope`
# reflects our choice of being uncertain of the smoothness of the signal
# (or equivalently the steepness of the power spectrum).

# %% [markdown]
# ### `offset_mean`
#
# The `offset_mean` parameter sets the average mean of the standardized signal
# $\bar{s}$. It acts in signal space, and hence does not have an effect on the
# power spectrum.
#
# To inspect the variation of the following hyperparameters, we set a few of the
# hyperparameters to values that best show the difference in the signal and
# amplitude spectrum realizations: Here, `loglogavgslope` is set to a value of
# $-3$. Comparing to the plots above, we see that a value of $-3$ results in a
# relatively smooth signal, and a falling power spectrum.

# %%
cf_kwargs["fluctuations"] = (1.0, 1e-16)
cf_kwargs["flexibility"] = (1e-3, 1e-16)
cf_kwargs["asperity"] = (1e-3, 1e-16)
cf_kwargs["loglogavgslope"] = (-3, 1e-16)
cf_kwargs["offset_std"] = (1e-3, 1e-16)

# %%
vary_parameter("offset_mean", [3.0, 0.0, -2.0], **cf_kwargs)
cf_kwargs["offset_mean"] = 0.0

# %% [markdown]
# ### `offset_std`
#
# The `offset_std` sets uncertanty of the average value of the signal. Although
# we model the signal here in one dimension, in case the signal has multiple
# dimensions, `offset_std` determines the global zero-mode of all subdomains. It
# is input as a tuple of (mean, std).
#
# `offset_std` must always be positive, and hence is modeled with a lognormal
# prior. The lognormal prior is parametrized by setting the mean $\mu_\alpha$
# and the standard deviation $\sigma_\alpha$ of the lognormal distribution.
#
# We start by inspecting variations in $\mu_\alpha$, and then $\sigma_\alpha$.

# %%
vary_parameter("offset_std", [(1e-16, 1e-16), (0.5, 1e-16), (2.0, 1e-16)], **cf_kwargs)

# %%
vary_parameter("offset_std", [(1.0, 0.01), (1.0, 0.1), (1.0, 1.0)], **cf_kwargs)

cf_kwargs["offset_std"] = (0.5, 0.2)

# %% [markdown]
# ### `fluctuations`
#
# The `fluctuations` parameter steers how much the samples fluctuate around the
# average value. In other words, with the `fluctuations` parameter we set the
# prior distribution on the standard deviation of the Gaussian process. As the
# standard deviation needs to be positive we model it with a lognormal
# distribution. As before we parametrize the lognormal distribution by it's mean
# and standard deviation.
#
# Again, let us first look at the influence of $\mu_a$ on the signal realisations,
# then at the one of $\sigma_a$.

# %%
vary_parameter("fluctuations", [(0.05, 1e-16), (0.5, 1e-16), (1.0, 1e-16)], **cf_kwargs)
cf_kwargs["fluctuations"] = (1.0, 1e-16)

# %%
vary_parameter("fluctuations", [(1.0, 0.01), (1.0, 0.1), (1.0, 1.0)], **cf_kwargs)
cf_kwargs["fluctuations"] = (1.0, 1e-16)


# %% [markdown]
# ### `flexibility` and `asperity`
#
#
# There are two hyperparameters remaining: `flexibility` and `asperity`. The
# flexibility $\eta$ sets the amplitude of the deviations of the power spectrum
# from a power law on the double logarithmic scale, and the asperity $\epsilon$
# sets the roughness of the deviations of the power spectrum. Internally in the
# correlated field model these deviations are modeled with a Wiener process and
# an integrated Wiener process. `flexibility` corresponds to the amplitude of
# the Integrated Wiener Process (IWP) component of the power spectrum, and
# `asperity` corresponds to the roughness of the IWP component.
#
# Both `asperity` and `flexibility` are required to be positive, and are modeled
# with a lognormal prior, for which we can set the mean and variance.
#
# First, let us look at the impact on the signal and power spectra upon varying
# `flexibility`. First, we vary $\mu_\eta$ and then $\sigma_\eta$.

# %%
vary_parameter(
    "flexibility", [(0.001, 1e-16), (1.0, 1e-16), (10.0, 1e-16)], **cf_kwargs
)

# %%
vary_parameter("flexibility", [(1.0, 0.01), (1.0, 0.1), (1.0, 1.0)], **cf_kwargs)

# %% [markdown]
# To look at `asperity`, we reset the value of `flexibility` to $5.0$.

# %%
cf_kwargs["flexibility"] = (5.0, 1e-16)

# %%
vary_parameter("asperity", [(0.001, 1e-16), (1.0, 1e-16), (100.0, 1e-16)], **cf_kwargs)

# %%
vary_parameter("asperity", [(1.0, 0.01), (1.0, 0.1), (1.0, 1.0)], **cf_kwargs)
cf_kwargs["asperity"] = (1.0, 1e-16)

# %% [markdown]
# ## Summary
#
#
# In this notebook we discussed the correlated field model,
# where we set up a non-parametric generative model to infer the powerspetrum
# of the Gaussian process additionally to the process itself.
# The hyperparameters of the `CorrelatedFieldMaker` are
# the steepness of the slope `loglogavgslope`,
# the average mean of the signal `offset_mean`
# and the standard deviation of this average mean `offset_std`,
# the standard deviation of the Gaussian process `fluctuations`,
# as well as deviations from a power law `flexibility`
# and its roughness `asperity`.
