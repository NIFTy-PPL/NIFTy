#!/usr/bin/env python
# coding: utf-8
# # Showcasing the Correlated Field Model
#
# The field model works roughly like this:
#
# ``f = HT( A * zero_mode * xi ) + offset``
#
# The correlated field is constructed using:
#
# \begin{equation*}
# cf = \verb|offset_mean| + \left(\bigotimes_{i} \frac{1}{V_i} HT_i \right) \left( \verb|zero_mode| \cdot (\bigotimes_{i} A_i (k))
# \right)
# \end{equation*}
#
# where the outer product $\bigotimes_{i}$ is taken over all subdomains, $V_i$ is the volume of each sub-space, $HT_i$ is the harmonic transform over each sub-space
#
# `A`  is a spectral power field which is constructed from power spectra that are defined on subdomains of the target domain. It is scaled by a zero mode operator and then pointwise multiplied by a Gaussian excitation field, yielding a representation of the field in harmonic space. It is then transformed into the target real space and a offset is added.
#
# The power spectra that `A` is constructed of, are in turn constructed as the sum of a power law component and an integrated Wiener process whose amplitude and roughness can be set.

# ## Preliminaries

# +
# Imports and Initializing dimensions and a seed

# %matplotlib inline
import jax
import nifty.re as jft
import matplotlib.pyplot as plt
from jax import numpy as jnp
from typing import Tuple
import numpy as np

jax.config.update("jax_enable_x64", True)

plt.rcParams["figure.dpi"] = 300

npix = 256
seed = 42
distances = 1
key = jax.random.PRNGKey(seed)
k_lengths = jnp.arange(0, npix) * (1 / npix)
totvol = jnp.prod(jnp.array(npix) * jnp.array(distances))
realisations = 5


# ## The Moment Matched Log-Normal Distribution
#
# Many properties of the correlated field are modelled as being lognormally distributed.
#
# The distribution models are parametrized via their means and standard-deviations (first and second position in tuple).
#
# To get a feeling of how the ratio of the `mean` and `stddev` parameters influences the distribution shape, here are a few example histograms: (observe the x-axis!)

# +
fig = plt.figure(figsize=(13, 3.5))
mean = 1.0
sigmas = [1.0, 0.5, 0.1]


for i in range(3):
    op = jft.LogNormalPrior(mean=mean, std=sigmas[i], name="foo")
    op_samples = np.zeros(10000)
    for j in range(10000):
        key, signalk = jax.random.split(key, num=2)
        s = jft.random_like(signalk, op.domain)
        op_samples[j] = op(s)

    ax = fig.add_subplot(1, 3, i + 1)
    ax.hist(op_samples, bins=50)
    ax.set_title(f"mean = {mean}, sigma = {sigmas[i]}")
    ax.set_xlabel("x")
    del op_samples


plt.show()

# +
# Making the Correlated Field in Nifty.re


def fieldmaker(npix: Tuple, distances: Tuple, matern, **args):
    cf = jft.CorrelatedFieldMaker("")
    cf.set_amplitude_total_offset(
        offset_mean=args["offset_mean"], offset_std=args["offset_std"]
    )
    args.pop("offset_mean")
    args.pop("offset_std")
    # There are two choices to the kwarg non_parametric_kind, power and amplitude. NIFTy.re's default is amplitude.
    cf.add_fluctuations_matern(
        npix,
        distances,
        non_parametric_kind="power",
        renormalize_amplitude=False,
        **args,
    ) if matern else cf.add_fluctuations(
        npix, distances, non_parametric_kind="power", **args
    )
    cf_model = cf.finalize()

    return cf_model, cf.power_spectrum


def vary_parameter(parameter, values, matern, **args):
    global key
    for i, j in enumerate(values):
        syn_data = np.zeros(shape=(npix, realisations))
        syn_pow = np.zeros(shape=(int(npix / 2 + 1), realisations))
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
            cf_model, pow_cf = fieldmaker(npix, distances, matern, **args)
            key, signalk = jax.random.split(key, num=2)
            syn_signal = jft.random_like(signalk, cf_model.domain)
            syn_data[:, k] = cf_model(syn_signal)
            syn_pow[:, k] = pow_cf(syn_signal)
            ax1.plot(k_lengths, syn_data[:, k], linewidth=1)
            if not matern:
                ax2.plot(
                    np.arange(len(k_lengths) / 2 + 1),
                    np.sqrt(syn_pow[:, k] / totvol ** 2),
                    linewidth=1,
                )
                ax2.set_ylim(1e-6, 2.0)
            else:
                ax2.plot(np.arange(len(k_lengths) / 2 + 1), syn_pow[:, k], linewidth=1)
                ax2.set_ylim(1e-1, 1e2)


# -

# ### The Amplitude Spectrum in NIFTy.re
#
# The correlation operator $A$ which used to transform a to-be-inferred signal $s$ to standardized $\xi$ coordinates, is given by
# $ s = A \xi $
#
# and A is defined as (see below for derivation)
#
# $$ A \mathrel{\mathop{:}}= F^{-1}\hat{S}^{1/2} $$
#
# Where $F^{-1}$ is the inverse Fourier transform, and $\hat{S}$ is the diagonalized correlation structure in harmonic space.
#
# The norm is defined as:
#
# \begin{equation*}
#     \text{norm} = \sqrt{\frac{\int \textnormal{d}k p(k)}{V}}
# \end{equation*}
#
# The amplitude spectrum $\text{amp(k)}$ is then
# \begin{align*}
#     \text{amp(k)} = \frac{\text{fluc} \cdot \sqrt{V}}{\text{norm}} \cdot \sqrt{p(k)} \\
#     \text{amp(k)} = \frac{\text{fluc} \cdot V \cdot \sqrt{p(k)}}{\sqrt{\int \textnormal{d}k p(k)}}
# \end{align*}
#
# The power spectrum is just the amplitude spectrum squared:
#
# \begin{align*}
#     p(k) &= \text{amp}^2(k) \\
#     & = \frac{\text{fluc}^2 \cdot V^2 \cdot p(k)}{\int \textnormal{d}k p(k)} \\
#     \int \textnormal{d}k p(k) & = \text{fluc}^2 \cdot V^2
# \end{align*}
#
# Hence, the fluctuations in `NIFTy.re` are given by
#
# \begin{equation*}
#     \text{fluc} = \sqrt{\frac{\int \textnormal{d}k p(k)}{V^2}}
# \end{equation*}
#
# This is different as compared to the Numpy-based NIFTy, where:
#
# \begin{equation*}
#     \text{fluc} = \int \textnormal{d}k p(k)
# \end{equation*}
#
#

# ## The Neutral Field

# +
cf_args = {
    "fluctuations": (1e-3, 1e-16),
    "loglogavgslope": (0.0, 1e-16),
    "flexibility": (1e-3, 1e-16),
    "asperity": (1e-3, 1e-16),
    "prefix": "",
    "offset_mean": 0.0,
    "offset_std": (1e-3, 1e-16),
}

cf_model, pow_cf = fieldmaker(npix, distances, matern=False, **cf_args)

key, signalk = jax.random.split(key, num=2)
syn_signal = jft.random_like(signalk, cf_model.domain)
syn_data = cf_model(syn_signal)
syn_pow = pow_cf(syn_signal)
fig = plt.figure(tight_layout=True, figsize=(10, 3))
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
ax2.set_ylim(1e-6, 2.0)
ax1.plot(k_lengths, syn_data, linewidth=1)
ax2.plot(np.arange(len(k_lengths) / 2 + 1), np.sqrt(syn_pow / totvol ** 2), linewidth=1)

# -

# ## The `fluctuations` parameters of `add_fluctuations()`
#
# `fluctuations` determine the **amplitude of variations** along the field dimension for which add_fluctuations is called.
#
# `fluctuations[0]` set the average amplitude of the fields fluctuations along the given dimension,
# `fluctuations[1]` sets the width and shape of the amplitude distribution.
#

# ## `fluctuations` mean

vary_parameter(
    "fluctuations", [(0.05, 1e-16), (0.5, 1e-16), (1.0, 1e-16)], matern=False, **cf_args
)
cf_args["fluctuations"] = (1.0, 1e-16)

# ## `fluctuations` std

vary_parameter(
    "fluctuations", [(1.0, 0.01), (1.0, 0.1), (1.0, 1.0)], matern=False, **cf_args
)
cf_args["fluctuations"] = (1.0, 1e-16)


# ## The `loglogavgslope` parameters of `add_fluctuations()`
#
# The value of `loglogavgslope` determines the __slope of the loglog-linear (power law) component of the power spectrum.__
#
# The slope is modelled to be normally distributed.

# ## `loglogavgslope` mean

vary_parameter(
    "loglogavgslope",
    [(-6.0, 1e-16), (-2.0, 1e-16), (2.0, 1e-16)],
    matern=False,
    **cf_args,
)

# ## `loglogavgslope` std

vary_parameter(
    "loglogavgslope", [(-2.0, 0.02), (-2.0, 0.2), (-2.0, 2.0)], matern=False, **cf_args
)
cf_args["loglogavgslope"] = (-2.0, 1e-16)

# ## The `flexibility` parameters of `add_fluctuations()`
#
# Values for `flexibility` determine the __amplitude of the integrated Wiener process component of the power spectrum__ (how strong the power spectrum varies besides the power-law).
#
# `flexibility[0]` sets the _average_ amplitude of the i.g.p. component,
# `flexibility[1]` sets how much the amplitude can vary.
# These two parameters feed into a moment-matched log-normal distribution model, see above for a demo of its behavior.

# ## `flexibility` mean

vary_parameter(
    "flexibility", [(0.4, 1e-16), (4.0, 1e-16), (12.0, 1e-16)], matern=False, **cf_args
)

# ## `flexibility` std

vary_parameter(
    "flexibility", [(4.0, 0.02), (4.0, 0.2), (4.0, 2.0)], matern=False, **cf_args
)
cf_args["flexibility"] = (4.0, 1e-16)

# ## The `asperity` parameters of `add_fluctuations()`
#
# `asperity` determines how __rough the integrated Wiener process component of the power spectrum is.__
#
# `asperity[0]` sets the average roughness, `asperity[1]` sets how much the roughness can vary.
# These two parameters feed into a moment-matched log-normal distribution model, see above for a demo of its behavior.
#
#

# ## `asperity` mean

vary_parameter(
    "asperity", [(0.001, 1e-16), (1.0, 1e-16), (5.0, 1e-16)], matern=False, **cf_args
)

# ## `asperity` std

vary_parameter(
    "asperity", [(1.0, 0.01), (1.0, 0.1), (1.0, 1.0)], matern=False, **cf_args
)
cf_args["asperity"] = (1.0, 1e-16)

# ## The `offset_mean` parameter of `CorrelatedFieldMaker()`
#
# The `offset_mean` parameter defines a global additive offset on the field realizations.
#
# If the field is used for a lognormal model `f = field.exp()`, this acts as a global signal magnitude offset.
#
#

# Reset model to neutral
cf_args["fluctuations"] = (1e-3, 1e-16)
cf_args["flexibility"] = (1e-3, 1e-16)
cf_args["asperity"] = (1e-3, 1e-16)
cf_args["loglogavgslope"] = (1e-3, 1e-16)

vary_parameter("offset_mean", [3.0, 0.0, -2.0], matern=False, **cf_args)

vary_parameter(
    "offset_std", [(1e-16, 1e-16), (0.5, 1e-16), (2.0, 1e-16)], matern=False, **cf_args
)

vary_parameter(
    "offset_std", [(1.0, 0.01), (1.0, 0.1), (1.0, 1.0)], matern=False, **cf_args
)

# ## Matern Fluctuation Kernels
#
# The correlated fields model also supports parametrizing the power spectra of field dimensions using Matern kernels. In the following, the effects of their parameters are demonstrated.
#
# Contrary to the field fluctuations parametrization showed above, the Matern kernel parameters show strong interactions. For example, the field amplitude does not only depend on the amplitude scaling parameter `scale`, but on the combination of all three parameters `scale`, `cutoff` and `loglogavgslope`.

# +
# Neutral model parameters yielding a quasi-constant field

cf_args_matern = {
    "scale": (1e-2, 1e-16),
    "cutoff": (1.0, 1e-16),
    "loglogslope": (-2.0, 1e-16),
    "prefix": "",
    "offset_mean": 0.0,
    "offset_std": (1e-3, 1e-16),
}
# -

vary_parameter(
    "scale", [(0.01, 1e-16), (0.1, 1e-16), (1.0, 1e-16)], matern=True, **cf_args_matern
)

vary_parameter(
    "scale", [(0.5, 0.01), (0.5, 0.1), (0.5, 0.5)], matern=True, **cf_args_matern
)
cf_args_matern["scale"] = (0.5, 1e-16)

vary_parameter(
    "cutoff", [(10.0, 1.0), (10.0, 3.16), (10.0, 100.0)], matern=True, **cf_args_matern
)

# ## Theory of the Correlated Field Model
#
# - Want to infer a signal $s$, and have a (multivariate Gaussian) prior $\mathcal{G}(s,S)$
# - Inference algorithms are sensitive to coordinates, more efficient to infer $\mathcal{G}(\xi, 1)$ as opposed to $\mathcal{G}(s,S)$
# - Can find a coordinate transform (known as the amplitude transform) which changes the distribution to standardized coordinates $\xi \hookleftarrow \mathcal{G}(\xi, 1)$.
# - This transform is given by $ s = F^{-1}\hat{S}^{1/2}\xi \mathrel{\mathop{:}}= A\xi $
#
# We can now define a Correlation Operator $A$ as
# \begin{equation}
#     A \mathrel{\mathop{:}}= F^{-1}\hat{S}^{1/2}
# \end{equation}
# which relates $\xi$ from the latent space to $s$ in physical space.
#
# The correlation structure $S$ and $A$ are related as,
# \begin{equation}
#     S = A A^{\dagger}
# \end{equation}
#
#
# (Here we assume that the correlation structure $S$ is statistically homogenous and stationary and isotropic (which means that $S$ does not have a preferred location or direction a priori), which then according to the Wiener-Khintchin theorem can be represented as a diagonal matrix in Harmonic (Fourier) Space.)
#
# Then,
#
# $S$ can be written as $S = F^\dagger (\hat{p_S}) F $ where $\hat{p}$ is a diagonal matrix with the power spectra values for each mode on the diagonal.
#
# From the relation given above, the amplitude spectrum $p_A$ is related to the power spectrum as:
# $$ p_A(k) = \sqrt{p_S(k)} $$
# $$ C_s(k) = \lim\limits_{V \to \infty} \frac{1}{V} \left< \left| \int_{V} dx s^{x} e^{ikx} \right| \right>_{s}$$
#
#
# If we do not have enough information about the correlation structure $S$ (and consequently the amplitude $A$), we build a model for it and learn it using data. Again, assuming statistical homogeneity a priori, then the Wiener Khintchin theorem yields:
#
# $$ A^{kk'} = (2\pi)^{d} \delta(k - k') p_A(k) $$
#
# We can now non-parametrically model $p_A(k)$, with the only constraint being the positivity of the power spectrum. To enforce positivity, let us model $\gamma(k)$ where
# $$ p_A(k) \propto e^{\gamma(k)} $$
#
# $\gamma(k)$ is modeled with an Integrated Wiener Process in $l = log|k|$ coordinates. In logarithmic coordinates, the zero mode $|k| = 0$ is infinitely far away from the other modes, hence it is treated separately.
#
# The integrated Wiener process is a stochastic process that is described by the equation:
#
# $$ \frac{d^2 \gamma(k)}{dl^2} = \eta (l) $$
# where $\eta(l)$ is standard Gaussian distributed.
#
# One way to solve the differential equation is by splitting the single second order differential equation into two first order differential equations by substitution.
#
# $$ v = \frac{d \gamma(k)}{dl} $$
# $$ \frac{dv}{dl} = \eta(l) $$
#
# Integrating both equations wrt $l$ results in:
#
# \begin{equation}
#     \int_{l_0}^{l} v(l') dl' = \gamma(k) - \gamma(0)
# \end{equation}
#
# \begin{equation}
#     v(l') - v(l_0) = \int_{l_0}^{l'} \eta(l'') dl''
# \end{equation}
#
#
#
#
#
#
#

# ## Mathematical intuition for the Fluctuations parameter
#
# The two-point correlation function between two locations $x$ and $y$ is given by $S^{xy}$ which is a continuous function of the distance between the two points, $x-y$ assuming a priori statistical homogeneity.  $$ S^{xy}= C_s(x-y) \mathrel{\mathop{:}}= C_s(r)$$
#
# When this is the case, the correlation structure is diagonalized in harmonic space and is described fully by the Power Spectrum $P_s(k)$ i.e,
#
# \begin{align*}
#     S^{xy} & = (F^\dagger)^r_k P_s (k) \\
#            & = \int \frac{\mathop{dk}}{(2\pi)^u} \exp{[-ikr]} P_s(k) \\
# \end{align*}
#
# Then, the auto correlation
#
# \begin{align*}
#     S^{xx} & = \left< s^x s^{x*}\right> \\
#     & = C_s(0) = \int \frac{dk}{(2\pi)^u} e^{0} P_s(k) \\
#     \int P_s(k) \mathop{dk} & = \left< |s^x|^2 \right>
#
# \end{align*}
#
# This $P_s(k)$ is modified to a power spectrum with added fluctuations,
#
# \begin{align*}
# P_s ' = \frac{P_s}{\sqrt{\int \mathop{dk} P_s(k)}} a^2
# \end{align*}
