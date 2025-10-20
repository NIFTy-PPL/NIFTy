# %% [markdown]
# # Inference with NIFTy

# %% [markdown]
# This notebook is the continuation of the previous NIFTy models
# [notebook](0_models). While the previous notebook gives a basic
# introduction to prior models and likelihoods in NIFTy, this notebook
# showcases how to obtain the posterior distribution for a given model in NIFTy.
#
# After a brief mathematical introduction to the inference techniques used in
# NIFTy, this notebook will proceed with the linear regression example from the
# [previous notebook](0_models). For this reason, please read the introduction
# to generative models first.

# %% [markdown]
# ## Recap

# %% [markdown]
# The previous notebook introduced a NIFTy model for linear regression in one
# dimension. Specifically, the model was

# %% [markdown]
# $$ \vec{d} = a \vec{x} + b \vec{1} + \vec{n}, $$

# %% [markdown]
# with $\vec{d}$ being the measured data, $\vec{x}$ the locations at which we
# measured, $\vec{n}$ some unknown noise in the measurements, and $a$ and $b$ the
# scalar parameters of the linear function we fit. We imposed a lognormal prior
# on $a$ and a normal prior on $b$, both in the form of a standardized model as
# required by NIFTy. This means we have a priori standard normally distributed
# latent parameters $\vec{\xi} = (\xi_a, \xi_b)$ which are mapped to
# $a(\xi_a), b(\xi_b)$ with a mapping constructed such that $a$ and $b$ have
# the desired prior distributions.
#
# In this notebook, we will infer the posterior distribution of $\vec{\xi}$,
# which determines the posterior for $a(\xi_a)$ and $b(\xi_b)$.
#
# ### Code for the NIFTy Model
#
# Below you can find the NIFTy code for the linear regression model we developed
# in the last notebook. In this notebook, we will not explain the NIFTy code
# again. Thus, if you are not familiar with NIFTy and have not read the previous
# notebook, please do so first.

# %%
import nifty.re as jft

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

import matplotlib.pyplot as plt

# enable float64 precision
jax.config.update("jax_enable_x64", True)


# initialize JAX random key
seed = 42
key = random.PRNGKey(seed)

# measurement locations and data
x, d = np.loadtxt("data.txt", delimiter="\t", skiprows=1, unpack=True)


# models for a and b
a = jft.LogNormalPrior(mean=4, std=3, name="a_input")
b = jft.NormalPrior(mean=0, std=3, name="b_input")


# linear regression model
class LinearModel(jft.Model):
    def __init__(self, x, a, b):
        self.x = x
        self.a = a
        self.b = b
        super().__init__(domain=a.domain | b.domain, white_init=True)

    def __call__(self, inp):
        return self.x * self.a(inp) + self.b(inp)


my_model = LinearModel(x, a, b)


# likelihood
cov = 10**2
noise_cov_inv = lambda x: x / cov  # diagonal matrix
lh = jft.Gaussian(data=d, noise_cov_inv=noise_cov_inv).amend(my_model)

# %% [markdown]
# To summarize, the above code implements a generative model that maps the latent
# parameters $(\xi_a, \xi_b)$ to $\vec{y} = a(\xi_a) \vec{x} + b(\xi_b) \vec{1}$,
# and defines the likelihood $P(d|\xi_a, \xi_b)$. The generative process for
# $a$ and $b$ is designed such that the latent parameters $\xi_a$ and $\xi_b$
# follow standard normal (unit Gaussian) priors.

# %% [markdown]
# ## Theory

# %% [markdown]
# Using Bayes' Theorem, we can express the posterior distribution
# $P(\vec{\xi} \,|\, d)$ corresponding to the above prior and likelihood as:
#
# $$P(\vec{\xi} \,|\, d) = \frac{P(d \,|\, \vec{\xi}) \, P(\vec{\xi})}{P(d)},$$
#
# where the evidence $P(d)$ is given by
# $P(d) = \int d\vec{\xi} \, P(d \,|\, \vec{\xi}) \, P(\vec{\xi}).$
#
# For NIFTy models, the prior probability $P(\vec{\xi})$ is always a standard
# Gaussian. Bayes' Theorem can also be formulated in terms of information
# Hamiltonians, which are defined as the negative logarithm of the corresponding
# probability: $H = -\ln P$. For numerical reasons, NIFTy always performs
# computations in terms of Hamiltonians. Expressed in Hamiltonians, Bayes'
# Theorem becomes:
#
# $$H(\vec{\xi} \,|\, d) = H(d \,|\, \vec{\xi}) + H(\vec{\xi}) - H(d).$$
#
# While Bayes' Theorem provides a mathematically exact expression for the
# posterior distribution, directly extracting information from it is often
# challenging.
#
# The primary difficulty lies in numerical integration, which becomes
# computationally intractable in high-dimensional spaces. For example,
# evaluating the evidence
# $P(d) = \int d\vec{\xi} \, P(d \,|\, \vec{\xi}) \, P(\vec{\xi})$
# requires integration over $\vec{\xi}$. Similarly, computing summary statistics,
# such as the posterior mean
# $\langle a \rangle = \int d\vec{\xi} \, a(\vec{\xi}) \, P(\vec{\xi} \,|\, d),$
# also requires integration over $\vec{\xi}$.
#
# In our example, $\vec{\xi} = (\xi_a, \xi_b)$ is only two-dimensional, making
# numerical integration feasible. However, in many real-world problems, the
# number of parameters is much larger, rendering direct numerical integration
# impractical. Therefore, Bayesian inference often relies on advanced algorithms
# to avoid explicit high-dimensional integration.
#

# %% [markdown]
# ### Maximum a posteriori

# %% [markdown]
# The simplest, though often inaccurate, inference method is to compute only
# the maximum of the posterior distribution and use this as an estimate. This
# approach, known as maximum a posteriori (MAP) estimation, is computationally
# cheap and easy to implement.
#
# To maximize $P(\vec{\xi} \,|\, d)$ (or equivalently minimize
# $H(\vec{\xi} \,|\, d) = -\ln P(\vec{\xi} \,|\, d) = H(d \,|\, \vec{\xi}) + H(\vec{\xi}) - H(d)$),
# we do not need to compute the evidence $H(d)$, as it is constant with respect
# to $\vec{\xi}$. Therefore, no integration over the $\vec{\xi}$ space is required.
# If $H(\vec{\xi} \,|\, d)$ is differentiable with respect to $\vec{\xi}$,
# gradient-based minimizers can efficiently find the minimum. This makes MAP
# estimation a very fast method for accessing posterior information.
#
# However, MAP estimation has significant drawbacks, which is why more accurate,
# though computationally expensive, inference methods are often preferred:
#
# 1. **No Uncertainty Quantification:** MAP yields only a point estimate and does
#    not provide information about the uncertainty or spread of the posterior.
#
# 2. **Volume Effects in High Dimensions:** For high-dimensional distributions,
#    the mode (maximum density point) is not necessarily representative of typical
#    samples from the posterior. This is because MAP considers only the probability
#    density, not the volume of regions in parameter space. A region with moderate
#    density but large volume may contribute more to the overall posterior
#    probability mass than the small region near the MAP point. This "volume effect"
#    becomes increasingly dominant as the number of dimensions grows, because the
#    total volume grows exponentially with the number of dimensions.
#
# Due to these limitations, MAP estimation is often insufficient, and more advanced
# inference algorithms are required. Two prominent alternatives are:
#
# - **Monte Carlo Sampling:** Generates samples from the posterior and uses them
#   to compute statistics such as the posterior mean.
# - **Variational Inference:** Approximates the posterior with a simpler
#   distribution, optimizing the parameters of this approximation to best match
#   the true posterior.
#
# While Monte Carlo methods can achieve high accuracy if enough samples are drawn,
# variational inference is typically computationally more efficient. For this
# reason, NIFTy primarily relies on variational inference.

# %% [markdown]
# ### Variational Inference

# %% [markdown]
# This section explains the basic idea behind the variational inference
# algorithms of NIFTy. For a detailed mathematical introduction, please refer to
# the original publications, specifically
# [arXiv:2105.10470](https://arxiv.org/abs/2105.10470) and, for an earlier
# version, [arXiv:1901.11033](https://arxiv.org/abs/1901.11033).
#
# The idea of variational inference is to approximate the posterior distribution
# with a simpler distribution. The variational inference algorithms in NIFTy are
# built around Gaussian distributions, approximating the posterior with a
# Gaussian. The variational parameter optimized during this process is the mean
# of the Gaussian. The covariance is not directly optimized but is instead
# constructed from the inverse Fisher information metric, which leverages
# curvature information of the posterior. Thereby the variational inference
# algorithms of NIFTy can capture posterior correlations.
#
# In the Metric Gaussian Variational Inference (MGVI) algorithm
# ([arXiv:1901.11033](https://arxiv.org/abs/1901.11033)), this Gaussian
# approximation is performed directly in the space of $\vec{\xi}$. The
# Geometric Variational Inference (geoVI) method
# ([arXiv:2105.10470](https://arxiv.org/abs/2105.10470)) applies an additional
# nonlinear transformation of the $\vec{\xi}$ space to make the true posterior
# more Gaussian, thereby improving the accuracy of the approximation.
#
# From information theory, a distance measure for probability distributions can
# be derived that quantifies their similarity. This measure, known as the
# Kullback-Leibler (KL) divergence, serves as the cost function in variational
# inference. In NIFTy, the KL divergence between the true posterior and the
# Gaussian approximation is minimized with respect to the posterior mean of the
# approximating Gaussian.

# %% [markdown]
# The closer the true posterior distribution is to a Gaussian, the more accurate
# the approximation becomes. This is one reason why NIFTy enforces a standard
# Gaussian prior on $\vec{\xi}$, while non-Gaussianities are captured in the
# mapping $\xi \rightarrow s$. The idea is that in a coordinate system where the
# prior is Gaussian, the posterior is also relatively close to a Gaussian
# distribution. In particular, for degrees of freedom that are unconstrained or
# only weakly constrained by the likelihood, the posterior remains Gaussian, and
# the variational inference algorithms do not introduce significant approximation
# errors.
#
# In the following, we briefly outline the variational inference procedure used
# in NIFTy. For a more thorough introduction to variational inference, see the
# original publications listed above.
#
# 1. Initialize a starting point for the variational inference algorithm in the
#    $\vec{\xi}$ space. This starting point will be the mean of the initial
#    Gaussian approximation. Typically, a Gaussian random sample is used as the
#    starting point.
# 2. Draw samples $\vec{\xi}_0, \ldots, \vec{\xi}_n$ from the current Gaussian
#    approximation of the posterior.
# 3. Use these samples to approximate the Kullback-Leibler divergence between
#    the approximating distribution and the true posterior, and minimize this
#    divergence with respect to the mean of the approximation.
# 4. Draw new samples from the updated approximation and repeat the minimization
#    of the Kullback-Leibler divergence. Iterate this process until convergence.
# 5. Output the final set of samples from the approximate distribution to the
#    user.

# %% [markdown]
# ## NIFTy implementation

# %% [markdown]
# In the following, we will discuss how to use the variational inference
# algorithms presented above in NIFTy. For simplicity, we demonstrate the use of
# MGVI here. For examples utilizing the more accurate geoVI algorithm, see the
# [demos folder](https://gitlab.mpcdf.mpg.de/ift/nifty/-/tree/main/demos/re?ref_type=heads)
# of NIFTy. More advanced features of the `jft.optimize_kl` function, which is
# used to run the variational inference, can be found on the
# [API reference page](https://ift.pages.mpcdf.de/nifty/mod/nifty.re.optimize_kl.html).
#
# First, we generate a random starting position for the variational inference
# run. This random position serves as the mean value of the Gaussian in the
# initial variational inference iteration. To obtain this random position, we
# first generate a JAX random number key, which is then used to create a random
# position in the latent space of our model. Finally, we convert this latent
# position into a `jft.Vector`. This conversion is necessary because NIFTy needs
# to perform arithmetic operations on the latent space position. While
# mathematical operations such as $\vec{\xi}_1 + \vec{\xi}_2$ are defined on
# latent space vectors, Python cannot add two dictionaries. This issue is
# resolved by converting the dictionary into a `jft.Vector`.

# %%
key, subkey = random.split(key, 2)
init_pos = lh.init(subkey)
print("initial position: ", init_pos)
init_pos = jft.Vector(init_pos)
print("initial position vector: ", init_pos)


# %% [markdown]
# Next, we specify how many iterations of drawing samples and minimizing the
# Kullback-Leibler divergence we want to do:

# %%
n_vi_iterations = 6

# %% [markdown]
# Furthermore, we specify that the number of independent samples used to
# approximate the Kullback-Leibler divergence should be 4. Note that NIFTy draws
# pairs of antithetical samples, meaning we will actually have $4 \cdot 2 = 8$
# samples.


# %%
n_samples = 4

# %% [markdown]
# Drawing random samples from the approximating distribution requires generating
# JAX random numbers. To do so, we as always need to generate a new JAX random
# key:

# %%
key, sampling_key = random.split(key, 2)

# %% [markdown]
# It is not possible to directly draw samples from the Gaussian posterior
# approximation due to subtleties in the parametrization of the Gaussian
# covariance. Therefore, sampling from the current approximation requires not
# only random numbers but also a numerical optimization algorithm—specifically,
# the conjugate gradient (CG) algorithm.
#
# For the CG algorithm, we specify the parameter `cg_name=None` to suppress
# NIFTy’s output, making the notebook more readable when rendered as a webpage.
# In general, however, it is recommended to set `cg_name` to a descriptive
# string to see the output of the optimizer. Additionally, we pass `cg_kwargs`,
# where we define a convergence criterion `absdelta=1e-5 * jft.size(lh.domain)`
# and a maximum number of iterations `maxiter=100` for the CG algorithm.

# %%
draw_linear_kwargs = dict(
    cg_name=None,
    cg_kwargs=dict(absdelta=1e-5 * jft.size(lh.domain), maxiter=100),
)

# %% [markdown]
# Similar to sampling, the minimization of the Kullback-Leibler divergence
# between the true posterior and the approximation is performed using a numerical
# optimization algorithm, in this case, the Newton conjugate gradient scheme.
# As before, we set `name=None` to suppress output and make the notebook more
# readable when viewed as a webpage. For production runs, however, we recommend
# setting a descriptive string for `name`. Additionally, we define a convergence
# criterion `xtol=1e-4` and set the maximum number of iterations to `maxiter=35`.

# %%
kl_kwargs = dict(minimize_kwargs=dict(name=None, xtol=1e-4, maxiter=35))

# %% [markdown]
# As a final step, we specify the `sampling_mode`. This argument allows the user
# to choose which variational inference algorithm to use. Here, we set it to
# `"linear_resample"`, which corresponds to the MGVI algorithm described in
# [arXiv:1901.11033](https://arxiv.org/abs/1901.11033). Other available options
# can be found on the
# [API reference page](https://ift.pages.mpcdf.de/nifty/mod/nifty.re.optimize_kl.html).

# %%
sample_mode = "linear_resample"

# %% [markdown]
# After having specified all necessary arguments for the variational inference
# algorithm, we collect these arguments in a Python dictionary.

# %%
optimize_kl_args = dict(
    likelihood=lh,
    position_or_samples=init_pos,
    n_total_iterations=n_vi_iterations,
    n_samples=n_samples,
    key=sampling_key,
    draw_linear_kwargs=draw_linear_kwargs,
    kl_kwargs=kl_kwargs,
    sample_mode=sample_mode,
)

# %% [markdown]
# Now we can run the variational inference using the `jft.optimize_kl` function.
# After each iteration, NIFTy prints a summary that includes:
#
# * **Iteration information**: The current iteration index of the overall
#   variational inference loop, along with the latest estimate of the
#   Kullback-Leibler divergence between the approximation and the true posterior.
#
# * **Sampling details**: For each sample pair (4 in our example), NIFTy reports
#   whether the conjugate gradient method used for sampling was successful.
#   Here, 0 means success and 1 means failure.
#
# * **KL minimization details**: The number of KL minimization steps performed.
#
# * **Results for prior and likelihood**: This section is the most important.
#   NIFTy reports the reduced $\chi^2$ of the residuals between the reconstructed
#   data and the actual data together with the average of the reconstructed data
#   samples and the number of degrees of freedom.
#   A reduced $\chi^2$ value larger than 1 can indicate that the
#   reconstruction has not converged, that noise statistics assumptions are
#   incorrect, or that the model does not fully capture all relevant effects.
#   NIFTy also reports those values for the latent parameters. Since we assume
#   a standard Gaussian prior in the latent space, $\chi^2$ values significantly
#   greater than 1 indicate tension between the prior model and the likelihood.

# %%
samples, state = jft.optimize_kl(**optimize_kl_args)

# %% [markdown]
# The `optimize_kl` function returns `samples` and `state`. In most use cases,
# only the `samples` object is relevant as it contains the samples of the
# approximating distribution from the final iteration. These samples are samples
# in the latent $\xi$ space. To obtain samples for the quantity of interest, in
# NIFTy language the signal $s$, or in our examples the parameters $a$ and $b$,
# these samples need to be transformed using the standardized model.
#
# In our examples, this transformation can be done as follows:

# %%
a_samples = tuple(a(s) for s in samples)
b_samples = tuple(b(s) for s in samples)
y_samples = tuple(my_model(s) for s in samples)

# %% [markdown]
# Now we can compute the posterior mean values of our parameters and print them.

# %%
a_mean = jft.mean(a_samples)
b_mean = jft.mean(b_samples)
y_mean = jft.mean(y_samples)
print("mean of a: ", a_mean)
print("mean of b: ", b_mean)

# %% [markdown]
# Often, it is insightful to plot the reconstruction alongside the data. Such a
# plot can also be very helpful to detect potential problems in the inference.
# The plot below visualizes again the data as a scatter plot. On top, the
# posterior mean of the reconstruction is plotted in red. In grey, the
# individual samples are shown.

# %%
plt.figure(figsize=(8, 5))
plt.scatter(x, d, label="data")
plt.plot(x, y_mean, label="posterior mean", linewidth=3.0, color="red")
for ys in y_samples:
    plt.plot(x, ys, linewidth=2.0, color="black", alpha=0.5, zorder=0)
plt.legend()
plt.show()

# %% [markdown]
# ## Summary

# %% [markdown]
# This notebook introduced the concept of variational inference. Variational
# inference is a technique for accessing information from the posterior
# distribution in a computationally efficient way. The core idea is to
# approximate the potentially complex posterior distribution with a simpler
# distribution and generate samples from this simpler distribution. In addition
# to providing a brief introduction to these concepts, this notebook
# demonstrates how to use variational inference in NIFTy.
