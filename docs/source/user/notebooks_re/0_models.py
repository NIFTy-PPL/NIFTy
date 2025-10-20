# %% [markdown] vscode={"languageId": "plaintext"}
# # NIFTy Models

# %% [markdown]
# This notebook provides an overview of the theory and implementation of
# statistical models in NIFTy. The first part introduces the mathematical
# background on how prior models are handled in NIFTy. The subsequent section
# discusses the implementation of prior models, and the final section focuses
# on likelihoods.

# %% [markdown]
# ## Introduction

# %% [markdown]
# Bayesian statistics allows us to combine the likelihood, which contains new
# information such as measurements, with prior knowledge encoded in a prior
# distribution. In NIFTy, the quantity we want to infer is typically called the
# signal $s$, and the data from which we want to infer $s$ is denoted by $d$.
# Using this notation, Bayes' Theorem can be written as
#
# $$P(s|d) = \frac{P(d|s) P(s)}{P(d)},$$
#
# where $P(s|d)$ is the posterior, $P(d|s)$ the likelihood, and $P(s)$ the prior.
# The term $P(d)$, called the evidence, acts as a normalization constant for the
# posterior and is often ignored in NIFTy.
#
# To reconstruct a signal $s$ from a dataset $d$ with NIFTy, you will need to
# code a prior model, a likelihood, and then run an inference algorithm. This
# notebook is about prior and likelihood models in NIFTy. Specifically, we will
# discuss the mathematical background of how priors are handled in NIFTy, as
# well as some implementation details for priors and likelihoods. In the next
# notebook, you will learn about how to infer the posterior distribution once
# you have a NIFTy model.

# %% [markdown]
# ## Prior Models in NIFTy

# %% [markdown]
# In many real-world applications, the prior distribution we want to impose on
# the quantity we infer (in NIFTy often called signal $s$) can be quite
# complicated. Nevertheless, the variational inference methods used in NIFTy to
# obtain the posterior distribution are built around Gaussian distributions. For
# this reason, a direct variational inference approximation of the posterior of
# the signal $s$ would lead to a significant approximation error. To circumvent
# this problem, NIFTy requires the user to reparameterize the model such that
# the model parameters are a priori standard normal distributed. In the NIFTy
# language, the reparameterized model parameters are often called $\xi$, and the
# signal $s(\xi)$ becomes a function of the new standardized parameters. The
# space of the standardized parameters is called the latent space.
#
# This notebook briefly introduces the mathematical foundation for standardizing
# models and explains how to code such models in NIFTy to map to the desired
# prior distribution.

# %% [markdown] vscode={"languageId": "plaintext"}
# ### Standardized Models

# %% [markdown]
# The inference algorithms of NIFTy assume that all parameters of the model are
# a priori standard normally distributed. We often denote these standard normally
# distributed parameters with $\xi$. If we now have a signal $s$ that we want to
# reconstruct from some data $d$, but want to impose a non-standard normal
# prior, then we have to construct a function transforming from the standard
# normal distribution to the desired prior distribution.
#
# Mathematically, such a mapping always exists and can be constructed from the
# Cumulative Density Function (CDF) of the Gaussian $\text{CDF}_{G}$ and the
# inverse Cumulative Density Function $\text{CDF}_{P(s)}^{-1}$ of the desired
# distribution $P(s)$. It can be shown that if $\xi$ is standard normally
# distributed $P(\xi) = G(0,1)$, then
#
# $$ s(\xi) = (\text{CDF}_{P(s)}^{-1} \circ \text{CDF}_{G}) (\xi) $$
#
# is distributed according to $P(s)$. Such mappings between parameters with a
# "simple" distribution and parameters with a more complex distribution are
# widely used in the statistics and machine learning community. They are, for
# example, also known as the reparametrization trick and are the core idea of
# inverse transform sampling.
#
# Bayes' Theorem expressing the posterior distribution $P(s|d)$ in terms of the
# likelihood $P(d|s)$ and the prior $P(s)$
#
# $$P(s|d) = \frac{P(d|s)P(s)}{P(d)}, $$
#
# can now be rewritten in terms of $\xi$
#
# $$P(\xi|d) = \frac{P(d|s(\xi))P(\xi)}{P(d)}, $$
#
# with $P(\xi)$ being the standard normal distribution. For a more detailed
# mathematical introduction tailored towards the application in NIFTy, see
# [arXiv:1812.04403](https://arxiv.org/abs/1812.04403).

# %% [markdown]
# To summarize, all the complexity of the desired prior distribution on $s$ is
# now encoded in the mapping $\xi \rightarrow s$. The posterior inference
# algorithms discussed in the next notebook will infer the posterior
# distribution $P(\xi|d)$. Via the mapping from $\xi \rightarrow s$, the
# posterior of $\xi$ determines the posterior distribution for $s$.

# %% [markdown]
# ### Implementation in NIFTy

# %% [markdown] vscode={"languageId": "plaintext"}
# The previous section introduced the mathematical foundation of standardized
# prior models. In this section, we will focus on the implementation in NIFTy.

# %% [markdown]
# As discussed, NIFTy always assumes that the model is standardized such that
# $P(\xi)$ is standard normally distributed. For this reason, NIFTy will
# automatically construct the corresponding standard Gaussian prior $P(\xi)$ for
# the parameters of the likelihood $P(d|s(\xi))$ the user has implemented. What
# NIFTy cannot do automatically is construct the mapping $\xi \rightarrow s$ and
# the likelihood itself $P(d|s)$, as these depends on the prior the use wants to
# impose and the statistics of the data. In this section, we discuss with an
# example how to implement $\xi \rightarrow s$. Afterwards, we will explain the implementation of $P(d|s)$.

# %% [markdown]
# #### Example Model

# %% [markdown]
# Implementation details of NIFTy models are best explained with examples. In
# this notebook, we will discuss a simple example: reconstructing the slope and
# offset of a linear function from measured data points. In other words,
# Bayesian linear regression. This simple example is ideal to get started with
# the NIFTy syntax. You can find more advanced examples showcasing prior models
# for image and volume reconstructions in the
# [demos folder](https://gitlab.mpcdf.mpg.de/ift/nifty/-/tree/main/demos/re?ref_type=heads).

# %% [markdown]
# For our example, let us assume that we have measured the data
# $\vec{d} \in \mathbb{R}^{N}$ at some locations $\vec{x} \in \mathbb{R}^{N}$,
# and that there is a relation between $\vec{d}$ and $\vec{x}$ following the
# form
#
# $$ \vec{d} = a \vec{x} + b\vec{1} + \vec{n}, $$
#
# with $a$ and $b$ being some unknown scalar parameters that we want to infer.
# Thus the parameters $a, b$ are our signal that we want to reconstruct from the
# data. $n$ stands for some additive noise in the measurement. Furthermore,
# $\vec{1}$ symbolizes a vector with all entries being equal to 1. Our goal is
# now to obtain the posterior distribution $P(a,b|d)$ of the parameters $a$ and
# $b$. To do so, we will first need to code a standardized prior model and a
# likelihood function, which we will do in this notebook. The next notebook will
# demonstrate how to obtain the posterior given the NIFTy model.

# %% [markdown]
# To implement the likelihood function, we code a generative model mapping from
# some latent a priori standard Gaussian distributed parameters $\vec{\xi}$ to
# the parameters $a(\vec{\xi})$ and $b(\vec{\xi})$ and then mapping to the
# predicted values of $\vec{y} = a(\vec{\xi}) \vec{x} + b(\vec{\xi})\vec{1}$.
# Thus, in the end, we will have a function mapping $\vec{\xi}$ to the
# corresponding values $\vec{y}$.

# %% [markdown]
# Before starting with the actual implementation, let us import NIFTy and the
# relevant JAX libraries. Furthermore, we activate the float64 precision in JAX,
# which is recommended for NIFTy.

# %%
import nifty.re as jft

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# Furthermore, let us define the positions $x$ at which we have measured and the
# actual data $d$ we obtained. In an application to real data, you would load
# your dataset here.

# %%
x, d = np.loadtxt("data.txt", delimiter="\t", skiprows=1, unpack=True)

# %% [markdown]
# A first good step when dealing with new data is to visualize it. To visualize
# our data, we will plot the locations at which we have measured on the x-axis
# and the corresponding datapoint on the y-axis.

# %%
plt.figure(figsize=(8, 5))
plt.scatter(x, d)
plt.xlabel("x")
plt.ylabel("d")
plt.show()

# %% [markdown]
# For this data, we now want to do a Bayesian linear regression. Let us assume
# that we have some prior knowledge of the value of the parameters $a$ and $b$.
# Namely, let us assume about $a$ that we already know that it must be a
# positive number and that we believe that its value is around $4$ with an
# uncertainty of $\pm3$. A suitable probability distribution for encoding this
# prior knowledge is the Log-Normal distribution. NIFTy already implements
# models for commonly used probability distributions, such as the Log-Normal
# distribution. A list of all implemented probability distributions can be found
# on the
# [API reference page](https://ift.pages.mpcdf.de/nifty/mod/nifty.re.prior.html).

# %% [markdown]
# We specify a NIFTy model for a Log-Normal distribution with `mean=4` and
# `std=3`. The meaning of the parameter `name` will be explained in the cell
# below.

# %%
a = jft.LogNormalPrior(mean=4, std=3, name="a_input")

# %% [markdown]
# `a` is now a NIFTy model implementing the mapping of a standard normal
# distributed $\xi$ to the Log-Normal distributed parameters $a$. Besides this
# mapping, the NIFTy model implements additional useful functionality, including
# information about the input and output of the mapping function. Specifically,
# `a.domain` contains information on how the input to the mapping function
# should be formatted.

# %%
print(a.domain)

# %% [markdown]
# As we see, the input to the model `a` should be a Python dictionary. This
# dictionary should include a key named `a_input` containing a single float.
# This float will be mapped to the log-normal distributed quantity. Why it is
# useful that the input is wrapped inside a dictionary will become clear in the
# discussion below. The fact that the key containing the input for the model `a`
# is named `a_input` is because we set this for the `name` in the initialization
# of the model `a`. We will also see in the later discussion that different
# components of our prior model, such as the models $a$ and $b$, must use
# different keys.

# %% [markdown]
# The `target` property contains information about the model's output. We see
# that the output of the model is a single float:

# %%
print(a.target)

# %% [markdown]
# We can also create some test input and apply the model `a` to it to showcase
# how to apply the model:

# %%
a_test_input = {"a_input": 0.0}
res = a(a_test_input)
print(res)

# %% [markdown]
# To showcase how the Gaussian distribution on the domain of `a` is mapped to a
# Log-Normal distribution, let us draw Gaussian samples and pass them into `a`.
# Before doing so, we first need to introduce the JAX random number generator.
#
# To draw random numbers in NIFTy, we use the JAX random number generator. You
# might want to learn more about the JAX random number generator on the
# [JAX webpage](https://docs.jax.dev/en/latest/random-numbers.html), as the JAX
# random number generator works slightly differently than the NumPy random
# number generator. In essence, to get a random number, you first
# generate a random key. Given this key, deterministic random numbers are
# created. To generate new random numbers, the key can be split into a new key
# and a subkey.
#
# Below is the code to generate random Gaussian samples for $\xi_a$ and compute
# the corresponding samples for $a(\xi_a)$. To do so, we first set the random
# seed and then generate the initial random key for JAX. Afterwards, we generate
# $2000$ random samples in a for loop. Inside the loop, we first split the JAX
# random key to get new random numbers in each iteration. Then we use the split
# random key to get a Gaussian random sample on the domain of `a` with
# `jft.random_like(subkey, a.domain)`. The Gaussian latent sample is then
# transformed into a sample of `a` by computing `a(latent_sample)`. Finally, we
# append both the latent sample and the corresponding sample from `a` to
# lists. After the for loop, we plot the lists of samples as histograms with
# matplotlib. The left plot visualizes the latent Gaussian samples, while the
# right plot shows the log-normal distributed samples from `a`.

# %%
seed = 42
key = random.PRNGKey(seed)

latent_sample_list = []
a_sample_list = []
for i in range(2000):
    key, subkey = random.split(key)
    latent_sample = jft.random_like(subkey, a.domain)
    a_sample = a(latent_sample)
    latent_sample_list.append(latent_sample["a_input"])
    a_sample_list.append(a_sample)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
axs[0].hist(latent_sample_list, bins=20)
axs[0].set_title(r"Gaussian samples of $\xi_a$")
axs[1].hist(a_sample_list, bins=20)
axs[1].set_title(r"Log-Normal samples of $a$")
plt.show()

# %% [markdown]
# Besides the prior model for $a$, we have to specify a model for $b$. Let us
# assume that we want to use a Gaussian prior with a mean of 0 and a standard
# deviation of 3 for $b$.

# %%
b = jft.NormalPrior(mean=0, std=3, name="b_input")

# %% [markdown]
# As for `a`, we can look at the domain and target properties of `b`, giving
# information on how the input to `b` should look, and what the output will be:

# %%
print(b.domain)
print(b.target)


# %% [markdown]
# Analogous to `a`, the input should be a dictionary with the value of the
# `name` parameter as a key. The output of `b` will be a single float.

# %% [markdown]
# Now we have implemented the generative models for $\xi \rightarrow s$.
# Specifically, we have implemented the models $\xi_a \rightarrow a$ and
# $\xi_b \rightarrow b$. The next step is to code a model for linear regression.
# Specifically, what we have to code is a model that predicts
# $y(\xi_a ,\xi_b) = a(\xi_a)x + b(\xi_b)$ for given input $(\xi_a, \xi_b)$ and
# given locations $x$. While for the models `a` and `b` we could use already
# existing models in NIFTy, this time we have to code our own model.

# %% [markdown]
# In its most basic form, a NIFTy model can simply be a Python function that
# implements the mapping from standard normal distributed parameters to the
# desired quantity. In our example case, the final quantity our model should map
# to is $y(\xi_a ,\xi_b)$. Nevertheless, very often it is helpful to not only
# implement the function itself but also store information about the input and
# output of the model. For the models `a` and `b`, we already looked at this
# information, namely the properties `.domain` for the input and `.target` for
# the output. To help users implement custom models that also contain
# information about the domain and target, NIFTy provides the base class
# `jft.Model`, from which custom models can inherit. Below is an implementation
# of our model. In the following, we will explain the implementation. However,
# if you are not familiar with object-oriented programming in Python, we
# recommend reading one of the numerous tutorials on the web first.


# %%
class LinearModel(jft.Model):
    def __init__(self, x, a, b):
        self.x = x
        self.a = a
        self.b = b
        super().__init__(domain=a.domain | b.domain, white_init=True)

    def __call__(self, inp):
        return self.x * self.a(inp) + self.b(inp)


my_model = LinearModel(x, a, b)

# %% [markdown]
# The class `LinearModel` inherits from the base class `jft.Model` and
# implements our model $\vec{y}(\xi_a, \xi_b) = a(\xi_a)\vec{x} + b(\xi_b)$ for
# the data. To the `__init__` function of the class, we pass the locations $x$ at
# which we have measured, together with our prior models for the parameters $a$
# and $b$. `x`, `a`, and `b` are then stored as properties of the model.
# Furthermore, the `__init__` method sets the domain of the model by calling the `__init__`
# of the `jft.Model` class. What exactly the domain needs to be depends on the
# details of the `__call__` function. For this reason, we will first discuss the
# `__call__` function before we come back to the domain.
#
# The `__call__` function implements the actual model. In our case, the
# `__call__` function implements the mapping from the latent parameters $\xi_a$
# and $\xi_b$ to the predicted values for $y$. Expressed as a formula, the call
# function computes:
# $(\xi_a ,\xi_b) \rightarrow \vec{y} = a(\xi_a)\vec{x} + b(\xi_b)\vec{1}$.
# In code, the `__call__` function receives (besides `self`) the
# variable `inp` as input, which contains the values of $\xi_a$ and $\xi_b$.
# Then the variable `inp` is passed to the models `a` and `b`. Afterwards, the
# output of `a(inp)` is multiplied by `x`, and `b(inp)` is added. Both `a` and
# `b` expect their input to be a Python dict containing the keys `a_input` and
# `b_input`, respectively. Therefore, to make this code line work without an
# error, `inp` also needs to be a Python dict with at least the keys `a_input`
# and `b_input`. For both these keys, the `inp` dict should contain a single
# float with the values of $\xi_a$ and $\xi_b$. This determines what the domain
# of the model should be, namely, the domain of the model is a Python dict with
# the keys `a_input` and `b_input`. Thus, the domain is simply the union of the
# domains of the models `a` and `b`. In the call to the `__init__` method of
# `jft.Model`, we additionally set `white_init=True`. This is optional, but
# instructs NIFTy to initialize the inference of our model with uncorrelated
# standard Gaussian random numbers.

# %%
print(a.domain | b.domain)

# %% [markdown]
# This union `a.domain | b.domain` is then passed as the `domain` to the init
# method of `jft.Model`, from which our `LinearModel` has inherited from. The
# `jft.Model` class implements the functionality that our model now also has a
# property `domain`.

# %%
print(my_model.domain)

# %% [markdown]
# That the domain of our model is also a Python dict is not a necessity, but a
# consequence of how we have implemented the call function. If we had implemented
# the call function differently, the domain could also have been something else.
# For example, if the call function were to evaluate the model like this:
# `self.x * self.a(inp[0]) + self.b(inp[1])`, then `inp` would need to be a Python
# list with the first element containing the input for `a` and the second
# element containing the input for `b`. Correspondingly, in this case, we should
# have set the `domain` to `[a.domain, b.domain]`. However, especially for large
# models, coding the call methods such that the input can be a dict is far more
# convenient than using a list or other Python data containers. For example, when
# using Python lists as the domain of all models, it is significantly more
# error-prone to add new components to a model, as one must ensure that each
# component still receives its input from the correct index location. When using
# Python dictionaries for the input, this is much easier, as we just have to
# ensure that the new components use distinct keys in the dictionary.

# %% [markdown]
# Before continuing with implementing a likelihood, let us once more visualize
# our model by plotting prior samples from it. This time, we plot prior samples
# of our model alongside the data. We start with a scatter plot of our data.
# Then, we draw $10$ prior samples of our model using a for loop and add them to
# the plot. Drawing a prior sample from the full model is very similar to
# drawing a prior sample from `a`, as discussed above. Again, we split the random
# key, generate a latent sample, and transform this latent sample into a sample
# of our model.


# %%
plt.figure(figsize=(8, 5))
plt.scatter(x, d)

for i in range(10):
    key, subkey = random.split(key)
    latent_sample = jft.random_like(subkey, my_model.domain)
    y_sample = my_model(latent_sample)
    plt.plot(x, y_sample)

plt.xlabel("x")
plt.ylabel("d")
plt.show()

# %% [markdown]
# ## Likelihood Models in NIFTy


# %% [markdown]
# To summarize, so far we have coded a model taking $(\xi_a, \xi_b)$ as an
# input and computing the corresponding
# $y(\xi_a, \xi_b) = a(\xi_a)\vec{x} + b(\xi_b)\vec{1}$. As a next step, we will
# implement the actual likelihood function, allowing us to evaluate
# $P(d|\xi_a, \xi_b)$. For this, we need to assume something about the
# distribution of the noise in the measurement equation
# $d = y + n = a\vec{x} + b\vec{1} + n$. For this introductory example, we will
# assume that $n$ is Gaussian distributed with a standard deviation of $10$ and
# all measurements are uncorrelated. For the Gaussian case, we can use the
# `jft.Gaussian` likelihood. Likelihoods for other noise distributions and
# measurement setups can be found on the
# [API reference page](https://ift.pages.mpcdf.de/nifty/mod/nifty.re.likelihood_impl.html).

# %% [markdown]
# Assuming that $n$ is Gaussian distributed, the likelihood is given by
#
# $$ P(d|\xi_a, \xi_b) = P(d|y(\xi_a,\xi_b)) = P(n = d - y(\xi_a,\xi_b)) = G(d-y, 10^2).$$
#
# Thus, the likelihood is a Gaussian distribution with the mean being the data
# $d$ and the covariance being the covariance of the noise, which is, in our
# case, a diagonal matrix with all values on the diagonal being $10^2 = 100$.
# For numerical reasons, NIFTy does not operate directly on the likelihood
# itself, but instead uses the negative logarithm of it. In NIFTy language, the
# negative logarithm is called the Hamiltonian and denoted by $H$. In formulas
# the likelihood Hamiltonian is given by
#
# $$H(d|\xi_a, \xi_b) = - \ln(P(d|\xi_a, \xi_b)).$$
#
# The corresponding NIFTy code looks like this:

# %%
cov = 10**2
noise_cov_inv = lambda x: x / cov  # diagonal matrix
lh = jft.Gaussian(data=d, noise_cov_inv=noise_cov_inv).amend(my_model)

# %% [markdown]
# Conceptually, as well as in the implementation, the likelihood `lh` shares
# many similarities with normal models in NIFTy. For example, the model and the
# likelihood are a mapping of the same latent parameters $(\xi_a, \xi_b)$. We
# can verify this by looking at the `.domain` properties of the likelihood and
# model:

# %%
print("domain of model: ", my_model.domain)
print("domain of lh: ", lh.domain)

# %% [markdown]
# The likelihood and the model both have a `.target` property indicating the
# type of the output. While the model maps to an array of 20 floats
# (the values of $y$), the likelihood maps to the log of the probability, which
# is a single float.

# %%
print("target of model: ", my_model.target)
print("target of lh: ", lh.target)

# %% [markdown]
# ## Summary

# %% [markdown]
# This notebook introduces the concept of standardized generative models. These
# models encode a potentially complex probability distribution of some
# parameters via a non-linear mapping from independent and identically
# distributed (i.i.d.) Gaussian-distributed parameters $\vec{\xi}$ to the actual
# parameters of interest. NIFTy builds upon this concept. NIFTy assumes for all
# parameters of a model a standard Gaussian prior, and a non-Gaussian
# distribution can be encoded in the form of a generative model in the
# likelihood function. The sections above have introduced how to code such a
# likelihood, including a generative model. In the next notebook, we will
# discuss how to access the corresponding posterior distribution with NIFTy.
