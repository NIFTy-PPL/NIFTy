---
title: 'Re-Envisioning Numerical Information Field Theory (NIFTy.re): A Library for Gaussian Processes and Variational Inference'
tags:
  - Python
  - Astronomy
  - Imaging
  - Gaussian Processes
  - Variational Inference
authors:
  - name: Gordian Edenhofer
    orcid: 0000-0003-3122-4894
    corresponding: true
    affiliation: "1, 2"  # Multiple affiliations must be quoted
  - name: et al.
affiliations:
 - name: Max Planck Institute for Astrophysics, Karl-Schwarzschild-Straße 1, 85748 Garching bei München, Germany
   index: 1
 - name: Ludwig Maximilian University of Munich, Geschwister-Scholl-Platz 1, 80539 München, Germany
   index: 2
date: 31 August 2023
bibliography: paper.bib
header-includes:
  - \usepackage{mathtools}

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

<!--
## JAX + NIFTy Paper
* USP: selling point: speed
* Bonus: higher order diff for more efficient optimization and all of Tensorflow and Tensorflow for all
* GP
  * Regular Grid Refinement
  * KISS-GP
  * Grid Refinement
* Posterior Approx.
  * HMC but with variable dtype handling
  * JIT-able VI and also (indirectly) available for Tensorflow
* predecessor enabled 100B reconstruction
* middle ground between tools like blackjax and pymc
-->

# Summary

* Imaging at the most basic level is the process of transforming noisy, incomplete data into a space that humans can interpret.
* \texttt{NIFTy} is a Bayesian imaging framework that propagates the statistical uncertainty in the data and the model to the image domain.
* \texttt{NIFTy} has already successfully been applied to the fields of radio astronomy, galactic tomography, and observational cosmology.
* A focus on CPU computing and previous design decisions, held the performance and the development of new inference methods in \texttt{NIFTy} back.
* We present a re-write of NIFTy, coined \texttt{NIFTy.re}, which bridges \texttt{NIFTy} to Machine Learning ecosystem, reworks the modeling principle, the inference strategy, and outsources much of the heavy lifting to JAX to ease maintainability and allow for a faster development cycle.
* The re-write dramatically accelerated models written in NIFTy, lays the foundation for new kinds of inference machineries, and enables the interoperability of \texttt{NIFTy} with the JAX/XLA Machine Learning ecosystem.

# Statement of Need

Imaging commonly involves millions to billions of pixels.
Each pixel usually corresponds to one or more correlated degrees of freedom in the model space.
Modeling millions to billions of degrees of freedom is computationally demanding.
However, imaging is not only computationally demanding but also statistically challenging.
The noisy input requires a statistical treatment and needs to be accurately propagated from the input to the final image.
To infer an image from noisy data, we require an inference machine that not only handles millions to billions of degrees of freedom but one that does so in a statistically rigorous way.

\texttt{NIFTy} is a Bayesian imaging library [@Selig2013; @Steiniger2017; @Arras2019].
It is designed to infer the million to billion dimensional posterior distribution in the image space from noisy input data.
At the core of \texttt{NIFTy} lies a set of powerful Gaussian Processes (GP) for modeling correlated degrees of freedom in high dimensions and accurate Variational Inference (VI) algorithms.

\texttt{NIFTy.re} is a rewrite of \texttt{NIFTy} in JAX [@Jax2018] with all relevant previous GP models, a new, even more powerful GP model, and a much more flexible posterior approximation machinery.
By being written in JAX, \texttt{NIFTy.re} effortlessly runs on accelerator hardware such as the GPU and TPU, vectorizes models whenever possible, just-in-time compiles code for additional performance, and enables new kinds of inference machineries thanks to being able to retrieve higher order derivates.
By switching from a home-grown automatic differentiation engine to JAX, we envision to harness significant gains in maintainability of \texttt{NIFTy.re} compared to \texttt{NIFTy},  and moving to faster development cycle for new features.

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->
We expect \texttt{NIFTy.re} to be highly useful for many imaging applications and envision many applications within and outside of astrophysics [@Arras2022; @Leike2019; @Leike2020; @Mertsch2023; @Roth2023DirectionDependentCalibration; @Hutschenreuter2023; @Tsouros2023; @Roth2023FastCadenceHighContrastImaging; @Hutschenreuter2022].
\texttt{NIFTy.re} has already been successfully used in two galactic tomography publications [@Leike2022; @Edenhofer2023].
A very early version of \texttt{NIFTy.re} enabled a 100 billion dimensional reconstruction using a maximum posterior inference.
In a newer publication, \texttt{NIFTy.re} was used to infer a 500 million dimensional posterior dimensional using VI [@Knollmueller2019].
The latter publication extensively used \texttt{NIFTy.re}'s GPU support, which yielded orders of magnitude speed-ups.
With \texttt{NIFTy.re} bridging ideas from \texttt{NIFTy} to JAX/XLA, we envision many new possibilities for inferring classical Machine Learning models with \texttt{NIFTy}'s inference methods and using \texttt{NIFTy}-components such as the GP models in classical neural networks frameworks.

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->
\texttt{NIFTy.re} competes with other GP libraries as well as with probabilistic programming languages and frameworks.
Compared to GPyTorch [@Hensman2015], GPflow [@Matthews2017], george [@Sivaram2015], or TinyGP [@ForemanMackey2024], \texttt{NIFTy} and \texttt{NIFTy.re} focus on GP models for structured spaces.
These spaces can be arbitrarily deformed in the new GP model implemented in \texttt{NIFTy.re} [@Edenhofer2022].
<!-- I think the last to sentences might make it a bit unclear if the focus of nifty are GP models for structured or arbitrary spaces  -->
Compared to classical probabilistic programming languages such as Stan [@Carpenter2017] and frameworks such pyro [@Bingham2019], numpyro [@Phan2019], pyMC3 [@Salvatier2016], Emcee [@ForemanMackey2013], dynesty [@Speagle2020; @Koposov2023], or blackjax [@blackjax2020], \texttt{NIFTy} and \texttt{NIFTy.re} focus on high dimensional inference with millions to billions of degrees of freedom.
In contrast to many GP libraries, neither \texttt{NIFTy} nor \texttt{NIFTy.re} assume the posterior to be analytically accessible. <!-- Would it makes sense to move this sentence up such that is directly behind the citation of other GP libraries?  -->
Instead, \texttt{NIFTy} and \texttt{NIFTy.re} try to approximate the true posterior using VI.
With \texttt{NIFTy.re} the GP models and the VI machinery are now fully accessible in the JAX ecosystem and \texttt{NIFTy.re} components interact seamlessly with other JAX packages such as `blackjax` and `jaxopt` [@Blondel2021].

# Core Components

\texttt{NIFTy.re} brings tried and tested structured GP models and VI algorithms to JAX.
GP models are especially useful for imaging problems, while VI algorithms are essential to probing high dimensional posteriors, which are often encountered in imaging problems.
\texttt{NIFTy.re} treats the imaging problem as a Bayesian inference problem.<!-- I think at this location in the text this sentence doesn't really make sense. I would either have it at the beginning of the section than it would explain again why we have that many degrees of freedom, or remove since it was already mentioned.  -->
\texttt{NIFTy.re} infers the parameters of interest from noisy data and a stochastic mapping from the parameters of interest to the data a.k.a. a model.

\texttt{NIFTy} and \texttt{NIFTy.re} build up hierarchical models for the posterior.
The log-posterior function reads $\ln\mathcal{p(\theta|d)} \coloneqq \mathcal{l}(d, f(\theta)) + \ln\mathcal{p}(\theta) + \mathrm{const}$ with log-likelihood $\mathcal{l}$, forward model $f$ mapping the parameters of interest $\theta$ to the data space, and log-prior $\ln\mathcal{p(\theta)}$.
The goal of the inference is to draw samples from the posterior $\mathcal{p}(\theta|d)$.

What is considered part of the likelihood or part of the prior is ill-defined.
Without loss of generality \texttt{NIFTy} and \texttt{NIFTy.re} formulate models such that the prior always is a standard Gaussian.
This choice of re-parameterization [@Rezende2015] is called standardization.
All relevant details of the prior model are encoded in the forward model $f$.
<!-- I think the rest of the paragraph is not really essential. If we have to shorten the paper we could remove it.  -->
The standardization is often carried out implicitly in the background, however, for prior models outside the current toolbox the user has to manually implement the necessary component for $f$ that corresponds to the desired non-Gaussian prior.

## Gaussian Processes

One standard tool from the \texttt{NIFTy.re} toolbox are the structured GP models from \texttt{NIFTy}.
These models usually rely on the harmonic domain being easily accessible, e.g. for pixels spaced on a regular Cartesian grid, the natural choice to represent a stationary kernel is the Fourier domain.
In the generative picture, a realization $s$ drawn from a GP then reads $s = \mathcal{HT} \cdot \sqrt{P} \cdot \xi$ with $\mathcal{HT}$ the harmonic transform, $\sqrt{P}$ the square-root of the power-spectrum in harmonic space, and $\xi$ standard Gaussian random variables.
In the implementation in \texttt{NIFTy.re} and \texttt{NIFTy}, the user can choose between a non-parametric kernel $\sqrt{P}$ and a Matérn kernel $\sqrt{P}$ [@Arras2022; @Guardiani2022 for details on their implementation].
An example initializing a non-parameteric GP prior for a $128 \times 128$ space with unit volume is shown in the following.

```python
from nifty8 import re as jft

dims = (128, 128)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(offset_mean=2, offset_std=(1e-1, 3e-2))
cfm.add_fluctuations(  # Axis over which the kernle is defined
  dims,
  distances=tuple(1.0 / d for d in dims),
  fluctuations=(1.0, 5e-1),
  loglogavgslope=(-3.0, 2e-1),
  flexibility=(1e0, 2e-1),
  asperity=(5e-1, 5e-2),
  prefix="ax1",
  non_parametric_kind="power",
)
correlated_field = cfm.finalize()  # forward model for a GP prior
```

Not all problems are well described by regularly spaced pixels.
For more complicated spaced pixels, \texttt{NIFTy.re} features Iterative Charted Refinement [@Edenhofer2022], a GP model for arbitrarily deformed spaces.
This model exploits nearest neighbor relations on various coarsening of the discretized modeled space and runs very efficiently on GPUs.
See the demonstration scripts in the repository for an example.

## Models as PyTrees

Models are rarely just a GP prior.
Commonly, a model contains at least several non-linearities that transform the GP prior or combine it with other random variables.
For building more complex models, \texttt{NIFTy.re} provides a `Model` class that offers a somewhat familiar object-oriented design yet is fully JAX compatible and functional under the hood.
By inheriting from `Model`, a class is registered as a PyTree in JAX.
The following code showcases such a model that builds up a slightly more involved model using the objects from the previous example.

```python
from jax import numpy as jnp


class Forward(jft.Model):
  def __init__(self, correlated_field):
    self._cf = correlated_field
    # Track a method with which a random input for the model. This is not
    # strictly required but is usually handy when building deep models.
    super().__init__(init=correlated_field.init)

  def __call__(self, x):
    # NOTE, any kind of masking of the output, non-linear and linear
    # transformation could be carried out here. Models can also combined and
    # nested in any way and form.
    return jnp.exp(self._cf(x))


forward = Forward(correlated_field)

data = jnp.load("data.npy")
lh = jft.Poissonian(data).amend(forward)
```
<!-- How much should we expect the reader to know about jax? I think even for many people of the IFT Group this section might quite technical. Should we briefly explain that tracing, inlining and (non)-static mean? Is it clear what a pytree is?   -->

All GP models in \texttt{NIFTy.re} as well as all likelihoods are registered as PyTrees and can be traced by JAX.
Thus, `correlated_field`, `forward`, and `lh` from the code snippets shown here are all PyTrees in JAX and, e.g., the following is valid code `jax.jit(lambda l, x: l(x))(lh, x0)` with `x0` some arbitrarily chosen valid input to `lh`.
Inspired by equinox [@Kidger2021], individual attributes of the class can be marked as static or non-static via `dataclass.field(metadata=dict(static=...))`.
Depending on the value, JAX will either trace through the attribute under just-in-time compiles or treat them as static and inline them.
This mechanism is extensively used in likelihoods to avoid inlining large constants such as the data and avoiding expensive re-compiles whenever possible.

## Variational Inference

\texttt{NIFTy.re} is built for models with millions to billions of degrees of freedom.
To probe the posterior efficiently, \texttt{NIFTy.re} relies on VI.
Specifically, \texttt{NIFTy.re} utilizes Metric Gaussian Variational Inference (MGVI) and its successor geometric Variational Inference (geoVI) [@Knollmueller2019 @Frank2021 @Frank2022].
At the core of both MGVI and geoVI lies an alternating procedure in which we switch between optimizing the Kullback–Leibler divergence for a specific shape of the variational posterior and updating the shape of the variational posterior.
MGVI and geoVI define the variational posterior via samples, specifically, via samples drawn around an expansion point.
The samples in MGVI and geoVI exploit model-intrinsic knowledge of the posterior's approximate shape, encoded in the Fisher information metric and the prior curvature [@Frank2021].

\texttt{NIFTy.re} implements both MGVI and geoVI and allows for much finer control over the way samples are drawn and updated compared to \texttt{NIFTy}.
Furthermore, \texttt{NIFTy.re} exposes stand-alone functions for drawing MGVI and geoVI samples from any arbitrary model with a likelihood from \texttt{NIFTy.re} and a forward model that is differentiable by JAX.
In addition to stand-alone sampling functions, \texttt{NIFTy.re} also provides tools to configure and execute the alternating Kullback–Leibler divergence optimization and sample adaption at a higher abstraction level.
These tools are provided in a jaxopt-style optimizer class [@Blondel2021].

A typical minimization with \texttt{NIFTy.re} is shown in the following.
It retrieves six antithetically mirrored samples from the approximate posterior via 25 iterations of alternating between optimization and sample adaption.
The final result is stored in the `samples` variable.
A convenient one-shot wrapper for the below is `jft.optimize_kl`.
By virtue of all modeling tools in \texttt{NIFTy.re} being written in JAX, it is also possible to combine \texttt{NIFTy.re} tools with blackjax [@blackjax2020] or any other posterior sampler in the JAX ecosystem.

```python
from jax import random

key = random.PRNGKey(42)
key, sk = random.split(key, 2)
# NIFTy is agnostic w.r.t. the type of input it gets as long as it supports core
# arithmetic properties. Tell NIFTy to treat our parameter dictionary as a
# vector.
samples = jft.Samples(pos=jft.Vector(lh.init(sk)), samples=None, keys=None)

delta = 1e-4
absdelta = delta * jft.size(samples.pos)

opt_vi = jft.OptimizeVI(lh, n_total_iterations=25)
opt_vi_st = opt_vi.init_state(
  key,
  # Typically on the order of 2-12
  n_samples=lambda i: 1 if i < 2 else (2 if i < 4 else 6),
  # Arguments for the conjugate gradient method used to drawing samples from
  # an implicit covariance matrix
  draw_linear_kwargs=dict(
    cg_name="SL", cg_kwargs=dict(absdelta=absdelta / 10.0, maxiter=100)
  ),
  # Arguements for the minimizer in the nonlinear updating of the samples
  nonlinearly_update_kwargs=dict(
    minimize_kwargs=dict(
      name="SN", xtol=delta, cg_kwargs=dict(name=None), maxiter=5
    )
  ),
  # Arguments for the minimizer of the KL-divergence cost potential
  kl_kwargs=dict(minimize_kwargs=dict(name="M", xtol=delta, maxiter=35)),
  sample_mode=lambda i: "nonlinear_resample" if i < 3 else "nonlinear_update",
)
for i in range(opt_vi.n_total_iterations):
  print(f"Iteration {i+1:04d}")
  # Continuously updates the samples of the approximate posterior distribution
  samples, opt_vi_st = opt_vi.update(samples, opt_vi_st)
  print(opt_vi.get_status_message(samples, opt_vi_st))
```

![Data (left), posterior mean (middle), and posterior uncertainty (right) for a simple toy example.\label{fig:minimal_reconstruction_data_mean_std}](minimal_reconstruction_data_mean_std.png)

\autoref{fig:minimal_reconstruction_data_mean_std} shows an exemplary posterior reconstruction employing the above model.
The posterior mean agrees with the data but completely removes the noisy structures.
Differences between the posterior mean and the data are on the order of the posterior standard deviation.

## Performance compared to old NIFTy

We test the performance of \texttt{NIFTy.re} against \texttt{NIFTy} for the simple yet representative model from above.
To assess the performance, we benchmark the $M_p \coloneqq F_p + \mathbb{1}$ with $F_p$ denoting the Fisher metric of the overall likelihood at position $p$ and $\mathbb{1}$ the analogous curvature of the prior.
Within \texttt{NIFTy.re} the Fisher metric of the overall likelihood is decomposed into $J_{f,p}^\dagger N^{-1} J_{f,p}$ with $J_{f,p}$ the implicit Jacobian of the forward model $f$ at $p$ and $N$ the Fisher-metric of the Poisson likelihood itself.
We choose to benchmark $M_p$ as a typical minimization in \texttt{NIFTy.re} and \texttt{NIFTy} is dominated by calls to this function.
These calls are essential in both the sampling and the approximate second order minimization. <!-- For shortening the paper we could remove this senctence, as it is here not esential and to make the reader really understand why the metric is dominating the runtime further explanations would anywayes be necesary   -->

![Performance comparison of \texttt{NIFTy.re} on the CPU (one thread of an Intel Xeon Platinum 8358 CPU clocked at 2.60G Hz) and GPU (A100 SXM4 80 GB HBM2) versus \texttt{NIFTy}.\label{fig:benchmark_nthreads=1_devices=cpu+gpu}](benchmark_nthreads=1_devices=cpu+gpu.png)

\autoref{fig:benchmark_nthreads=1_devices=cpu+gpu} shows the evaluation time in \texttt{NIFTy} for applying $M_p$ to a new tangent position and the evaluation time in \texttt{NIFTy.re} for building $M_p$ and applying it to a new tangent position for exponentially larger models.
The benchmarks uses one thread on a compute-node with an Intel Xeon Platinum 8358 CPU clocked at 2.60G Hz and an NVIDIA A100 SXM4 80 GB HBM2 GPU.<!-- I thing here you should exlain why you use only one thread, maybe also use more threads, at least for the larger models   -->
We vary the size of the model by increasing the size of two-dimensional square image grid.
We choose to exclude the build time of $M_p$ in \texttt{NIFTy} from the comparison, putting \texttt{NIFTy} at an advantage, as \texttt{NIFTy} incurs a noticeable overhead for the assembly of $M_p$ that is typically only required on roughly every tenth evaluation of $M_p$ as $p$ is varied less often than the tangent position in a typical minimization. <!-- I think this might be to techincal for the reader. Also it is not clear what the message is you want the reader to understand. Do you want to reader to think this is a realistic benchmark, or do you want the reader to think that in practiacal applications there will more overhead in nifty that significantly slows? I think you expect to much from the reader to hestimate himself how important the overhead of building the metric is.  -->

For small image sizes, \texttt{NIFTy.re} on the CPU is about one order of magnitude faster than \texttt{NIFTy}.
Both reach about the same performance at an image size of roughly 15,000 pixels and continue to perform roughly the same for larger image sizes.
\texttt{NIFTy.re} on the GPU is consistently about one to two orders of magnitude faster than \texttt{NIFTy} for images larger than 100,000 pixels.

We believe the performance benefits of \texttt{NIFTy.re} on the CPU stem from the reduced python overhead by just-in-time compiling computations.
At image sizes larger than roughly 15,000 pixels, both evaluation times are dominated by the fast Fourier transform and are hence the same as both use the same underlying implementation [@ducc0].
Typical models in \texttt{NIFTy.re} and \texttt{NIFTy} are often well aligned with GPU programming models and thus consistently perform well on the GPU.
Models such as the new GP model implemented in \texttt{NIFTy.re} are even better aligned with GPU programming models and yield even higher performance gains, see @Edenhofer2022.

# Conclusion

We implemented the core GP and VI machinery of the Bayesian imaging package \texttt{NIFTy} in JAX.
The re-write moves much of the heavy-lifting from home-grown solutions into JAX, and we envision significant gains in maintainability of \texttt{NIFTy.re} and a faster development cycle moving forward.
The re-write accelerates typical models written in \texttt{NIFTy} by one to two orders of magnitude, lays the foundation for new kinds of inference machineries by enabling higher order derivates via JAX, and enables the interoperability of \texttt{NIFTy} with the VI and GP methods from the JAX/XLA Machine Learning ecosystem.

# Acknowledgements

Gordian Edenhofer acknowledges support from the German Academic Scholarship Foundation in the form of a PhD scholarship (``Promotionsstipendium der Studienstiftung des Deutschen Volkes'').
Philipp Frank acknowledges funding through the German Federal Ministry of Education and Research for the project "ErUM-IFT: Informationsfeldtheorie für Experimente an Großforschungsanlagen" (Förderkennzeichen: 05D23EO1).
Jakob Roth acknowledges financial support by the German Federal Ministry of Education and Research (BMBF) under grant 05A20W01 (Verbundprojekt D-MeerKAT).

# References

<!-- Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"
# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }
-->
