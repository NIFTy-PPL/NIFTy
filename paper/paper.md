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
* We present a re-write of NIFTy, coined \texttt{NIFTy.re}, which bridges \texttt{NIFTy} to Machine Learning ecosystem, reworks the modeling principle, the inference strategy, and outsources much of the heavy lifting to JAX to easy future developments.
* The re-write dramatically accelerated models written in NIFTy, lays the foundation for a new kind of inference machinery, and enables the interoperability of \texttt{NIFTy} with the JAX/XLA Machine Learning ecosystem.

# Statement of Need

Imaging commonly involves millions to billions of pixel.
Each pixel usually corresponds to one or more correlated degree of freedom in the model space.
Modeling millions to billions of degrees of freedom is computationally demanding.
Yet, imaging is not only computationally demanding but also statistically challenging.
The noisy input requires a statistical treatment and needs to be accurately propagates from input to the final image.
Thus, to infer an image from noisy data, we require an inference machine that not only handles the modeling of millions to billions of degrees of freedom but one that does so in a statistically rigorous way.

\texttt{NIFTy} is a Bayesian imaging library [@Selig2013; @Steiniger2017; @Arras2019].
It is designed to infer the million to billion dimensional posterior distribution in the image space from noisy input data.
At the core of \texttt{NIFTy} lies a set of powerful Gaussian Process (GP) models for modeling correlated degrees of freedom in high dimensions and an accurate statistical inference machinery of Variational Inference (VI) algorithms.

\texttt{NIFTy.re} is a rewrite of \texttt{NIFTy} in JAX [@Jax2018] with all relevant previous GP models, a new even more powerful GP model, and a much more flexible posterior approximation machinery.
By virtue of being written in JAX, \texttt{NIFTy.re} effortlessly runs on the accelerator hardware such as the GPU, extensively vectorizes models whenever possible, just-in-time compiles code for additional performance, and enables a new kind of inference machinery thanks to being able to retrieve higher order derivates.
By switching from a home-grown automatic differentiation engine to JAX, we envision to harness significant gains in maintainability of \texttt{NIFTy} moving forward and a faster development cycle for new features.

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->
We expect \texttt{NIFTy.re} to be highly useful for many imaging applications and envision many applications within and outside of astrophysics [@Arras2022; @Leike2019; @Leike2020; @Mertsch2023; @Roth2023DirectionDependentCalibration; @Hutschenreuter2023; @Tsouros2023; @Roth2023FastCadenceHighContrastImaging; @Hutschenreuter2022].
\texttt{NIFTy.re} has already been successfully used in two galactic tomography publications [@Leike2022; @Edenhofer2023].
A very early version of \texttt{NIFTy.re} enabled a 100 billion dimensional reconstruction using a maximum posterior inference.
In a newer publication, \texttt{NIFTy.re} was used to infer a 500 million dimensional posterior dimensional using VI [@Knollmueller2019].
Both publications extensively use \texttt{NIFTy.re}'s GPU support which yielded order of magnitude speed-ups.
With \texttt{NIFTy.re} bridging \texttt{NIFTy} to JAX, we envision many new possibilities for inferring classical Machine Learning models with \texttt{NIFTy}'s inference methods and using \texttt{NIFTy} components such as the GP models in classical neural networks frameworks.

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->
\texttt{NIFTy.re} competes with other GP libraries as well as with probabilistic programming languages and frameworks.
Compared to GPyTorch [@Hensman2015], GPflow [@Matthews2017], george [@Sivaram2015], or TinyGP [@ForemanMackey2024], \texttt{NIFTy} and \texttt{NIFTy.re} focus on GP models for structured spaces.
These spaces can be arbitrarily deformed in the new GP model implemented in NIFTy.re [@Edenhofer2022].
Compared to classical probabilistic programming languages such as Stan [@Carpenter2017] and frameworks such pyro [@Bingham2019], numpyro [@Phan2019], pyMC3 [@Salvatier2016], Emcee [@ForemanMackey2013], dynesty [@Speagle2020; @Koposov2023], or blackjax [@blackjax2020], \texttt{NIFTy} and \texttt{NIFTy.re} focus on high dimensional inference with millions to billions of degrees of freedom.
In contrast to many GP libraries, nether \texttt{NIFTy} nor \texttt{NIFTy.re} assume the posterior to be analytically accessible and instead try to approximate it using VI.
With \texttt{NIFTy.re} the GP models and the VI machinery is now fully accessible in JAX ecosystem and \texttt{NIFTy.re} components interact seamlessly with other JAX packages such as `blackjax` and `jaxopt` [@Blondel2021].

# Core Components

\texttt{NIFTy.re} brings tried and tested structured GP models and VI algorithms to JAX.
The first is essential for modeling in the domain of imaging and the other is essential to probe the high dimensional posterior.
\texttt{NIFTy.re} treats the imaging problem as a Bayesian inference problem.
\texttt{NIFTy.re} infers the parameters of interest, often an image, from noisy data and a stochastic mapping from the parameters of interest to the data a.k.a. a model.

\texttt{NIFTy} and \texttt{NIFTy.re} build up hierarchical models for the posterior.
The log-posterior function reads $\ln\mathcal{p(\theta|d)} \coloneqq \mathcal{l}(d, f(\theta)) + \ln\mathcal{p}(\theta) + \mathrm{const}$ with log-likelihood $\mathcal{l}$, forward model $f$ mapping the parameters of interest $\theta$ to the data space, and log-prior $\ln\mathcal{p(\theta)}$.
The goal of the inference is to draw samples from the posterior $\mathcal{p}(\theta|d)$.

What is considered part of the likelihood and what is considered part of the prior is ill-defined.
Without loss of generality \texttt{NIFTy} and \texttt{NIFTy.re} formulate models such that the prior always is a standard Gaussian.
This choice of re-parameterization [@Rezende2015] is called standardization.
All relevant details of the prior model are encoded in the forward model $f$.
The standardization is often carried out implicitly in the background, however, for prior models outside of the current toolbox the user has to manually implement the necessary component for $f$ that corresponds to the desired non-Gaussian prior.

## Gaussian Processes

One standard tool from the \texttt{NIFTy.re} toolbox are the structured GP models from \texttt{NIFTy}.
The models rely on the harmonic domain being easily accessible, e.g. for pixels spaced on a regular Cartesian grid, the natural choice to represent a stationary kernel is the Fourier domain.
In the generative picture, a realization $s$ drawn from a GP then reads $s = \mathcal{HT} \cdot \sqrt{P} \cdot \xi$ with $\mathcal{HT}$ the harmonic transform, $\sqrt{P}$ the square-root of the power-spectrum in harmonic space as diagonal operator, and $\xi$ standard Gaussian random variables.
In the implementation in \texttt{NIFTy.re} and \texttt{NIFTy}, user can choose-between a non-parametric kernel $\sqrt{P}$ or a Matérn kernel $\sqrt{P}$ [@Arras2022; @Guardiani2022 for details on their implementation].
An example initializing a non-parameteric GP prior for a $128 \times 128$ space with unit volume is shown in the following.

```python
from nifty import re as jft

dims = (128, 128)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(offset_mean=0., offset_std=(1e-3, 1e-4))
cfm.add_fluctuations(  # Axis over which the kernle is defined
  dims,
  distances=tuple(1. / d for d in dims),
  fluctuations=(1e-1, 5e-3),
  loglogavgslope=(-1., 1e-2),
  flexibility=(1e+0, 5e-1),
  asperity=(5e-1, 5e-2),
  prefix="ax1",
  non_parametric_kind="power"
)
correlated_field = cfm.finalize()  # forward model for a GP prior
```

Not all problems are well described by regularly spaced pixels.
For more complicated spaced pixels, \texttt{NIFTy.re} features @Edenhofer2022, a GP model for arbitrarily deformed spaces.
This model exploits nearest neighbor relations on various coarsening of the discretized modeled space and runs very efficiently on GPUs.
See the demonstration scripts in the repository for an example.

## Models as PyTrees

Models are rarely just a GP prior.
Commonly a model contains at least several non-linearity that transforms the GP prior or combines it with other random variables.
To built up more involved models, \texttt{NIFTy.re} provides a `Model` class that a somewhat familiar object-oriented design yet is fully JAX compatible and functional under the hood.
By inheritng from `Model`, a class is registered as a PyTree in JAX.
Individual attrributes of the class can be marked as stitic or non-static via `dataclass.field(metadata=dict(static=...))`.
Depending on the value, JAX will either trace through the attribute under just-in-time compiles or hides from JAX and treats them as static.
The dataclass-style implementation took inspiration from equinox [@Kidger2021].

```python
from jax import numpy as jnp

class Forward(Model):
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

forward = Forward(cf)

data = np.load("")  # TODO
noise_cov_inv = np.load("")  # TODO
lh = jft.Gaussian(data, noise_cov_inv).amend(forward)
```

All GP models in \texttt{NIFTy.re} as well as all likelihoods are registered as PyTrees and can be traced by JAX.
Thus, `cf`, `forward`, and `lh` from the preceding code snippet are all PyTrees in JAX and the following is valid code `jax.jit(lambda l, x: l(x))(lh, x0)` with `x0` some arbitrarily chosen valid input to `lh`.
By registering fields of `Forward` via `dataclass.field` and setting the `static` key in the metadata to `False`/`True`, we can control what gets in-lined and what is traced by JAX.
This mechanism is extensively used in likelihoods to avoid in-lining large constants such as the data and avoiding expensive re-compiles whenever possible.

## Variational Inference

\texttt{NIFTy.re} is build for models with millions to billions of degrees of freedoms.
To probe the posterior efficiently, \texttt{NIFTy.re} relies on VI.
Specifically, \texttt{NIFTy.re} utilizes Metric Gaussian Variational Inference (MGVI) and its successor geometric Variational Inference (geoVI) [@Knollmueller2019 @Frank2021 @Frank2022].
At the core of both MGVI and geoVI lies an alternating procedure in which we switch between optimizing the Kullback–Leibler divergence for a specific shape of variational posterior and updating the shape of the variational posterior.
Both MGVI and geoVI define the variational posterior via samples, specifically, via samples that are drawn around an expansion point.
The samples in MGVI and geoVI exploit model-intrinsic knowledge of the approximate shape of the posterior which is encoded in the Fisher information metric and the prior curvature [@Frank2021].

\texttt{NIFTy.re} implements both MGVI and geoVI and allows for much finer control over the way samples are drawn and/or updated compared to \texttt{NIFTy}.
Furthermore, \texttt{NIFTy.re} exposes stand-alone functions for drawing MGVI and geoVI samples from any arbitrary model with a likelihood from \texttt{NIFTy.re} and a forward model that is differentiable by JAX.
In additiona to stand-alone sampling functions, \texttt{NIFTy.re} also provides tools at a higher abstractions to configure and execute the alternating Kullback–Leibler divergence optimiztion and sample adaption.
These tools are provided in a jaxopt-style optimizer class [@Blondel2021].

A typical minimization with \texttt{NIFTy.re} is shown in the following.
It retrieves six antithetically mirrored samples from the approximate posterior via 25 iterations of alternating between optimization and sample adaption.
The final result is stored in the `samples` variable.
A convenient one-shot wrapper for the below is `jft.optimize_kl`.
By virtue of all modeling tools in \texttt{NIFTy.re} being written in JAX, it is also possible to combine \texttt{NIFTy.re} tools with blackjax [@blackjax2020] or any other posterior sampler in the JAX ecosystem.

```python
# Initialize an empty jft.Samples class, `OptimizeVI.update`
samples = jft.Samples(pos=initial_pos, samples=None, keys=None)
opt_vi = jft.OptimizeVI(lh, n_total_iterations=25)
opt_vi_st_init = opt_vi.init_state(
  key,
  # Typically on the order of 2-12
  n_samples=lambda i: 1 if i < 2 else (2 if i < 4 else 6),
  # Arguments for the conjugate gradient method used to drawing samples from
  # an implicit covariance matrix
  draw_linear_kwargs=dict(
    cg_name="SL", cg_kwargs=dict(absdelta=absdelta / 10., maxiter=100)
  ),
  # Arguements for the minimizer in the nonlinear updating of the samples
  nonlinearly_update_kwargs=dict(
    minimize_kwargs=dict(
      name="SN",
      xtol=delta,
      cg_kwargs=dict(name=None),
      maxiter=5,
    )
  ),
  # Arguments for the minimizer of the KL-divergence cost potential
  kl_kwargs=dict(
    minimize_kwargs=dict(
      name="M", absdelta=absdelta, cg_kwargs=dict(name="MCG"), maxiter=35
    )
  ),
  sample_mode=lambda i: "nonlinear_resample" if i < 3 else "nonlinear_update",
)
for i in range(opt_vi.n_total_iterations):
  print(f"Iteration {i+1:04d}")
  # Continuously updates the samples of the approximate posterior distribution
  samples, opt_vi_st = opt_vi.update(samples, opt_vi_st)
  print(opt_vi.get_status_message(samples, opt_vi_st))
```

## Performance compared to old NIFTy

* Test performance of \texttt{NIFTy.re} versus \texttt{NIFTy} on a simple 2D log-normal model with varying dimensions
* Model reads $d = \rho + n$ with $\rho=e^s$ and $s$ 2D GP with a homogeneous and stationary kernel model from TODO:cite_M87 and $n$ white Gaussian noise
* Ensure that \texttt{NIFTy.re} and \texttt{NIFTy} model agree up to numerical precision
* A typical minimization in \texttt{NIFTy} is dominated by calls to the $M \coloneqq J_\rho^\dagger N J_\rho + 1$ with $J$ the implicit Jacobian of the model and $N$ the covariance of the noice.
* These calls are both essential for the sampling and the approximate second order minimization underpinning NIFTy
* Thus, we compare the performance of $M$ for our model in \texttt{NIFTy.re} and NIFTy

* TODO: do performance benchmark and insert figure

* Figure shows performance of \texttt{NIFTy} versus \texttt{NIFTy.re} on the the CPU as well as versus \texttt{NIFTy.re} running on the GPU
* TODO: describe
* For a simple log-normal \texttt{NIFTy.re} is faster by TODO on the CPU and TODO on the GPU.
* For methods that are better optimized for the GPU e.g. ICR, the performance gain can be even larger

# Conclusion

* \texttt{NIFTy} is faster, foundation for new inference machinery with higher order derivates, better maintainable

# Acknowledgements

Gordian Edenhofer acknowledges support from the German Academic Scholarship Foundation in the form of a PhD scholarship (``Promotionsstipendium der Studienstiftung des Deutschen Volkes'').
Philipp Frank acknowledges funding through the German Federal Ministry of Education and Research for the project "ErUM-IFT: Informationsfeldtheorie für Experimente an Großforschungsanlagen" (Förderkennzeichen: 05D23EO1).

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
