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
With \texttt{NIFTy.re} the GP models and the VI machinery is now fully accessible in JAX ecosystem and \texttt{NIFTy.re} components interact seamlessly with other JAX packages such as `blackjax` and  `jaxopt` [@Blondel2021].

# Core Components

\texttt{NIFTy.re} brings tried and tested structured GP models and VI algorithms to JAX.
* Imaging requires GP models, inference requires VI
* \texttt{NIFTy.re} treats this problem as a Bayesian inference task.
* It implements MGVI and geoVI TODO:cite_both.

* \texttt{NIFTy} model from a Machine Learning perspective reads `loss(d, f(\theta)) + reg(\theta)`
* From a Bayesian perspective the reduction `loss(d, f(\theta))` represents the likelihood and the regularization `reg(\theta)` the prior.
* Formulation is degenerate between what is considered part of the likelihood/loss and what is considered a regularization/prior.
* Without loss of generalizty \texttt{NIFTy} formulates model such that the regularization/prior always is an L_2 regularization respectively a standard Gaussian prior.
* This means that all relevant details of the model are encoded in the forward model `f` and the loss `loss`
* \texttt{NIFTy} is all about building up the forward model `f` and choosing a suitable loss `loss`

## GP

* \texttt{NIFTy.re} brings \texttt{NIFTy}'s structured GP models to JAX.
* The models rely for reconstructions with regularly spaced pixels
* In NIFTy's generative picture: GP realization $s = \mathcal{FT} \cdot \sqrt{P} \cdot \xi$
* Stationary kernels with an accesible harmonic space can be straight-forwadly implemented this way
* User can choose-between a non-parametric kernel or a Matern kernel [@Arras2022; @Guardiani2022 for details on their implementation].

* New GP model called ICR for reconstructions with arbitrarily deformed pixel spacings
* Recap ICR

```python
from nifty import re as jft

jft.CorrelatedFieldMaker()
# TODO
```

## Models as PyTrees

* Object-oriented model as PyTree
* Attributes can be marked as jax-traceable or can be hidden from jax (idea of using python's `dataclasses` library is heavily inspired by equinox)
* Likelihood as PyTrees

```python
from jax import numpy as jnp

class Forward(Model):
  def __init__(self, correlated_field):
    self._cf = correlated_field

  def __call__(self, x):
    # NOTE, any kind of masking of the output, non-linear and linear
    # transformation could be carried out here. Models can also combined and
    # nested in any for that suites the program flow.
    return jnp.exp(self._cf(x))

forward = Forward(cf)
lh = TODO
```

* As `cf`, `forward`, and `lh` are all PyTrees in JAX, the following is valid code `jax.jit(lambda l, x: l(x))(lh, x0)`.
* By registering fields of models as dataclass.field and setting the `static` key in the metadata to `False`, we can control what gets inlined and what is traced by JAX.
* This mechanism is extensively made use of in likelihoods to avoid inlining large constants and avoiding re-compiles.

## VI

* highlight need for VI
* recap MGVI and geoVI as expansion-point inference algorithms

* Completely functional sampling and jaxopt-style minimization

```python
jft.OptimizeVI()  # TODO
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
