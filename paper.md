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
* NIFTy is a Bayesian imaging framework that propagates the statistical uncertainty in the data and the model to the image domain.
* NIFTy has already successfully been applied to the fields of radio astronomy, galactic tomography, and observational cosmology.
* A focus on CPU computing and previous design decisions, held the performance and the development of new inference methods in NIFTy back.
* We present a re-write of NIFTy, coined NIFTy.re, which bridges NIFTy to Machine Learning ecosystem, reworks the modeling principle, the inference strategy, and outsources much of the heavy lifting to JAX to easy future developments.
* The re-write dramatically accelerated models written in NIFTy, lays the foundation for a new kind of inference machinery, and enables the interoperability of NIFTy with the JAX/XLA Machine Learning ecosystem.

# Statement of Need

* Imaging commonly encounters millions to billions of parameters as each pixel in an image can be considered an independent degree-of-freedom.
* Modeling millions to billions of degrees-of-freedom is tricky
* The inference of a human-accesible image from noisy requires inference machine to work with millions to billions of paramters
* NIFTy infers "truth" including its uncertainty from noisy data for million to billion dimensional problems

* NIFTy.re is a library for Gaussian Processes and Variational Inference based on JAX.
* It makes many of the algorithms from NIFTy available on the GPU, enables vectorizing models over parameters, and taking second order derivatives for optimization.
* GPU already enabled much faster model evaluations, we envision that higher order derivates will allow probing new inference techniques~(TODO:cite: Riemmanien manifold HMC), and better maintanability for the next decade in astrophysics research

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->
* NIFTy.re is envisioned to be used for many imaging applications
* A very early version `NIFTy.re` enabled the 100B dimensional reconstruction and was recently used to infer 500M dimensional inference problem.
* TODO:cite_all_of_NIFTY papers
* Many new possiblities of inferring Machine Learning models with NIFTy inference methods and using NIFTy components such as the GP models in NN

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->
* Competes with other GP libraries as well as with probabilistic programming languages and frameworks.
* Compared to Gpytorch, GPflow, george, tinygp
* Compared to classical propabilistic programming languages and or frameworks such as Stan, pyro, pyMC3, Emcee, dynesty, numpyro, or blackjax (TODO:cite), NIFTy has a very strong focus on high dimensional inference with 1M+ to hundreds of billions of parameters, and features a set of models that quickly these number of parameters: GPs
* Middle ground between tools like blackjax and pymc
* NIFTy's GP and VI are now accesible and integrate into other JAX packages such as `blackjax`(TODO:cite), `numpyro`(TODO:cite), and `jaxopt`(TODO:cite).

# Core Components

* NIFTy.re brings Harmonic Space based GP models from NIFTy to JAX (CITE:M87_paper) for reconstructions with regularly spaced pixels and a new GP model called ICR for reconstructions with arbitrarily deformed pixel spacings.
* Imaging usually has as many or more degrees of freedom as pixels in the image which is on the order of tens or hundreds of million.
* NIFTy.re treats this problem as a Bayesian inference task.
* It implements MGVI and geoVI TODO:cite_both.

* NIFTy model from a Machine Learning perspective reads `loss(d, f(\theta)) + reg(\theta)`
* From a Bayesian perspective the reduction `loss(d, f(\theta))` represents the likelihood and the regularization `reg(\theta)` the prior.
* Formulation is degenerate between what is considered part of the likelihood/loss and what is considered a regularization/prior.
* Without loss of generalizty NIFTy formulates model such that the regularization/prior always is an L_2 regularization respectively a standard Gaussian prior.
* This means that all relevant details of the model are encoded in the forward model `f` and the loss `loss`
* NIFTy is all about building up the forward model `f` and choosing a suitable loss `loss`

## GP

* Common component of a forward model in imaging is the Gaussian process
* highlight need for GP and applications
* recap GP for structured spaces
  * recap FT GP
  * recap ICR

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

* Test performance of NIFTy.re versus NIFTy on a simple 2D log-normal model with varying dimensions
* Model reads $d = \rho + n$ with $\rho=e^s$ and $s$ 2D GP with a homogeneous and stationary kernel model from TODO:cite_M87 and $n$ white Gaussian noise
* Ensure that NIFTy.re and NIFTy model agree up to numerical precision
* A typical minimization in NIFTy is dominated by calls to the $M \coloneqq J_\rho^\dagger N J_\rho + 1$ with $J$ the implicit Jacobian of the model and $N$ the covariance of the noice.
* These calls are both essential for the sampling and the approximate second order minimization underpinning NIFTy
* Thus, we compare the performance of $M$ for our model in NIFTy.re and NIFTy

* TODO: do performance benchmark and insert figure

* Figure shows performance of NIFTy versus NIFTy.re on the the CPU as well as versus NIFTy.re running on the GPU
* TODO: describe
* For a simple log-normal NIFTy.re is faster by TODO on the CPU and TODO on the GPU.
* For methods that are better optimized for the GPU e.g. ICR, the performance gain can be even larger

# Conclusion

* NIFTy is faster, foundation for new inference machinery with higher order derivates, better maintainable

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
