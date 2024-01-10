---
title: 'Re-Envisioning Numerical Information Field Theory (NIFTy.re)'
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

<!-- The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration). -->

* Imaging at the most basic level is the process of transforming noisy, incomplete data into a space that humans can interpret.
* NIFTy is a Bayesian imaging framework that propagates the statistical uncertainty in the data and the model to the image domain.
* NIFTy has already successfully been applied to the fields of radio astronomy, galactic tomography, and observational cosmology.
* Ever larger models and previous design decisions and modeling principles, held the performance and the development of new inference strategies in NIFTy back.
* We present a re-write of NIFTy, coined NIFTy.re, which bridges NIFTy to Machine Learning ecosystem, reworks the modeling principle, the inference strategy, and outsources much of the heavy lifiting to JAX to easy future developments.
* The re-write dramatically accelerated models written in NIFTy and enables the interoperability of NIFTy with the JAX Machine Learning ecosystem.

# Statement of need

TODO: make it sound less like a NIFTy rewrite but rather an independent package!
* NIFTy.re is a library for Gaussian Processes and Variational Inference based on JAX.
* It makes many of the algorithms from NIFTy available on the GPU, enables vectorizing models over parameters, and taking second order derivatives for optimization.
* GPU already enabled much faster model evaluations, and we envision that higher order derivates will allow probing new inference techniques~(TODO:cite: Riemmanien manifold HMC).
* Many new possiblities by bridging NIFTy algorithms to JAX ecosystem.

* NIFTy.re extends the NIFTy for modeling Gaussian Processes on structured spaces and infering the parameters
* Core component of NIFTy.re are the GP models and the VI methods.
* Both are now accesible and integrate into other JAX packages such as `blackjax`(TODO:cite), `numpyro`(TODO:cite), and `jaxopt`(TODO:cite).

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->
* Competes with other GP libraries as well as with probabilistic programming languages and frameworks.
* Compared to Gpytorch, GPflow, george, tinygp
* Compared to classical propabilistic programming languages and or frameworks such as Stan, pyro, pyMC3, Emcee, dynesty, numpyro, or blackjax (TODO:cite), NIFTy has a very strong focus on high dimensional inference with 1M+ to hundreds of billions of parameters, and features a set of models that quickly these number of parameters: GPs

<!-- `Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`). -->

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->
* NIFTy.re is envisioned to be used for many imaging applications
* A very early version `NIFTy.re` enabled the 100B dimensional reconstruction and was recently used to infer 500M dimensional inference problem.

<!-- `Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike. -->

# Core Components

* NIFTy.re brings a Harmonic Space based GP model from NIFTy to JAX (CITE: ARRAS paper) for reconstructions with regularly spaced pixels and a new GP model called ICR for reconstructions with arbitrarily deformed pixel spacings.
* Imaging usually has as many or more degrees of freedom as pixels in the image which is on the order of tens or hundreds of million.
* NIFTy.re treats this problem as a Bayesian inference task.
* It implements MGVI and geoVI.

## GP

* highlight need for GP and applications
* background for GP and other ppl's GP libraries
* recap GP for structured spaces
  * recap FT GP
  * recap ICR

## VI

* describe NIFTy model of `red(d, f(\theta)) + reg(\theta)`

* highlight need for VI
* recap other VI tools and in which way NIFTy is different
* recap MGVI, recap geoVI

* minimization; implement scipy style minimize for NCG because we already have access to the curvature

# Acknowledgements

Gordian Edenhofer acknowledges support from the German Academic Scholarship Foundation in the form of a PhD scholarship (``Promotionsstipendium der Studienstiftung des Deutschen Volkes'').

# References

<!-- Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

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
