NIFTy - Numerical Information Field Theory
==========================================
[![pipeline status](https://gitlab.mpcdf.mpg.de/ift/nifty/badges/NIFTy_8/pipeline.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty/-/commits/NIFTy_8)
[![coverage report](https://gitlab.mpcdf.mpg.de/ift/nifty/badges/NIFTy_8/coverage.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty/-/commits/NIFTy_8)

**NIFTy** project homepage:
[https://ift.pages.mpcdf.de/nifty](https://ift.pages.mpcdf.de/nifty/index.html)

Summary
-------

### Description

**NIFTy**, "**N**umerical **I**nformation **F**ield **T**heor<strong>y</strong>", is
a versatile library designed to enable the development of signal
inference algorithms that operate regardless of the underlying grids
(spatial, spectral, temporal, …) and their resolutions.
Its object-oriented framework is written in Python, although it accesses
libraries written in C++ and C for efficiency.

NIFTy offers a toolkit that abstracts discretized representations of
continuous spaces, fields in these spaces, and operators acting on
these fields into classes.
This allows for an abstract formulation and programming of inference
algorithms, including those derived within information field theory.
NIFTy's interface is designed to resemble IFT formulae in the sense
that the user implements algorithms in NIFTy independent of the topology
of the underlying spaces and the discretization scheme.
Thus, the user can develop algorithms on subsets of problems and on
spaces where the detailed performance of the algorithm can be properly
evaluated and then easily generalize them to other, more complex spaces
and the full problem, respectively.

The set of spaces on which NIFTy operates comprises point sets,
*n*-dimensional regular grids, spherical spaces, their harmonic
counterparts, and product spaces constructed as combinations of those.
NIFTy takes care of numerical subtleties like the normalization of
operations on fields and the numerical representation of model
components, allowing the user to focus on formulating the abstract
inference procedures and process-specific model properties.


Installation
------------

Detailed installation instructions can be found in the NIFTy Documentation for:

- [users](http://ift.pages.mpcdf.de/nifty/user/installation.html)
- [developers](http://ift.pages.mpcdf.de/nifty/dev/index.html)

### Run the tests

To run the tests, additional packages are required:

    sudo apt-get install python3-pytest-cov

Afterwards the tests (including a coverage report) can be run using the
following command in the repository root:

    pytest-3 --cov=nifty8 test

### First Steps

For a quick start, you can browse through the [informal
introduction](https://ift.pages.mpcdf.de/nifty/user/code.html) or
dive into NIFTy by running one of the demonstrations, e.g.:

    python3 demos/getting_started_1.py

### Licensing terms

Most of NIFTy is licensed under the terms of the
[GPLv3](https://www.gnu.org/licenses/gpl.html) license while NIFTy.re is also licensed under the terms of the BSD-2-Clause license.
All of NIFTy is distributed *without any warranty*.

### Citing NIFTy

For the probabilistic programming framework NIFTy, please cite the following.

```
@misc{niftyre,
    author       = {{Edenhofer}, Gordian and {Frank}, Philipp and {Leike}, Reimar H. and {Roth}, Jakob and {Guerdi}, Massin and {Enßlin}, Torsten A.},
    title        = {{Re-Envisioning Numerical Information Field Theory (NIFTy.re): A Library for Gaussian Processes and Variational Inference}},
    keywords     = {Software},
    year         = 2024,
    howpublished = {in preparation}
}
```

Please also consider crediting the Gaussian Process models if you use them and the inference machinery. See [the corresponding entry on citing NIFTy in the documentation](https://ift.pages.mpcdf.de/nifty/user/citations.html) for further details.
