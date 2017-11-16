WARNING:
========

The code on this branch is not meant to be an official version of NIFTy.
As a consequence, it does not install as package "nifty", but rather as
"nifty2go", to allow parallel installation alongside the official NIFTy and
avoid any conflicts.


NIFTY - Numerical Information Field Theory
==========================================
[![build status](https://gitlab.mpcdf.mpg.de/ift/NIFTy/badges/nifty2go/build.svg)](https://gitlab.mpcdf.mpg.de/ift/NIFTy/commits/nifty2go)
[![coverage report](https://gitlab.mpcdf.mpg.de/ift/NIFTy/badges/nifty2go/coverage.svg)](https://gitlab.mpcdf.mpg.de/ift/NIFTy/commits/nifty2go)

**NIFTY** project homepage:
[http://www.mpa-garching.mpg.de/ift/nifty/](http://www.mpa-garching.mpg.de/ift/nifty/)

Summary
-------

### Description

**NIFTY**, "**N**umerical **I**nformation **F**ield **T**heor<strong>y</strong>", is
a versatile library designed to enable the development of signal
inference algorithms that operate regardless of the underlying spatial
grid and its resolution. Its object-oriented framework is written in
Python, although it accesses libraries written in C++ and C for
efficiency.

NIFTY offers a toolkit that abstracts discretized representations of
continuous spaces, fields in these spaces, and operators acting on
fields into classes. Thereby, the correct normalization of operations on
fields is taken care of automatically without concerning the user. This
allows for an abstract formulation and programming of inference
algorithms, including those derived within information field theory.
Thus, NIFTY permits its user to rapidly prototype algorithms in 1D, and
then apply the developed code in higher-dimensional settings of real
world problems. The set of spaces on which NIFTY operates comprises
point sets, *n*-dimensional regular grids, spherical spaces, their
harmonic counterparts, and product spaces constructed as combinations of
those.

### Class & Feature Overview

The NIFTY library features three main classes: **spaces** that represent
certain grids, **fields** that are defined on spaces, and **operators**
that apply to fields.

-   [Spaces](http://www.mpa-garching.mpg.de/ift/nifty/space.html)
    -   `RGSpace` - *n*-dimensional regular Euclidean grid
    -   `LMSpace` - spherical harmonics
    -   `GLSpace` - Gauss-Legendre grid on the 2-sphere
    -   `HPSpace` - [HEALPix](http://sourceforge.net/projects/healpix/)
        grid on the 2-sphere
-   [Fields](http://www.mpa-garching.mpg.de/ift/nifty/field.html)
    -   `Field` - generic class for (discretized) fields

<!-- -->

    Field.conjugate     Field.dim          Field.norm
    Field.vdot          Field.weight

-   [Operators](http://www.mpa-garching.mpg.de/ift/nifty/operator.html)
    -   `DiagonalOperator` - purely diagonal matrices in a specified
        basis
    -   `FFTOperator` - conversion between spaces and their harmonic
                        counterparts
    -   (and more)
-   (and more)

*Parts of this summary are taken from* [1] *without marking them
explicitly as quotations.*

Installation
------------

### Requirements

-   [Python](http://www.python.org/) (v2.7.x or 3.5.x)
    -   [NumPy](http://www.numpy.org/)

### Sources

The current version of Nifty3 can be obtained by cloning the repository:

    git clone https://gitlab.mpcdf.mpg.de/ift/NIFTy.git


### Installation via pip

It is possible to simply install NIFTy with all its dependencies via the command

pip install --user --process-dependency-links --egg git+https://gitlab.mpcdf.mpg.de/ift/NIFTy.git@nifty2go

### Running the tests

In oder to run the tests one needs two additional packages:

    pip install nose parameterized

Afterwards the tests (including a coverage report) are run using the following
command in the repository root:

    nosetests -x --with-coverage --cover-html --cover-package=nifty2go


### First Steps

For a quick start, you can browse through the [informal
introduction](http://www.mpa-garching.mpg.de/ift/nifty/start.html) or
dive into NIFTY by running one of the demonstrations, e.g.:

    python demos/wiener_filter_via_curvature.py

Acknowledgement
---------------

Please acknowledge the use of NIFTY in your publication(s) by using a
phrase such as the following:

> *"Some of the results in this publication have been derived using the
> NIFTY package [Selig et al., 2013]."*

### References

Release Notes
-------------

The NIFTY package is licensed under the
[GPLv3](http://www.gnu.org/licenses/gpl.html) and is distributed
*without any warranty*.

* * * * *

**NIFTY** project homepage:
[](http://www.mpa-garching.mpg.de/ift/nifty/)

[1] Selig et al., "NIFTY - Numerical Information Field Theory - a
versatile Python library for signal inference", [A&A, vol. 554, id.
A26](http://dx.doi.org/10.1051/0004-6361/201321236), 2013;
[arXiv:1301.4499](http://www.arxiv.org/abs/1301.4499)
