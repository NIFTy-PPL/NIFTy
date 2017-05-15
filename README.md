NIFTY - Numerical Information Field Theory
==========================================
[![build status](https://gitlab.mpcdf.mpg.de/ift/NIFTy/badges/master/build.svg)](https://gitlab.mpcdf.mpg.de/ift/NIFTy/commits/master)
[![coverage report](https://gitlab.mpcdf.mpg.de/ift/NIFTy/badges/master/coverage.svg)](https://gitlab.mpcdf.mpg.de/ift/NIFTy/commits/master)

**NIFTY** project homepage:
[http://www.mpa-garching.mpg.de/ift/nifty/](http://www.mpa-garching.mpg.de/ift/nifty/)

Summary
-------

### Description

**NIFTY**, "**N**umerical **I**nformation **F**ield **T**heor**y**", is
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
    Field.dot           Field.set_val      Field.weight

-   [Operators](http://www.mpa-garching.mpg.de/ift/nifty/operator.html)
    -   `DiagonalOperator` - purely diagonal matrices in a specified
        basis
    -   `ProjectionOperator` - projections onto subsets of a specified
        basis
    -   `PropagatorOperator` - information propagator in Wiener filter
        theory
    -   (and more)
-   (and more)

*Parts of this summary are taken from* [1] *without marking them
explicitly as quotations.*

Installation
------------

### Requirements

-   [Python](http://www.python.org/) (v2.7.x)
    -   [NumPy](http://www.numpy.org/)

### Download

The current version of Nifty3 can be obtained by cloning the repository:

    git clone https://gitlab.mpcdf.mpg.de/ift/NIFTy.git

and switching to the "master" branch:

    cd NIFTy
    git checkout master

### Installation on Ubuntu

This is for you if you want to install NIFTy on your personal computer
running with an Ubuntu-like linux system were you have root priviledges.
Starting with a fresh Ubuntu installation move to a folder like
`~/Downloads`:

-   Install basic packages like python, python-dev, gsl and others:

        sudo apt-get install curl git autoconf python-dev python-pip python-numpy

-   Install pyHealpix:

        git clone https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git
        cd pyHealpix
        autoreconf -i && ./configure --prefix=$HOME/.local && make -j4 && make install
        cd ..

-   Finally, NIFTy:

        git clone https://gitlab.mpcdf.mpg.de/ift/NIFTy.git
        cd NIFTy
        git checkout master
        python setup.py install --user
        cd ..

### Installation on Linux systems in general

Since all the "unconventional" packages (i.e. pyHealpix and NIFTy) listed in the
section above are installed
within the home directory of the user, the installation instructions for these
should also work on any Linux machine where you do not have root access.
In this case you have to ensure with your system administrators that the
"standard" dependencies (python, numpy, etc.) are installed system-wide.

### Installation on OS X 10.11

We advise to install the following packages in the order as they appear
below. We strongly recommend to install all needed packages via
MacPorts. Please be aware that not all packages are available on
MacPorts, missing ones need to be installed manually. It may also be
mentioned that one should only use one package manager, as multiple ones
may cause trouble.

-   Install numpy:

        sudo port install py27-numpy

-   Install pyHealpix:

        git clone https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git
        cd pyHealpix
        autoreconf -i && ./configure --prefix=`python-config --prefix` && make -j4 && sudo make install
        cd ..

    (The third command installs the package system-wide. User-specific
    installation would be preferrable, but we haven't found a simple recipe yet
    how to determine the installation prefix ...)

-   Install NIFTy:

        git clone https://gitlab.mpcdf.mpg.de/ift/NIFTy.git
        cd NIFTy
        git checkout master
        python setup.py install --user
        cd ..

### Running the tests

In oder to run the tests one needs two additional packages:

    pip install nose
    pip install parameterized

Afterwards the tests (including a coverage report) are run using the following
command in the repository root:

    nosetests --exe --cover-html


### First Steps

For a quickstart, you can browse through the [informal
introduction](http://www.mpa-garching.mpg.de/ift/nifty/start.html) or
dive into NIFTY by running one of the demonstrations, e.g.:

    python demos/wiener_filter.py

Acknowledgement
---------------

Please, acknowledge the use of NIFTY in your publication(s) by using a
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