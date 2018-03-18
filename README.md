NIFTy - Numerical Information Field Theory
==========================================
[![build status](https://gitlab.mpcdf.mpg.de/ift/NIFTy/badges/NIFTy_4/build.svg)](https://gitlab.mpcdf.mpg.de/ift/NIFTy/commits/NIFTy_4)
[![coverage report](https://gitlab.mpcdf.mpg.de/ift/NIFTy/badges/NIFTy_4/coverage.svg)](https://gitlab.mpcdf.mpg.de/ift/NIFTy/commits/NIFTy_4)

**NIFTy** project homepage:
[https://ift.pages.mpcdf.de/NIFTy](https://ift.pages.mpcdf.de/NIFTy)

Summary
-------

### Description

**NIFTy**, "**N**umerical **I**nformation **F**ield **T**heor<strong>y</strong>", is
a versatile library designed to enable the development of signal
inference algorithms that operate regardless of the underlying spatial
grid and its resolution. Its object-oriented framework is written in
Python, although it accesses libraries written in C++ and C for
efficiency.

NIFTy offers a toolkit that abstracts discretized representations of
continuous spaces, fields in these spaces, and operators acting on
fields into classes. The correct normalization of operations on
fields is taken care of automatically without concerning the user. This
allows for an abstract formulation and programming of inference
algorithms, including those derived within information field theory.
Thus, NIFTy permits its user to rapidly prototype algorithms in 1D, and
then apply the developed code in higher-dimensional settings of real
world problems. The set of spaces on which NIFTy operates comprises
point sets, *n*-dimensional regular grids, spherical spaces, their
harmonic counterparts, and product spaces constructed as combinations of
those.


Installation
------------

### Requirements

- [Python](https://www.python.org/) (v2.7.x or 3.5.x)
- [NumPy](https://www.numpy.org/)
- [pyFFTW](https://pypi.python.org/pypi/pyFFTW)

Optional dependencies:
- [pyHealpix](https://gitlab.mpcdf.mpg.de/ift/pyHealpix) (for harmonic
    transforms involving domains on the sphere)
- [mpi4py](https://mpi4py.scipy.org) (for MPI-parallel execution)
- [matplotlib](https://matplotlib.org/) (for field plotting)
- [SciPy](https://www.scipy.org/) (for additional minimization algorithms)

### Sources

The current version of Nifty4 can be obtained by cloning the repository and
switching to the NIFTy_4 branch:

    git clone https://gitlab.mpcdf.mpg.de/ift/NIFTy.git

### Installation

In the following, we assume a Debian-based distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy4 and its mandatory dependencies can be installed via:

    sudo apt-get install git libfftw3-dev python python-pip python-dev
    pip install --user git+https://gitlab.mpcdf.mpg.de/ift/NIFTy.git@NIFTy_4

(Note: If you encounter problems related to `pyFFTW`, make sure that you are
using a pip-installed `pyFFTW` package. Unfortunately, some distributions are
shipping an incorrectly configured `pyFFTW` package, which does not cooperate
with the installed `FFTW3` libraries.)

Plotting support is added via:

    pip install --user matplotlib

Support for spherical harmonic transforms is added via:

    pip install --user git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git

MPI support is added via:

    sudo apt-get install openmpi-bin libopenmpi-dev
    pip install --user mpi4py

Scipy-based minimizers are enabled via:

    pip install --user scipy

### Installation for Python 3

If you want to run NIFTy with Python 3, you need to make the following changes
to the instructions above:

- in all `apt-get` commands, replace `python-*` by `python3-*`
- in all `pip` commands, replace `pip` by `pip3`

### Running the tests

In oder to run the tests one needs two additional packages:

    pip install --user nose parameterized coverage

Afterwards the tests (including a coverage report) can be run using the
following command in the repository root:

    nosetests -x --with-coverage --cover-html --cover-package=nifty4


### First Steps

For a quick start, you can browse through the [informal
introduction](https://ift.pages.mpcdf.de/NIFTy/code.html) or
dive into NIFTy by running one of the demonstrations, e.g.:

    python demos/wiener_filter_via_curvature.py


### Acknowledgement

Please acknowledge the use of NIFTy in your publication(s) by using a
phrase such as the following:

> *"Some of the results in this publication have been derived using the
> NIFTy package [(https://gitlab.mpcdf.mpg.de/ift/NIFTy)](https://gitlab.mpcdf.mpg.de/ift/NIFTy)"

and a citation to one of the [publications](https://ift.pages.mpcdf.de/NIFTy/citations.html)


### Release Notes

The NIFTy package is licensed under the terms of the
[GPLv3](https://www.gnu.org/licenses/gpl.html) and is distributed
*without any warranty*.

* * * * *

[1] Steininger et al., "NIFTy 3 - Numerical Information Field Theory - A Python framework for multicomponent signal inference on HPC clusters", 2017, submitted to PLOS One;
[arXiv:1708.01073](https://arxiv.org/abs/1708.01073)
