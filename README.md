NIFTy - Numerical Information Field Theory
==========================================
[![build status](https://gitlab.mpcdf.mpg.de/ift/nifty-dev/badges/NIFTy_5/build.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty-dev/commits/NIFTy_5)
[![coverage report](https://gitlab.mpcdf.mpg.de/ift/nifty-dev/badges/NIFTy_5/coverage.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty-dev/commits/NIFTy_5)

**NIFTy** project homepage:
[http://ift.pages.mpcdf.de/NIFTy](http://ift.pages.mpcdf.de/NIFTy)

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
- [SciPy](https://www.scipy.org/)
- [pyFFTW](https://pypi.python.org/pypi/pyFFTW)

Optional dependencies:
- [pyHealpix](https://gitlab.mpcdf.mpg.de/ift/pyHealpix) (for harmonic
    transforms involving domains on the sphere)
- [mpi4py](https://mpi4py.scipy.org) (for MPI-parallel execution)
- [matplotlib](https://matplotlib.org/) (for field plotting)

### Sources

The current version of Nifty5 can be obtained by cloning the repository and
switching to the NIFTy_5 branch:

    git clone https://gitlab.mpcdf.mpg.de/ift/NIFTy.git

### Installation

In the following, we assume a Debian-based distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy5 and its mandatory dependencies can be installed via:

    sudo apt-get install git libfftw3-dev python python-pip python-dev
    pip install --user git+https://gitlab.mpcdf.mpg.de/ift/NIFTy.git@NIFTy_5

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

    nosetests -x --with-coverage --cover-html --cover-package=nifty5


### First Steps

For a quick start, you can browse through the [informal
introduction](http://ift.pages.mpcdf.de/NIFTy/code.html) or
dive into NIFTy by running one of the demonstrations, e.g.:

    python demos/wiener_filter_via_curvature.py


### Acknowledgement

Please acknowledge the use of NIFTy in your publication(s) by using a
phrase such as the following:

> "Some of the results in this publication have been derived using the
> NIFTy package [(https://gitlab.mpcdf.mpg.de/ift/NIFTy)](https://gitlab.mpcdf.mpg.de/ift/NIFTy)"

and a citation to one of the [publications](http://ift.pages.mpcdf.de/NIFTy/citations.html).


### Release Notes

The NIFTy package is licensed under the terms of the
[GPLv3](https://www.gnu.org/licenses/gpl.html) and is distributed
*without any warranty*.
