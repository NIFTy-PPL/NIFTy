NIFTy - Numerical Information Field Theory
==========================================
[![build status](https://gitlab.mpcdf.mpg.de/ift/NIFTy/badges/NIFTy_5/build.svg)](https://gitlab.mpcdf.mpg.de/ift/NIFTy/commits/NIFTy_5)
[![coverage report](https://gitlab.mpcdf.mpg.de/ift/NIFTy/badges/NIFTy_5/coverage.svg)](https://gitlab.mpcdf.mpg.de/ift/NIFTy/commits/NIFTy_5)

**NIFTy** project homepage:
[http://ift.pages.mpcdf.de/nifty](http://ift.pages.mpcdf.de/nifty)

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

### Requirements

- [Python 3](https://www.python.org/) (3.5.x or later)
- [SciPy](https://www.scipy.org/)
- [pypocketfft](https://gitlab.mpcdf.mpg.de/mtr/pypocketfft)

Optional dependencies:
- [pyFFTW](https://pypi.python.org/pypi/pyFFTW) for faster Fourier transforms
- [pyHealpix](https://gitlab.mpcdf.mpg.de/ift/pyHealpix) (for harmonic
    transforms involving domains on the sphere)
- [nifty_gridder](https://gitlab.mpcdf.mpg.de/ift/nifty_gridder) (for radio
    interferometry responses)
- [mpi4py](https://mpi4py.scipy.org) (for MPI-parallel execution)
- [matplotlib](https://matplotlib.org/) (for field plotting)

### Sources

The current version of Nifty5 can be obtained by cloning the repository and
switching to the NIFTy_5 branch:

    git clone https://gitlab.mpcdf.mpg.de/ift/nifty.git

### Installation

In the following, we assume a Debian-based distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy5 and its mandatory dependencies can be installed via:

    sudo apt-get install git python3 python3-pip python3-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_5
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft

Plotting support is added via:

    sudo apt-get install python3-matplotlib

NIFTy uses pypocketfft by default. For large problems FFTW may be
used because of its higher performance. It can be installed via:

    sudo apt-get install libfftw3-dev
    pip3 install --user pyfftw

To enable FFTW usage in NIFTy, call

    nifty5.fft.enable_fftw()

at the beginning of your code.

(Note: If you encounter problems related to `pyFFTW`, make sure that you are
using a pip-installed `pyFFTW` package. Unfortunately, some distributions are
shipping an incorrectly configured `pyFFTW` package, which does not cooperate
with the installed `FFTW3` libraries.)

Support for spherical harmonic transforms is added via:

    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git

Support for the radio interferometry gridder is added via:

    pip3 install git+https://gitlab.mpcdf.mpg.de/ift/nifty_gridder.git

MPI support is added via:

    sudo apt-get install python3-mpi4py

### Running the tests

To run the tests, additional packages are required:

    sudo apt-get install python3-pytest-cov

Afterwards the tests (including a coverage report) can be run using the
following command in the repository root:

    pytest-3 --cov=nifty5 test


### First Steps

For a quick start, you can browse through the [informal
introduction](http://ift.pages.mpcdf.de/nifty/code.html) or
dive into NIFTy by running one of the demonstrations, e.g.:

    python3 demos/getting_started_1.py


### Acknowledgements

Please acknowledge the use of NIFTy in your publication(s) by using a
phrase such as the following:

> "Some of the results in this publication have been derived using the
> NIFTy package [(https://gitlab.mpcdf.mpg.de/ift/NIFTy)](https://gitlab.mpcdf.mpg.de/ift/NIFTy)"

and a citation to one of the [publications](http://ift.pages.mpcdf.de/nifty/citations.html).


### Licensing terms

The NIFTy package is licensed under the terms of the
[GPLv3](https://www.gnu.org/licenses/gpl.html) and is distributed
*without any warranty*.
