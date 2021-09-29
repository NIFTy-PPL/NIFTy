NIFTy - Numerical Information Field Theory
==========================================
[![pipeline status](https://gitlab.mpcdf.mpg.de/ift/nifty/badges/NIFTy_7/pipeline.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty/-/commits/NIFTy_7)
[![coverage report](https://gitlab.mpcdf.mpg.de/ift/nifty/badges/NIFTy_7/coverage.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty/-/commits/NIFTy_7)

**NIFTy** project homepage:
[https://ift.pages.mpcdf.de/nifty](https://ift.pages.mpcdf.de/nifty)

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

- [Python 3](https://www.python.org/) (3.6.x or later)
- [SciPy](https://www.scipy.org/)

Optional dependencies:
- [ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc) for faster FFTs, spherical
    harmonic transforms, and radio interferometry gridding support
- [mpi4py](https://mpi4py.scipy.org) (for MPI-parallel execution)
- [matplotlib](https://matplotlib.org/) (for field plotting)

### Sources

The current version of NIFTy7 can be obtained by cloning the repository and
switching to the NIFTy_7 branch:

    git clone https://gitlab.mpcdf.mpg.de/ift/nifty.git

### Installation for users

If you only want to to use NIFTy in your projects, but not change its source
code, the easiest way to install the package is the command:

    pip install --user nifty7

Depending on your OS, you may have to use `pip3` instead of `pip`.
This approach should work on Linux, MacOS and Windows.


### Installation for developers

In the following, we assume a Debian-based distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy7 and its mandatory dependencies can be installed via:

    sudo apt-get install git python3 python3-pip python3-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_7

Plotting support is added via:

    sudo apt-get install python3-matplotlib

The DUCC0 package is installed via:

    pip3 install ducc0

If this library is present, NIFTy will detect it automatically and prefer
`ducc0.fft` over SciPy's FFT. The underlying code is actually the same, but
DUCC's FFT is compiled with optimizations for the host CPU and can provide
significantly faster transforms.

MPI support is added via:

    sudo apt-get install python3-mpi4py

### Run the tests

To run the tests, additional packages are required:

    sudo apt-get install python3-pytest-cov

Afterwards the tests (including a coverage report) can be run using the
following command in the repository root:

    pytest-3 --cov=nifty7 test

### First Steps

For a quick start, you can browse through the [informal
introduction](https://ift.pages.mpcdf.de/nifty/code.html) or
dive into NIFTy by running one of the demonstrations, e.g.:

    python3 demos/getting_started_1.py

### Acknowledgements

Please consider acknowledging NIFTy in your publication(s) by using a phrase
such as the following:

> "Some of the results in this publication have been derived using the
> NIFTy package [(https://gitlab.mpcdf.mpg.de/ift/NIFTy)](https://gitlab.mpcdf.mpg.de/ift/NIFTy)"

and a citation to one of the [publications](https://ift.pages.mpcdf.de/nifty/citations.html).


### Licensing terms

The NIFTy package is licensed under the terms of the
[GPLv3](https://www.gnu.org/licenses/gpl.html) and is distributed
*without any warranty*.


Contributors
------------

### NIFTy7

- Andrija Kostic
- Gordian Edenhofer
- Jakob Knollmüller
- Jakob Roth
- Lukas Platz
- Matteo Guardiani
- Martin Reinecke
- [Philipp Arras](https://philipp-arras.de)
- [Philipp Frank](http://www.ph-frank.de)
- [Reimar Heinrich Leike](https://wwwmpa.mpa-garching.mpg.de/~reimar/)
- Simon Ding
- Vincent Eberle

### NIFTy6

- Andrija Kostic
- Gordian Edenhofer
- Jakob Knollmüller
- Lukas Platz
- Martin Reinecke
- [Philipp Arras](https://philipp-arras.de)
- [Philipp Frank](http://www.ph-frank.de)
- Philipp Haim
- [Reimar Heinrich Leike](https://wwwmpa.mpa-garching.mpg.de/~reimar/)
- Rouven Lemmerz
- [Torsten Enßlin](https://wwwmpa.mpa-garching.mpg.de/~ensslin/)
- Vincent Eberle


### NIFTy5

- Christoph Lienhard
- Gordian Edenhofer
- Jakob Knollmüller
- Julia Stadler
- Julian Rüstig
- Lukas Platz
- Martin Reinecke
- Max-Niklas Newrzella
- Natalia
- [Philipp Arras](https://philipp-arras.de)
- [Philipp Frank](http://www.ph-frank.de)
- Philipp Haim
- Reimar Heinrich Leike
- Sebastian Hutschenreuter
- Silvan Streit
- [Torsten Enßlin](https://wwwmpa.mpa-garching.mpg.de/~ensslin/)


### NIFTy4

- Christoph Lienhard
- Jakob Knollmüller
- Lukas Platz
- Martin Reinecke
- Mihai Baltac
- [Philipp Arras](https://philipp-arras.de)
- [Philipp Frank](http://www.ph-frank.de)
- Reimar Heinrich Leike
- Silvan Streit
- [Torsten Enßlin](https://wwwmpa.mpa-garching.mpg.de/~ensslin/)


### NIFTy3

- Daniel Pumpe
- Jait Dixit
- Jakob Knollmüller
- Martin Reinecke
- Mihai Baltac
- Natalia
- [Philipp Arras](https://philipp-arras.de)
- [Philipp Frank](http://www.ph-frank.de)
- Reimar Heinrich Leike
- Matevz Sraml
- Theo Steininger
- csongor

### NIFTy2

- Jait Dixit
- Theo Steininger
- csongor


### NIFTy1

- Johannes Buchner
- Marco Selig
- Theo Steininger
