Installation
============


In the following, we assume a Debian-based Linux distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy4 and its mandatory dependencies can be installed via::

    sudo apt-get install git libfftw3-dev python python-pip python-dev
    pip install --user git+https://gitlab.mpcdf.mpg.de/ift/NIFTy.git@NIFTy_4

(Note: If you encounter problems related to `pyFFTW`, make sure that you are
using a pip-installed `pyFFTW` package. Unfortunately, some distributions are
shipping an incorrectly configured `pyFFTW` package, which does not cooperate
with the installed `FFTW3` libraries.)

Plotting support is added via::

    pip install --user matplotlib

Support for spherical harmonic transforms is added via::

    pip install --user git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git

MPI support is added via::

    sudo apt-get install openmpi-bin libopenmpi-dev
    pip install --user mpi4py

Scipy-based minimizers are enabled via::

    pip install --user scipy

Installation for Python 3
-------------------------

If you want to run NIFTy with Python 3, you need to make the following changes
to the instructions above:

- in all `apt-get` commands, replace `python-*` by `python3-*`
- in all `pip` commands, replace `pip` by `pip3`

