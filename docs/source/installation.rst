Installation
============


In the following, we assume a Debian-based Linux distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy5 and its mandatory dependencies can be installed via::

    sudo apt-get install git python3 python3-pip python3-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/NIFTy.git@NIFTy_5

Plotting support is added via::

    pip3 install --user matplotlib

Since Jan. 2019 NIFTy uses Numpy's FFT implementation by default, in order to
minimize dependencies. However, for long-running production jobs we still
recommend using FFTW because of its higher performance. This is achieved via:

    sudo apt-get install libfftw3-dev
    pip3 install --user pyfftw

To actually enable FFTW in your NIFTy calculations, you need to call::

    nifty5.fft.enable_fftw()

at the beginning of your code.

(Note: If you encounter problems related to `pyFFTW`, make sure that you are
using a pip-installed `pyFFTW` package. Unfortunately, some distributions are
shipping an incorrectly configured `pyFFTW` package, which does not cooperate
with the installed `FFTW3` libraries.)

Support for spherical harmonic transforms is added via::

    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git

MPI support is added via::

    sudo apt-get install openmpi-bin libopenmpi-dev
    pip3 install --user mpi4py
