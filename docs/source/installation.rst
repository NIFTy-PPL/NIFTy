Installation
============


In the following, we assume a Debian-based Linux distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy5 and its mandatory dependencies can be installed via::

    sudo apt-get install git python3 python3-pip python3-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_5

Plotting support is added via::

    sudo apt-get install python3-matplotlib

NIFTy uses Numpy's FFT implementation by default. For large problems FFTW may be
used because of its higher performance. It can be installed via::

    sudo apt-get install libfftw3-dev
    pip3 install --user pyfftw

To enable FFTW usage in NIFTy, call::

    nifty5.fft.enable_fftw()

at the beginning of your code.

(Note: If you encounter problems related to `pyFFTW`, make sure that you are
using a pip-installed `pyFFTW` package. Unfortunately, some distributions are
shipping an incorrectly configured `pyFFTW` package, which does not cooperate
with the installed `FFTW3` libraries.)

Support for spherical harmonic transforms is added via::

    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git

Support for the radio interferometry gridder is added via:

    pip3 install git+https://gitlab.mpcdf.mpg.de/ift/nifty_gridder.git

MPI support is added via::

    sudo apt-get install python3-mpi4py

NIFTy documentation is provided by Sphinx. To build the documentation::

    sudo apt-get install python3-sphinx-rtd-theme dvipng
    cd <nifty_directory>
    sh docs/generate.sh

To view the documentation in firefox::

    firefox docs/build/index.html

(Note: Make sure that you reinstall nifty after each change since sphinx
imports nifty from the Python path.)

