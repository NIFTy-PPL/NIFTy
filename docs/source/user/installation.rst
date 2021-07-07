Installation
============


In the following, we assume a Debian-based Linux distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy7 and its mandatory dependencies can be installed via::

    sudo apt-get install git python3 python3-pip python3-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_7

Plotting support is added via::

    sudo apt-get install python3-matplotlib

The DUCC0 package is installed via::

    pip3 install --user ducc0

If this library is present, NIFTy will detect it automatically and prefer
`ducc0.fft` over SciPy's FFT. The underlying code is actually the same, but
DUCC's FFT is compiled with optimizations for the host CPU and can provide
significantly faster transforms.

MPI support is added via::

    sudo apt-get install python3-mpi4py

NIFTy documentation is provided by Sphinx. To build the documentation::

    sudo apt-get install python3-sphinx dvipng
    pip install --user pydata-sphinx-theme
    cd <nifty_directory>
    sh docs/generate.sh

To view the documentation in firefox::

    firefox docs/build/index.html

(Note: Make sure that you reinstall nifty after each change since sphinx
imports nifty from the Python path.)
