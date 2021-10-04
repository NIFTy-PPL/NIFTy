Installation for users
======================


If you only want to to use NIFTy in your projects, but not change its source
code, the easiest way to install the package is the command:

    pip install --user nifty7

Depending on your OS, you may have to use `pip3` instead of `pip`.
This approach should work on Linux, MacOS and Windows.


Installation for developers
===========================


In the following, we assume a Debian-based Linux distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy8 and its mandatory dependencies can be installed via::

    sudo apt-get install git python3 python3-pip python3-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_8

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

    sudo apt-get install dvipng texlive-latex-base texlive-latex-extra
    pip3 install sphinx pydata-sphinx-theme
    cd <nifty_directory>
    sh docs/generate.sh

To view the documentation in firefox::

    firefox docs/build/index.html

(Note: Make sure that you reinstall nifty after each change since sphinx
imports nifty from the Python path.)
