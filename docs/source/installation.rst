Installation
============


In the following, we assume a Debian-based Linux distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy5 and its mandatory dependencies can be installed via::

    sudo apt-get install git python3 python3-pip python3-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_5
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft

Plotting support is added via::

    sudo apt-get install python3-matplotlib

Support for spherical harmonic transforms is added via::

    pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/pyHealpix.git

Support for the radio interferometry gridder is added via::

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

