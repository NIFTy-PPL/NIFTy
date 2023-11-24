Development
===========

Coding conventions
------------------

Pure Python `assert` statements should only be used for internal consistency
checks and should not be used on user dependent input in production code. They 
are not guaranteed to be executed by Python and can be turned off by the user
(`python -O` in cPython). As an alternative use `ift.myassert`.


Installation for developers
---------------------------

In the following, we assume a Debian-based Linux distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy8 and its mandatory dependencies can be installed via::

    sudo apt-get install git python3 python3-pip python3-dev
    git clone -b NIFTy_8 https://gitlab.mpcdf.mpg.de/ift/nifty.git
    cd nifty
    pip3 install --user --editable .

For NITy8.re you must additionally `install JAX <https://github.com/google/jax#installation>`_.
For further details on optional dependencies see :doc:`here<../user/installation>`.


Build the documentation
-----------------------

NIFTy documentation is provided by `Sphinx <https://www.sphinx-doc.org/en/stable/index.html>`_.

To build the documentation::

    sudo apt-get install dvipng jupyter-nbconvert texlive-latex-base texlive-latex-extra
    pip3 install --user sphinx pydata-sphinx-theme
    cd <nifty_directory>
    sh docs/generate.sh

To view the documentation in firefox::

    firefox docs/build/index.html

(Note: Make sure that you reinstall nifty after each change since sphinx
imports nifty from the Python path.)
