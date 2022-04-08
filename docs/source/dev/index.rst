Development
===========

Coding conventions
------------------

We do not use pure Python `assert` statements in production code. They are not
guaranteed to by executed by Python and can be turned off by the user
(`python -O` in cPython). As an alternative use `ift.myassert`.



Installation for developers
---------------------------


In the following, we assume a Debian-based Linux distribution. For other
distributions, the "apt" lines will need slight changes.

NIFTy8 and its mandatory dependencies can be installed via::

    sudo apt-get install git python3 python3-pip python3-dev
    git clone -b NIFTy_8 https://gitlab.mpcdf.mpg.de/ift/nifty.git
    cd nifty
    pip3 install --user .

For a installation in editable mode add a `-e` to the last command.
Information for the installation of optional dependecies you can find :doc:`here.<../user/installation>`

Build the documentation
-----------------------

NIFTy documentation is provided by `Sphinx <https://www.sphinx-doc.org/en/stable/index.html>`_.

To build the documentation::

    sudo apt-get install dvipng jupyter-nbconvert texlive-latex-base texlive-latex-extra
    pip3 install sphinx pydata-sphinx-theme
    cd <nifty_directory>
    sh docs/generate.sh

To view the documentation in firefox::

    firefox docs/build/index.html

(Note: Make sure that you reinstall nifty after each change since sphinx
imports nifty from the Python path.)
