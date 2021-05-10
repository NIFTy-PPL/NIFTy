.. role:: raw-html-m2r(raw)
   :format: html

JIFTy
=====

**JAX** + **NIFTy**

.. image:: https://gitlab.mpcdf.mpg.de/ift/jax_nifty/badges/main/pipeline.svg
   :target: https://gitlab.mpcdf.mpg.de/ift/jax_nifty/-/commits/main
   :alt: pipeline status

**JIFTy** project homepage:
`http://ift.pages.mpcdf.de/jax_nifty <http://ift.pages.mpcdf.de/jax_nifty>`_

Summary
-------

Description
^^^^^^^^^^^

JIFTy combines the power of auto-differentiation and just-in-time compilation
of JAX with the inference machinery of NIFTy ("\ **N**\ umerical **I**\
nformation **F**\ ield **T**\ heor\ **y**\ "). Most importantly JIFTy allows to
define inference problems with the convenience of Numpy all while providing
methods to define and easily incorporate complex correlation structures into
the model.  JIFTy is a purely Bayesian inference framework. The workhorse for
the inference of parameters from data is MGVI (\ **M**\ etric **G**\ aussian
**V**\ ariational **I**\ nference).

Installation
------------

Requirements
^^^^^^^^^^^^

* `Python 3 <https://www.python.org/>`_ (3.8 or later)
* `JAX <https://jax.readthedocs.io/>`_
* `Numpy <https://numpy.org/>`_
* `SciPy <https://www.scipy.org/>`_

Optional dependencies:

* `matplotlib <https://matplotlib.org/>`_ (for plotting)

Installation
^^^^^^^^^^^^

In the following, we assume a Debian-based distribution. For other
distributions, the ``apt`` lines will need slight changes.

JIFTy and its mandatory dependencies can be installed via:

.. code-block::

   sudo apt-get install git python3 python3-pip python3-dev
   pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/jax_nifty.git

Plotting support is added via:

.. code-block::

   sudo apt-get install python3-matplotlib

Running the tests
^^^^^^^^^^^^^^^^^

To run the tests, additional packages are required:

.. code-block::

   sudo apt-get install python3-pytest-cov

Afterwards the tests (including a coverage report) can be run using the
following command in the repository root:

.. code-block::

   pytest-3 --cov=jifty1 test

First Steps
^^^^^^^^^^^

For a quick start, browse through the scripts in `demos <demos/>`_ and start
working through the scripts:

.. code-block::

   python3 demos/correlated_field_w_unknown_spectrum.py

Building the documentation from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation from source, install `sphinx
<https://www.sphinx-doc.org/en/stable/index.html>`_\ and the `PyData Sphinx
Theme <https://github.com/pydata/pydata-sphinx-theme>`_ on your system and

.. code-block::

   sh doc/generate.sh

Acknowledgements
^^^^^^^^^^^^^^^^

Please acknowledge the use of JIFTy in your publication(s).

Licensing terms
^^^^^^^^^^^^^^^

The NIFTy package is licensed under the terms of the `GPLv3
<https://www.gnu.org/licenses/gpl.html>`_ and is distributed *without any
warranty*.
