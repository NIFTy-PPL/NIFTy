Installation
============

Requirements
------------

- `Python 3 <https://www.python.org/>`_ (3.10.x or later)
- `SciPy <https://www.scipy.org/>`_

Optional dependencies:

- `jax <https://github.com/google/jax>`_  for implementing operators with JAX
- `ducc0 <https://gitlab.mpcdf.mpg.de/mtr/ducc>`_ for faster FFTs, spherical harmonic transforms, and radio interferometry gridding support
- `mpi4py <https://github.com/mpi4py/mpi4py/>`_ for MPI-parallel execution
- `h5py <https://www.h5py.org/>`_ for writing results to HDF5 files
- `astropy <https://www.astropy.org/>`_ for writing FITS files
- `matplotlib <https://matplotlib.org/>`_  for field plotting


Installation for users
----------------------

If you only want to use NIFTy in your projects, but not change its source
code, the easiest way to install the package is the command::

    pip3 install --user nifty8

This approach should work on Linux, MacOS, and Windows.
To install NIFTy.re, one additionally needs to
`install JAX <https://github.com/google/jax#installation>`_. A convenient alias
to install the CPU-only version of NIFTy.re (and JAX) is to install
:code:`nifty8[re]` from PyPi.

Installation for developers
---------------------------

Information for the installation for developers you can find :doc:`here.<../dev/index>`
