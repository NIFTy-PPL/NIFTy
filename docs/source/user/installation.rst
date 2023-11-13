Installation
============

Requirements
------------

- `Python 3 <https://www.python.org/>`_ (3.8.x or later)
- `SciPy <https://www.scipy.org/>`_

Optional dependencies:

- `ducc0 <https://gitlab.mpcdf.mpg.de/mtr/ducc>`_ for faster FFTs, spherical harmonic transforms, and radio interferometry gridding support
- `mpi4py <https://github.com/mpi4py/mpi4py/>`_ for MPI-parallel execution
- `h5py <https://www.h5py.org/>`_ for writing results to HDF5 files
- `astropy <https://www.astropy.org/>`_ for writing FITS files
- `matplotlib <https://matplotlib.org/>`_  for field plotting
- `jax <https://github.com/google/jax>`_  for implementing operators with jax


Installation for users
----------------------


If you only want to to use NIFTy in your projects, but not change its source
code, the easiest way to install the package is the command::

    pip3 install --user nifty8

This approach should work on Linux, MacOS and Windows.

**NOTE**: nifty8 is not yet released. Consider the installation for developers.

Consider the installation of optional dependencies. (see below)


Installation for developers
---------------------------

Information for the installation for developers you can find :doc:`here.<../dev/index>`


Installation of optional dependencies
-------------------------------------

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

The h5py package for writing HDF5 files is installed via::

    pip3 install --user h5py

The astropy package for writing FITS files is installed via::

    pip3 install --user astropy

For installing jax refer to `google/jax:README#Installation <https://github.com/google/jax#installation>`_.
