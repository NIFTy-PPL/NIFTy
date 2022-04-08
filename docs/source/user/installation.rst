Installation for users
----------------------


If you only want to to use NIFTy in your projects, but not change its source
code, the easiest way to install the package is the command::

    pip install --user nifty8

Depending on your OS, you may have to use `pip3` instead of `pip`.
This approach should work on Linux, MacOS and Windows.

**NOTE**: nifty8 is not yet released. Consider the installation for developers.


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

    pip install h5py

The astropy package for writing FITS files is installed via::

    pip install astropy

For installing jax refer to `google/jax:README#Installation <https://github.com/google/jax#installation>`_.
