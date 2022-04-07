Installation for users
======================


If you only want to to use NIFTy in your projects, but not change its source
code, the easiest way to install the package is the command::

    pip install --user nifty7

Depending on your OS, you may have to use `pip3` instead of `pip`.
This approach should work on Linux, MacOS and Windows.

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

For installing jax refer to `google/jax:README#Installation <https://github.com/google/jax#installation>`_.


Installation for developers
===========================

Information for the installation for developers you can find :doc:`here <../dev/index>`
