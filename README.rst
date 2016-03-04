NIFTY - Numerical Information Field Theory
==========================================

**NIFTY** project homepage: `<http://www.mpa-garching.mpg.de/ift/nifty/>`_

Summary
-------

Description
...........

**NIFTY**, "\ **N**\umerical **I**\nformation **F**\ield **T**\heor\ **y**\ ",
is a versatile library designed to enable the development of signal inference
algorithms that operate regardless of the underlying spatial grid and its
resolution. Its object-oriented framework is written in Python, although it
accesses libraries written in Cython, C++, and C for efficiency.

NIFTY offers a toolkit that abstracts discretized representations of continuous
spaces, fields in these spaces, and operators acting on fields into classes.
Thereby, the correct normalization of operations on fields is taken care of
automatically without concerning the user. This allows for an abstract
formulation and programming of inference algorithms, including those derived
within information field theory. Thus, NIFTY permits its user to rapidly
prototype algorithms in 1D, and then apply the developed code in
higher-dimensional settings of real world problems. The set of spaces on which
NIFTY operates comprises point sets, *n*-dimensional regular grids, spherical
spaces, their harmonic counterparts, and product spaces constructed as
combinations of those.

Class & Feature Overview
........................

The NIFTY library features three main classes: **spaces** that represent
certain grids, **fields** that are defined on spaces, and **operators** that
apply to fields.

*   `Spaces <http://www.mpa-garching.mpg.de/ift/nifty/space.html>`_

    *   ``point_space`` - unstructured list of points
    *   ``rg_space`` - *n*-dimensional regular Euclidean grid
    *   ``lm_space`` - spherical harmonics
    *   ``gl_space`` - Gauss-Legendre grid on the 2-sphere
    *   ``hp_space`` - `HEALPix <http://sourceforge.net/projects/healpix/>`_
        grid on the 2-sphere
    *   ``nested_space`` - arbitrary product of grids

*   `Fields <http://www.mpa-garching.mpg.de/ift/nifty/field.html>`_

    *   ``field`` - generic class for (discretized) fields

::

    field.cast_domain   field.hat           field.power        field.smooth
    field.conjugate     field.inverse_hat   field.pseudo_dot   field.tensor_dot
    field.dim           field.norm          field.set_target   field.transform
    field.dot           field.plot          field.set_val      field.weight

*   `Operators <http://www.mpa-garching.mpg.de/ift/nifty/operator.html>`_

    *   ``diagonal_operator`` - purely diagonal matrices in a specified basis
    *   ``projection_operator`` - projections onto subsets of a specified basis
    *   ``vecvec_operator`` - matrices derived from the outer product of a
        vector
    *   ``response_operator`` - exemplary responses that include a convolution,
        masking and projection
    *   ``propagator_operator`` - information propagator in Wiener filter theory
    *   ``explicit_operator`` - linear operators with an explicit matrix
        representation
    *   (and more)

* (and more)

*Parts of this summary are taken from* [1]_ *without marking them explicitly as
quotations.*

Installation
------------

Requirements
............

*   `Python <http://www.python.org/>`_ (v2.7.x)

    *   `NumPy <http://www.numpy.org/>`_ 
    *   `SciPy <http://www.scipy.org/>`_
    *   `Cython <http://cython.org/>`_
    *   `matplotlib <http://matplotlib.org/>`_
    
*   `GFFT <https://github.com/mrbell/gfft>`_ (v0.1.0) - Generalized Fast
    Fourier Transformations for Python - **optional**

*   `HEALPy <https://github.com/healpy/healpy>`_ (v1.8.1 without openmp) - A
    Python wrapper for `HEALPix <http://sourceforge.net/projects/healpix/>`_ -
    **optional, only needed for spherical spaces**
 
*   `libsharp-wrapper <https://github.com/mselig/libsharp-wrapper>`_ (v0.1.2
    without openmp) - A Python wrapper for the
    `libsharp <http://sourceforge.net/projects/libsharp/>`_ library -
    **optional, only needed for spherical spaces**

Download
........

The latest release is tagged **v1.0.7** and is available as a source package
at `<https://gitlab.mpcdf.mpg.de/ift/NIFTy/tags>`_. The current
version can be obtained by cloning the repository::

    git clone https://gitlab.mpcdf.mpg.de/ift/NIFTy.git

Installation on Ubuntu
......................

This is for you if you want to install NIFTy on your personal computer running 
with an Ubuntu-like linux system were you have root priviledges. Starting with
a fresh Ubuntu installation move to a folder like ``~/Downloads``:

*    Install basic packages like python, python-dev, gsl and others::
 
        sudo apt-get install curl git autoconf 
        sudo apt-get install python-dev python-pip gsl-bin libgsl0-dev libfreetype6-dev libpng-dev  libatlas-base-dev gfortran 

*    Install matplotlib::

        sudo apt-get install python-matplotlib

*    Using pip install numpy, scipy, etc...::

        sudo pip install numpy scipy cython pyfits healpy

*    Now install the 'non-standard' dependencies. First of all gfft::

        curl -LOk https://github.com/mrbell/gfft/tarball/master 
        tar -xzf master 
        cd mrbell-gfft* 
        sudo python setup.py install 
        cd ..

*    Libsharp::

        git clone http://git.code.sf.net/p/libsharp/code libsharp-code 
        cd libsharp-code 
        sudo autoconf 
        ./configure --enable-pic --disable-openmp 
        sudo make 
        cd ..

*    Libsharpwrapper::

        git clone http://github.com/mselig/libsharp-wrapper.git libsharp-wrapper 
        cd libsharp-wrapper 
        sudo python setup.py build_ext 
        sudo python setup.py install 
        cd ..

*    Finally, NIFTy::

        git clone https://gitlab.mpcdf.mpg.de/ift/NIFTy.git
        cd nifty
        sudo python setup.py install 
        cd .. 

Installation on a linux cluster
...............................

This is for you if you want to install NIFTy on a HPC machine or cluster that is hosted by your university or institute. Most of the dependencies will most likely already be there, but you won't have superuser priviledges. In this case, instead::

    sudo python setup.py install 

use::

    python setup.py install --user

or::

    python setup.py install --install-lib=/SOMEWHERE


in the instruction above. This will install the python packages into your local user directory. 

Installation on OS X 10.11
..........................

We advice to install the following packages in the order as they appear below. We strongly recommend to install all needed packages via MacPorts. Please be aware that not all packages are available on MacPorts, missing ones need to be installed manually. It may also be mentioned that one should only use one package manager, as multiple ones may cause trouble.

*    Install basic packages python, scipy, matplotlib and cython::

       sudo port install py27-numpy
       sudo port install py27-scipy
       sudo port install py27-matplotlib
       sudo port install py27-cython

*    Install gfft. **Depending where you installed GSL you may need to change the path in setup.py!**::

        sudo port install gsl
        git clone https://github.com/mrbell/gfft.git}{https://github.com/mrbell/gfft.git
        sudo python setup.py install

*    Install healpy::

        sudo port install py27-pyfits
        git clone https://github.com/healpy/healpy.git
        cd healpy 
        sudo python setup.py install
        cd ..

*    Install libsharp and therefore autoconf. 
     Further install instructions for libsharp may be found here: 
     https://sourceforge.net/p/libsharp/code/ci/master/tree/::
                
        sudo port install autoconf
        
        git clone http://git.code.sf.net/p/libsharp/code libsharp-code 
        cd libsharp-code 
        sudo autoconf 
        ./configure --enable-pic --disable-openmp 
        sudo make 
        cd ..
        
*    Install libsharp-wrapper.
     **Adopt the path of the libsharp installation in setup.py** ::

        sudo port install gcc
        sudo port select gcc  mp-gcc5
        git clone https://github.com/mselig/libsharp-wrapper.git
        cd libsharp-wrapper
        sudo python setup.py install
        cd ..

*    Install NIFTy::

        git clone https://gitlab.mpcdf.mpg.de/ift/NIFTy.git
        cd nifty
        sudo python setup.py install 
        cd .. 

Installation using pypi
.......................

NIFTY can be installed using `PyPI <https://pypi.python.org/pypi>`_ and **pip** 
by running the following command::

    pip install ift_nifty

Alternatively, a private or user specific installation can be done by::

    pip install --user ift_nifty


First Steps
...........

For a quickstart, you can browse through the
`informal introduction <http://www.mpa-garching.mpg.de/ift/nifty/start.html>`_
or dive into NIFTY by running one of the demonstrations, e.g.::

        >>> run -m nifty.demos.demo_wf1

Acknowledgement
---------------

Please, acknowledge the use of NIFTY in your publication(s) by using a phrase
such as the following:

    *"Some of the results in this publication have been derived using the NIFTY
    package [Selig et al., 2013]."*

References
..........

.. [1] Selig et al., "NIFTY - Numerical Information Field Theory - a
    versatile Python library for signal inference",
    `A&A, vol. 554, id. A26 <http://dx.doi.org/10.1051/0004-6361/201321236>`_,
    2013; `arXiv:1301.4499 <http://www.arxiv.org/abs/1301.4499>`_

Release Notes
-------------

The NIFTY package is licensed under the
`GPLv3 <http://www.gnu.org/licenses/gpl.html>`_ and is distributed *without any
warranty*.

----

**NIFTY** project homepage: `<http://www.mpa-garching.mpg.de/ift/nifty/>`_

