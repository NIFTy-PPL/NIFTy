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

    *   `NumPy <http://www.numpy.org/>`_ and `SciPy <http://www.scipy.org/>`_
    *   `matplotlib <http://matplotlib.org/>`_
    *   `multiprocessing <http://docs.python.org/2/library/multiprocessing.html>`_
        (standard library)

*   `GFFT <https://github.com/mrbell/gfft>`_ (v0.1.0) - Generalized Fast
    Fourier Transformations for Python - **optional**

*   `HEALPy <https://github.com/healpy/healpy>`_ (v1.4 without openmp) - A
    Python wrapper for `HEALPix <http://sourceforge.net/projects/healpix/>`_ -
    **optional**

*   `libsharp-wrapper <https://github.com/mselig/libsharp-wrapper>`_ (v0.1.2
    without openmp) - A Python wrapper for the
    `libsharp <http://sourceforge.net/projects/libsharp/>`_ library -
    **optional**

Download
........

The latest release is tagged **v1.0.7** and is available as a source package
at `<https://github.com/information-field-theory/nifty/tags>`_. The current
version can be obtained by cloning the repository::

    git clone git://github.com/information-field-theory/nifty.git

Installation
............

*   NIFTY can be installed using `PyPI <https://pypi.python.org/pypi>`_ and
    **pip** by running the following command::

        pip install ift_nifty

    Alternatively, a private or user specific installation can be done by::

        pip install --user ift_nifty


*   NIFTY can be installed using **Distutils** by running the following
    command::

        cd nifty
        python setup.py install

    Alternatively, a private or user specific installation can be done by::

        python setup.py install --user
        python setup.py install --install-lib=/SOMEWHERE

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

