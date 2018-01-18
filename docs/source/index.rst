NIFTy -- Numerical Information Field Theory
===========================================

**NIFTy** [1]_, "\ **N**\umerical **I**\nformation **F**\ield **T**\heor\ **y**\ ", is a versatile library designed to enable the development of signal inference algorithms that operate regardless of the underlying spatial grid and its resolution. Its object-oriented framework is written in Python, although it accesses libraries written in Cython, C++, and C for efficiency.

NIFTy offers a toolkit that abstracts discretized representations of continuous spaces, fields in these spaces, and operators acting on fields into classes. Thereby, the correct normalization of operations on fields is taken care of automatically without concerning the user. This allows for an abstract formulation and programming of inference algorithms, including those derived within information field theory. Thus, NIFTy permits its user to rapidly prototype algorithms in 1D and then apply the developed code in higher-dimensional settings of real world problems. The set of spaces on which NIFTy operates comprises point sets, *n*-dimensional regular grids, spherical spaces, their harmonic counterparts, and product spaces constructed as combinations of those.

References
----------

.. [1] Selig et al., "NIFTy -- Numerical Information Field Theory -- a versatile Python library for signal inference", `A&A, vol. 554, id. A26 <http://dx.doi.org/10.1051/0004-6361/201321236>`_, 2013; `arXiv:1301.4499 <http://www.arxiv.org/abs/1301.4499>`_

Documentation
-------------

Welcome to NIFTy's documentation!


Indices and tables
..................

* :ref:`genindex`
* :any:`Module Index <mod/modules>`
* :ref:`search`
