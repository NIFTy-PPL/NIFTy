NIFTy -- Numerical Information Field Theory
===========================================

**NIFTy** [1]_, "\ **N**\umerical **I**\nformation **F**\ield **T**\heor\ **y**\ ", is a versatile library designed to enable the development of signal inference algorithms that are independent of the underlying spatial grid and its resolution.
Its object-oriented framework is written in Python, although it accesses libraries written in C++ and C for efficiency.

NIFTy offers a toolkit that abstracts discretized representations of continuous spaces, fields in these spaces, and operators acting on fields into classes.
Thereby, the correct normalization of operations on fields is taken care of automatically without concerning the user.
This allows for an abstract formulation and programming of inference algorithms, including those derived within information field theory.
Thus, NIFTy permits its user to rapidly prototype algorithms in 1D and then apply the developed code in higher-dimensional settings to real world problems.
The set of spaces on which NIFTy operates comprises point sets, *n*-dimensional regular grids, spherical spaces, their harmonic counterparts, and product spaces constructed as combinations of those.

References
----------

.. [1] Steininger et al., "NIFTy 3 - Numerical Information Field Theory - A Python framework for multicomponent signal inference on HPC clusters", 2017, submitted to PLOS One; `[arXiv:1708.01073] <https://arxiv.org/abs/1708.01073>`_

Contents
........

.. toctree::
   :maxdepth: 2

   ift
   Gallery <http://wwwmpa.mpa-garching.mpg.de/ift/nifty/gallery/>
   installation
   code
   citations

Indices and tables
..................

* :any:`Module Index <mod/modules>`
* :ref:`search`
