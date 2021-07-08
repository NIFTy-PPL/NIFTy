What is NIFTy?
==============

**NIFTy** [1]_ [2]_ [3]_, "\ **N**\umerical **I**\nformation **F**\ield **T**\heor\ **y**\ ", is a versatile library designed to enable the development of signal inference algorithms that are independent of the underlying grids (spatial, spectral, temporal, …) and their resolutions.
Its object-oriented framework is written in Python, although it accesses libraries written in C++ and C for efficiency.

NIFTy offers a toolkit that abstracts discretized representations of continuous spaces, fields in these spaces, and operators acting on these fields into classes.
This allows for an abstract formulation and programming of inference algorithms, including those derived within information field theory.
NIFTy's interface is designed to resemble IFT formulae in the sense that the user implements algorithms in NIFTy independent of the topology of the underlying spaces and the discretization scheme.
Thus, the user can develop algorithms on subsets of problems and on spaces where the detailed performance of the algorithm can be properly evaluated and then easily generalize them to other, more complex spaces and the full problem, respectively.

The set of spaces on which NIFTy operates comprises point sets, *n*-dimensional regular grids, spherical spaces, their harmonic counterparts, and product spaces constructed as combinations of those.
NIFTy takes care of numerical subtleties like the normalization of operations on fields and the numerical representation of model components, allowing the user to focus on formulating the abstract inference procedures and process-specific model properties.

Examples of nifty applications can be found in the `nifty gallery (external link) <https://wwwmpa.mpa-garching.mpg.de/~ensslin/nifty-gallery/index.html>`_.


References
----------

.. [1] Selig et al., "NIFTY - Numerical Information Field Theory. A versatile PYTHON library for signal inference ", 2013, Astronmy and Astrophysics 554, 26; `[DOI] <https://ui.adsabs.harvard.edu/link_gateway/2013A&A...554A..26S/doi:10.1051/0004-6361/201321236>`_, `[arXiv:1301.4499] <https://arxiv.org/abs/1301.4499>`_

.. [2] Steininger et al., "NIFTy 3 - Numerical Information Field Theory - A Python framework for multicomponent signal inference on HPC clusters", 2017, accepted by Annalen der Physik; `[arXiv:1708.01073] <https://arxiv.org/abs/1708.01073>`_

.. [3] Arras et al., "NIFTy5: Numerical Information Field Theory v5", 2019, Astrophysics Source Code Library; `[ascl:1903.008] <http://ascl.net/1903.008>`_
