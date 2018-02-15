NIFTy's domain classes
======================

.. currentmodule:: nifty4


**Abstract base class**

:class:`Domain` is the abstract ancestor for all of NIFTy's domains.

.. toctree::
    :maxdepth: 1

    Domain                          <../mod/nifty4.domains.domain>


**Unstructured domains**

Domains can be either *structured* (i.e. there is geometrical information
associated with them, like position in space and volume factors),
or *unstructured* (meaning that the data points have no associated manifold).

Unstructured domains can be described by instances of NIFTy's
:class:`UnstructuredDomain` class.

.. toctree::
    :maxdepth: 1

    UnstructuredDomain              <../mod/nifty4.domains.unstructured_domain>


**Structured domains**

In contrast to unstructured domains, these domains have an assigned geometry.
NIFTy requires these domains to provide the volume elements of their grid cells.
The additional methods are described in the abstract class
:class:`StructuredDomain`.

.. toctree::
    :maxdepth: 1

    StructuredDomain                <../mod/nifty4.domains.structured_domain>

NIFTy comes with several concrete subclasses of :class:`StructuredDomain`.

:class:`RGSpace` represents a regular Cartesian grid with an arbitrary
number of dimensions, which is supposed to be periodic in each dimension.
This domain can be constructed to represent either position or harmonic space.

:class:`HPSpace` and :class:`GLSpace` describe pixelisations of the
2-sphere; their counterpart in harmonic space is :class:`LMSpace`, which
contains spherical harmonic coefficients.

:class:`PowerSpace` is used to describe one-dimensional power spectra.

.. toctree::
    :maxdepth: 1

    RGSpace                         <../mod/nifty4.domains.rg_space>
    HPSpace                         <../mod/nifty4.domains.hp_space>
    GLSpace                         <../mod/nifty4.domains.gl_space>
    LMSpace                         <../mod/nifty4.domains.lm_space>

    PowerSpace                      <../mod/nifty4.domains.power_space>
