NIFTy's domain classes
======================

.. currentmodule:: nifty4


Abstract base class
-------------------

One of the fundamental building blocks of the NIFTy4 framework is the *domain*.
Its required capabilities are expressed by the abstract :class:`Domain` class.
A domain must be able to answer the following queries:

- its total number of data entries (pixels), which is accessible via the
  :attr:`~Domain.size` property
- the shape of the array that is supposed to hold these data entries
  (obtainable by means of the :attr:`~Domain.shape` property)
- equality comparison to another :class:`Domain` instance


Unstructured domains
--------------------

Domains can be either *structured* (i.e. there is geometrical information
associated with them, like position in space and volume factors),
or *unstructured* (meaning that the data points have no associated manifold).

Unstructured domains can be described by instances of NIFTy's
:class:`UnstructuredDomain` class.


Structured domains
------------------

In contrast to unstructured domains, these domains have an assigned geometry.
NIFTy requires them to provide the volume elements of their grid cells.
The additional methods are specified in the abstract class
:class:`StructuredDomain`:

- The attributes :attr:`~StructuredDomain.scalar_dvol`,
  :attr:`~StructuredDomain.dvol`, and  :attr:`~StructuredDomain.total_volume`
  provide information about the domain's pixel volume(s) and its total volume.
- The property :attr:`~StructuredDomain.harmonic` specifies whether a domain
  is harmonic (i.e. describes a frequency space) or not
- Iff the domain is harmonic, the methods
  :meth:`~StructuredDomain.get_k_length_array`,
  :meth:`~StructuredDomain.get_unique_k_lengths`, and
  :meth:`~StructuredDomain.get_fft_smoothing_kernel_function` provide absolute
  distances of the individual grid cells from the origin and assist with
  Gaussian convolution.

NIFTy comes with several concrete subclasses of :class:`StructuredDomain`:

- :class:`RGSpace` represents a regular Cartesian grid with an arbitrary
  number of dimensions, which is supposed to be periodic in each dimension.
- :class:`HPSpace` and :class:`GLSpace` describe pixelisations of the
  2-sphere; their counterpart in harmonic space is :class:`LMSpace`, which
  contains spherical harmonic coefficients.
- :class:`PowerSpace` is used to describe one-dimensional power spectra.

Among these, :class:`RGSpace` can be harmonic or not (depending on constructor arguments), :class:`GLSpace`, :class:`HPSpace`, and :class:`PowerSpace` are
pure position domains (i.e. nonharmonic), and :class:`LMSpace` is always
harmonic.
