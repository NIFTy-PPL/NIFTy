.. currentmodule:: nifty4

=============
Code Overview
=============


Executive summary
=================

The fundamental building blocks required for IFT computations are best
recognized from a large distance, ignoring all technical details.

From such a perspective,

- IFT problems largely consist of *minimization* problems involving a large
  number of equations.
- The equations are built mostly from the application of *linear operators*,
  but there may also be nonlinear functions involved.
- The unknowns in the equations represent either continuous physical *fields*,
  or they are simply individual measured *data points*.
- Discretized *fields* have geometrical information (like locations and volume
  elements) associated with every entry; this information is called the field's
  *domain*.

In the following sections, the concepts briefly presented here will be
discussed in more detail; this is done in reversed order of their introduction,
to avoid forward references.


Domains
=======


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

- The properties :attr:`~StructuredDomain.scalar_dvol`,
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


Combinations of domains
=======================

The fundamental classes described above are often sufficient to specify the
domain of a field. In some cases, however, it will be necessary to have the
field live on a product of elementary domains instead of a single one.
Some examples are:

- sky emission depending on location and energy. This could be represented by
  a product of an :class:`HPSpace` (for location) with an :class:`RGSpace`
  (for energy).
- a polarised field, which could be modeled as a product of any structured
  domain (representing location) with a four-element
  :class:`UnstructuredDomain` holding Stokes I, Q, U and V components.

Consequently, NIFTy defines a class called :class:`DomainTuple` holding
a sequence of :class:`Domain` objects, which is used to specify full field
domains. In principle, a :class:`DomainTuple` can even be empty, which implies
that the field living on it is a scalar.

A :class:`DomainTuple` supports iteration and indexing, and also provides the
properties :attr:`~DomainTuple.shape`, :attr:`~DomainTuple.size` in analogy to
the elementary :class:`Domain`.


Fields
======

A :class:`Field` object consists of the following components:

- a domain in form of a :class:`DomainTuple` object
- a data type (e.g. numpy.float64)
- an array containing the actual values

Usually, the array is stored in the for of a ``numpy.ndarray``, but for very
resource-intensive tasks NIFTy also provides an alternative storage method to
be used with distributed memory processing.

Fields support a wide range of arithmetic operations, either involving two
fields with equal domains, or a field and a scalar.
Contractions (like summation, integration, minimum/maximum, computation of
statistical moments) can be carried out either over an entire field (producing
a scalar result) or over sub-domains (resulting in a field living on a smaller
domain). Scalar products of two fields can also be computed easily.

There is also a set of convenience functions to generate fields with constant
values or fields filled with random numbers according to a user-specified
distribution.

Fields are the only fundamental NIFTy objects which can change state after they
have been constructed: while their data type, domain, and array shape cannot
be modified, the actual data content of the array may be manipulated during the
lifetime of the object. This is a slight deviation from the philosophy that all
NIFTy objects should be immutable, but this choice offers considerable
performance benefits.


Linear Operators
================

A linear operator (represented by NIFTy4's abstract :class:`LinearOperator`
class) can be interpreted as an (implicitly defined) matrix.
It can be applied to :class:`Field` instances, resulting in other :class:`Field`
instances that potentially live on other domains.


Operator basics
---------------

There are four basic ways of applying an operator :math:`A` to a field :math:`f`:

- direct multiplication: :math:`A\cdot f`
- adjoint multiplication: :math:`A^\dagger \cdot f`
- inverse multiplication: :math:`A^{-1}\cdot f`
- adjoint inverse multiplication: :math:`(A^\dagger)^{-1}\cdot f`

(For linear operators, inverse adjoint multiplication and adjoint inverse
multiplication are equivalent.)

These different actions of an operator ``Op`` on a field ``f`` can be invoked
in various ways:

- direct multiplication: ``Op(f)`` or ``Op.times(f)`` or ``Op.apply(f, Op.TIMES)``
- adjoint multiplication: ``Op.adjoint_times(f)`` or ``Op.apply(f, Op.ADJOINT_TIMES)``
- inverse multiplication: ``Op.inverse_times(f)`` or ``Op.apply(f, Op.INVERSE_TIMES)``
- adjoint inverse multiplication: ``Op.adjoint_inverse_times(f)`` or ``Op.apply(f, Op.ADJOINT_INVERSE_TIMES)``

Operator classes defined in NIFTy may implement an arbitrary subset of these
four operations. This subset can be queried using the
:attr:`~LinearOperator.capability` property.

If needed, the set of supported operations can be enhanced by iterative
inversion methods;
for example, an operator defining direct and adjoint multiplication could be
enhanced by this approach to support the complete set. This functionality is
provided by NIFTy's :class:`InversionEnabler` class, which is itself a linear
operator.

There are two :class:`DomainTuple` objects associated with a
:class:`LinearOperator`: a :attr:`~LinearOperator.domain` and a
:attr:`~LinearOperator.target`.
Direct multiplication and adjoint inverse multiplication transform a field
living on the operator's :attr:`~LinearOperator.domain` to one living on the operator's :attr:`~LinearOperator.target`, whereas adjoint multiplication
and inverse multiplication transform from :attr:`~LinearOperator.target` to :attr:`~LinearOperator.domain`.

Operators with identical domain and target can be derived from
:class:`EndomorphicOperator`; typical examples for this category are the :class:`ScalingOperator`, which simply multiplies its input by a scalar
value, and :class:`DiagonalOperator`, which multiplies every value of its input
field with potentially different values.

Further operator classes provided by NIFTy are

- :class:`HarmonicTransformOperator` for transforms from a harmonic domain to
  its counterpart in position space, and their adjoint
- :class:`PowerDistributor` for transforms from a :class:`PowerSpace` to
  an associated harmonic domain, and their adjoint
- :class:`GeometryRemover`, which transforms from structured domains to
  unstructured ones. This is typically needed when building instrument response
  operators.


Syntactic sugar
---------------

Nifty4 allows simple and intuitive construction of altered and combined
operators.
As an example, if ``A``, ``B`` and ``C`` are of type :class:`LinearOperator`
and ``f1`` and ``f2`` are of type :class:`Field`, writing::

    X = A*B.inverse*A.adjoint + C
    f2 = X(f1)

will perform the operation suggested intuitively by the notation, checking
domain compatibility while building the composed operator.
The combined operator infers its domain and target from its constituents,
as well as the set of operations it can support.
The properties :attr:`~LinearOperator.adjoint` and
:attr:`~LinearOperator.inverse` return a new operator which behaves as if it
were the original operator's adjoint or inverse, respectively.


.. _minimization:


Minimization
============

Most problems in IFT are solved by (possibly nested) minimizations of
high-dimensional functions, which are often nonlinear.


Energy functionals
------------------

In NIFTy4 such functions are represented by objects of type :class:`Energy`.
These hold the prescription how to calculate the function's
:attr:`~Energy.value`, :attr:`~Energy.gradient` and
(optionally) :attr:`~Energy.curvature` at any given :attr:`~Energy.position`
in parameter space.
Function values are floating-point scalars, gradients have the form of fields
living on the energy's position domain, and curvatures are represented by
linear operator objects.

Energies are classes that typically have to be provided by the user when
tackling new IFT problems.
Some examples of concrete energy classes delivered with NIFTy4 are
:class:`QuadraticEnergy` (with position-independent curvature, mainly used with
conjugate gradient minimization) and :class:`~nifty4.library.WienerFilterEnergy`.


Iteration control
-----------------

Iterative minimization of an energy reqires some means of
checking the quality of the current solution estimate and stopping once
it is sufficiently accurate. In case of numerical problems, the iteration needs
to be terminated as well, returning a suitable error description.

In NIFTy4, this functionality is encapsulated in the abstract
:class:`IterationController` class, which is provided with the initial energy
object before starting the minimization, and is updated with the improved
energy after every iteration. Based on this information, it can either continue
the minimization or return the current estimate indicating convergence or
failure.

Sensible stopping criteria can vary significantly with the problem being
solved; NIFTy provides one concrete sub-class of :class:`IterationController`
called :class:`GradientNormController`, which should be appropriate in many
circumstances, but users have complete freedom to implement custom sub-classes
for their specific applications.


Minimization algorithms
-----------------------

All minimization algorithms in NIFTy inherit from the abstract
:class:`Minimizer` class, which presents a minimalistic interface consisting
only of a :meth:`~Minimizer.__call__` method taking an :class:`Energy` object
and optionally a preconditioning operator, and returning the energy at the
discovered minimum and a status code.

For energies with a quadratic form (i.e. which
can be expressed by means of a :class:`QuadraticEnergy` object), an obvious
choice of algorithm is the :class:`ConjugateGradient` minimizer.

A similar algorithm suited for nonlinear problems is provided by
:class:`NonlinearCG`.

Many minimizers for nonlinear problems can be characterized as

- first deciding on a direction for the next step
- then finding a suitable step length along this direction, resulting in the
  next energy estimate.

This family of algorithms is encapsulated in NIFTy's :class:`DescentMinimizer`
class, which currently has three concrete implementations:
:class:`SteepestDescent`, :class:`VL_BFGS`, and :class:`RelaxedNewton`.
Of these algorithms, only :class:`RelaxedNewton` requires the energy object to
provide a :attr:`~Energy.curvature` property, the others only need energy
values and gradients.

The flexibility of NIFTy's design allows using externally provided
minimizers. With only small effort, adapters for two SciPy minimizers were
written; they are available under the names :class:`NewtonCG` and
:class:`L_BFGS_B`.


Application to operator inversion
---------------------------------

It is important to realize that the machinery presented here cannot only be
used for minimizing IFT Hamiltonians, but also for the numerical inversion of
linear operators, if the desired application mode is not directly available.
A classical example is the information propagator

:math:`D = \left(R^\dagger N^{-1} R + S^{-1}\right)^{-1}`,

which must be applied when calculating a Wiener filter. Only its inverse
application is straightforward; to use it in forward direction, we make use
of NIFTy's :class:`InversionEnabler` class, which internally performs a
minimization of a :class:`QuadraticEnergy` by means of the
:class:`ConjugateGradient` algorithm.
