
=============
Code Overview
=============


Executive summary
=================

The fundamental building blocks required for IFT computations are best
recognized from a large distance, ignoring all technical details.

From such a perspective,

- IFT problems largely consist of the combination of several high dimensional
  *minimization* problems.
- Within NIFTy, *operators* are used to define the characteristic equations and
  properties of the problems.
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

.. currentmodule:: nifty5.domains.domain

One of the fundamental building blocks of the NIFTy5 framework is the *domain*.
Its required capabilities are expressed by the abstract :py:class:`Domain` class.
A domain must be able to answer the following queries:
m

- its total number of data entries (pixels), which is accessible via the
  :attr:`~Domain.size` property
- the shape of the array that is supposed to hold these data entries
  (obtainable by means of the :attr:`~Domain.shape` property)
- equality comparison to another :class:`Domain` instance


Unstructured domains
--------------------

.. currentmodule:: nifty5.domains.unstructured_domain

Domains can be either *structured* (i.e. there is geometrical information
associated with them, like position in space and volume factors),
or *unstructured* (meaning that the data points have no associated manifold).

Unstructured domains can be described by instances of NIFTy's
:class:`UnstructuredDomain` class.


Structured domains
------------------

.. currentmodule:: nifty5.domains.structured_domain

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

.. currentmodule:: nifty5.domains

- :class:`~rg_space.RGSpace` represents a regular Cartesian grid with an arbitrary
  number of dimensions, which is supposed to be periodic in each dimension.
- :class:`~hp_space.HPSpace` and :class:`~gl_space.GLSpace` describe pixelisations of the
  2-sphere; their counterpart in harmonic space is :class:`~lm_space.LMSpace`, which
  contains spherical harmonic coefficients.
- :class:`~power_space.PowerSpace` is used to describe one-dimensional power spectra.

Among these, :class:`~rg_space.RGSpace` can be harmonic or not (depending on
constructor arguments), :class:`~gl_space.GLSpace`, :class:`~hp_space.HPSpace`,
and :class:`~power_space.PowerSpace` are pure position domains (i.e.
nonharmonic), and :class:`~lm_space.LMSpace` is always harmonic.


Combinations of domains
=======================

The fundamental classes described above are often sufficient to specify the
domain of a field. In some cases, however, it will be necessary to define the
field on a product of elementary domains instead of a single one.
More sophisticated operators also require a set of several such fields.
Some examples are:

- sky emission depending on location and energy. This could be represented by a
  product of an :class:`~hp_space.HPSpace` (for location) with an
  :class:`~rg_space.RGSpace` (for energy).
- a polarized field, which could be modeled as a product of any structured
  domain (representing location) with a four-element
  :class:`~unstructured_domain.UnstructuredDomain` holding Stokes I, Q, U and V components.
- a model for the sky emission, which holds both the current realization
  (on a harmonic domain) and a few inferred model parameters (e.g. on an
  unstructured grid).

.. currentmodule:: nifty5

Consequently, NIFTy defines a class called :class:`~domain_tuple.DomainTuple`
holding a sequence of :class:`~domains.domain.Domain` objects, which is used to
specify full field domains. In principle, a :class:`~domain_tuple.DomainTuple`
can even be empty, which implies that the field living on it is a scalar.

A :class:`~domain_tuple.DomainTuple` supports iteration and indexing, and also
provides the properties :attr:`~domain_tuple.DomainTuple.shape`,
:attr:`~domain_tuple.DomainTuple.size` in analogy to the elementary
:class:`~domains.domain.Domain`.

An aggregation of several :class:`~domain_tuple.DomainTuple` s, each member
identified by a name, is described by the :class:`~multi_domain.MultiDomain`
class.

Fields
======

Fields on a single DomainTuple
------------------------------

A :class:`~field.Field` object consists of the following components:

- a domain in form of a :class:`~domain_tuple.DomainTuple` object
- a data type (e.g. numpy.float64)
- an array containing the actual values

Usually, the array is stored in the form of a ``numpy.ndarray``, but for very
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

Like almost all NIFTy objects, fields are immutable: their value or any other
attribute cannot be modified after construction. To manipulate a field in ways
that are not covered by the provided standard operations, its data content must
be extracted first, then changed, and a new field has to be created from the
result.

Fields defined on a MultiDomain
-------------------------------

The :class:`~multi_field.MultiField` class can be seen as a dictionary of
individual :class:`~field.Field` s, each identified by a name, which is defined
on a :class:`~multi_domain.MultiDomain`.


Operators
=========

All transformations between different NIFTy fields are expressed in the form of
:class:`~operators.operator.Operator` objects. The interface of this class is
rather minimalistic: it has a property called
:attr:`~operators.operator.Operator.domain` which returns a
:class:`~domain_tuple.DomainTuple` or :class:`~multi_domain.MultiDomain` object
specifying the structure of the :class:`~field.Field` or
:class:`~multi_field.MultiField` it expects as input, another property
:attr:`~operators.operator.Operator.target` describing its output, and finally
an overloaded :attr:`~operators.operator.Operator.apply` method, which can take:

- a :class:`~field.Field`/:class:`~multi_field.MultiField` object, in which case
  it returns the transformed :class:`~field.Field`/:class:`~multi_field.MultiField`.
- a :class:`~linearization.Linearization` object, in which case it returns the
  transformed :class:`~linearization.Linearization`.

This is the interface that all objects derived from
:class:`~operators.operator.Operator` must implement. In addition,
:class:`~operators.operator.Operator` objects can be added/subtracted,
multiplied, chained (via the :attr:`__call__` method
or the `@` operator) and support point-wise application of functions like
:class:`exp()`, :class:`log()`, :class:`sqrt()`, :class:`conjugate()`.


Advanced operators
------------------

NIFTy provides a library of commonly employed operators which can be used for
specific inference problems. Currently these are:

- :class:`AmplitudeOperator`, which returns a smooth power spectrum.
- :class:`InverseGammaOperator`, which models point sources which are
  distributed according to a inverse-gamma distribution.
- :class:`CorrelatedField`, which models a diffuse log-normal field. It takes an
  amplitude operator to specify the correlation structure of the field.


Linear Operators
================

A linear operator (represented by NIFTy5's abstract :class:`operators.linear_operator.LinearOperator`
class) is derived from `Operator` and can be interpreted as an
(implicitly defined) matrix. Since its operation is linear, it can provide some
additional functionality which is not available for the more generic :class:`operators.operator.Operator`
class.


Linear Operator basics
----------------------

There are four basic ways of applying an operator :math:`A` to a field :math:`f`:

- direct application: :math:`A\cdot f`
- adjoint application: :math:`A^\dagger \cdot f`
- inverse application: :math:`A^{-1}\cdot f`
- adjoint inverse application: :math:`(A^\dagger)^{-1}\cdot f`

(Because of the linearity, inverse adjoint and adjoint inverse application
are equivalent.)

These different actions of a linear operator ``Op`` on a field ``f`` can be
invoked in various ways:

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

Nifty5 allows simple and intuitive construction of altered and combined
operators.
As an example, if ``A``, ``B`` and ``C`` are of type :class:`operators.linear_operator.LinearOperator`
and ``f1`` and ``f2`` are of type :class:`~field.Field`, writing::

    X = A(B.inverse(A.adjoint)) + C
    f2 = X(f1)

will perform the operation suggested intuitively by the notation, checking
domain compatibility while building the composed operator.
The combined operator infers its domain and target from its constituents,
as well as the set of operations it can support.
The properties :attr:`~LinearOperator.adjoint` and
:attr:`~LinearOperator.inverse` return a new operator which behaves as if it
were the original operator's adjoint or inverse, respectively.


Operators
=========

Operator classes (represented by NIFTy5's abstract :class:`operators.operator.Operator` class) are used to construct
the equations of a specific inference problem.
Most operators are defined via a position, which is a :class:`~multi_field.MultiField` object,
their value at this position, which is again a :class:`~multi_field.MultiField` object and a Jacobian derivative,
which is a :class:`operators.linear_operator.LinearOperator` and is needed for the minimization procedure.

Using the existing basic operator classes one can construct more complicated operators, as
NIFTy allows for easy and self-consinstent combination via point-wise multiplication,
addition and subtraction. The operator resulting from these operations then automatically
contains the correct Jacobians, positions and values.
Notably, :class:`Constant` and :class:`Variable` allow for an easy way to turn
inference of specific quantities on and off.

The basic operator classes also allow for more complex operations on operators such as
the application of :class:`LinearOperators` or local non-linearities.
As an example one may consider the following combination of ``x``, which is an operator of type
:class:`Variable` and ``y``, which is an operator of type :class:`Constant`::

	z = x*x + y

``z`` will then be an operator with the following properties::

	z.value = x.value*x.value + y.value
	z.position = Union(x.position, y.position)
	z.jacobian = 2*makeOp(x.value)


Basic operators
---------------
# FIXME All this is outdated!

Basic operator classes provided by NIFTy are

- :class:`Constant` contains a constant value and has a zero valued Jacobian.
  Like other operators, it has a position, but its value does not depend on it.
- :class:`Variable` returns the position as its value, its derivative is one.
- :class:`LinearModel` applies a :class:`operators.linear_operator.LinearOperator` on the model.
- :class:`LocalModel` applies a non-linearity locally on the model.
	value and Jacobian are combined into corresponding :class:`MultiFields` and operators.


Advanced operators
------------------

NIFTy also provides a library of more sophisticated operators which are used for more
specific inference problems. Currently these are:

- :class:`AmplitudeOperator`, which returns a smooth power spectrum.
- :class:`InverseGammaOperator`, which models point sources which follow a inverse gamma distribution.
- :class:`CorrelatedField`, which models a diffuse log-normal field. It takes an amplitude operator
	to specify the correlation structure of the field.


.. _minimization:


Minimization
============

Most problems in IFT are solved by (possibly nested) minimizations of
high-dimensional functions, which are often nonlinear.


Energy functionals
------------------

In NIFTy5 such functions are represented by objects of type :class:`Energy`.
These hold the prescription how to calculate the function's
:attr:`~Energy.value`, :attr:`~Energy.gradient` and
(optionally) :attr:`~Energy.metric` at any given :attr:`~Energy.position`
in parameter space.
Function values are floating-point scalars, gradients have the form of fields
living on the energy's position domain, and metrics are represented by
linear operator objects.

Energies are classes that typically have to be provided by the user when
tackling new IFT problems.
Some examples of concrete energy classes delivered with NIFTy5 are
:class:`QuadraticEnergy` (with position-independent metric, mainly used with
conjugate gradient minimization) and :class:`~nifty5.library.WienerFilterEnergy`.


Iteration control
-----------------

Iterative minimization of an energy reqires some means of
checking the quality of the current solution estimate and stopping once
it is sufficiently accurate. In case of numerical problems, the iteration needs
to be terminated as well, returning a suitable error description.

In NIFTy5, this functionality is encapsulated in the abstract
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
provide a :attr:`~Energy.metric` property, the others only need energy
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
