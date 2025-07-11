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

.. currentmodule:: nifty8.domains.domain

One of the fundamental building blocks of the NIFTy8 framework is the *domain*.
Its required capabilities are expressed by the abstract :py:class:`Domain` class.
A domain must be able to answer the following queries:

- its total number of data entries (pixels), which is accessible via the
  :attr:`~Domain.size` property
- the shape of the array that is supposed to hold these data entries
  (obtainable by means of the :attr:`~Domain.shape` property)
- equality comparison to another :class:`Domain` instance


Unstructured domains
--------------------

.. currentmodule:: nifty8.domains.unstructured_domain

Domains can be either *structured* (i.e. there is geometrical information
associated with them, like position in space and volume factors),
or *unstructured* (meaning that the data points have no associated manifold).

Unstructured domains can be described by instances of NIFTy's
:class:`UnstructuredDomain` class.


Structured domains
------------------

.. currentmodule:: nifty8.domains.structured_domain

In contrast to unstructured domains, these domains have an assigned geometry.
NIFTy requires them to provide the volume elements of their grid cells.
The additional methods are specified in the abstract class
:class:`StructuredDomain`:

- The properties :attr:`~StructuredDomain.scalar_dvol`,
  :attr:`~StructuredDomain.dvol`, and  :attr:`~StructuredDomain.total_volume`
  provide information about the domain's pixel volume(s) and its total volume.
- The property :attr:`~StructuredDomain.harmonic` specifies whether a domain
  is harmonic (i.e. describes a frequency space) or not
- If (and only if) the domain is harmonic, the methods
  :meth:`~StructuredDomain.get_k_length_array`,
  :meth:`~StructuredDomain.get_unique_k_lengths`, and
  :meth:`~StructuredDomain.get_fft_smoothing_kernel_function` provide absolute
  distances of the individual grid cells from the origin and assist with
  Gaussian convolution.

NIFTy comes with several concrete subclasses of :class:`StructuredDomain`:

.. currentmodule:: nifty8.domains

- :class:`~rg_space.RGSpace` represents a regular Cartesian grid with an arbitrary
  number of dimensions, which is supposed to be periodic in each dimension.
- :class:`~log_rg_space.LogRGSpace` implements a Cartesian grid with logarithmically
  spaced bins and an arbitrary number of dimensions.
- :class:`~hp_space.HPSpace` and :class:`~gl_space.GLSpace` describe pixelisations of the
  2-sphere; their counterpart in harmonic space is :class:`~lm_space.LMSpace`, which
  contains spherical harmonic coefficients.
- :class:`~power_space.PowerSpace` is used to describe one-dimensional power spectra.

Among these, :class:`~rg_space.RGSpace` and :class:`~log_rg_space.LogRGSpace` can
be harmonic or not (depending on constructor arguments),
:class:`~gl_space.GLSpace`, :class:`~hp_space.HPSpace`,
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
- a polarized field, which could be modelled as a product of any structured
  domain (representing location) with a four-element
  :class:`~unstructured_domain.UnstructuredDomain` holding Stokes I, Q, U and V components.
- a model for the sky emission, which holds both the current realisation
  (on a harmonic domain) and a few inferred model parameters (e.g. on an
  unstructured grid).

.. currentmodule:: nifty8

Consequently, NIFTy defines a class called :class:`~domain_tuple.DomainTuple`
holding a sequence of :class:`~domains.domain.Domain` objects. The full domain is
specified as the product of all elementary domains. Thus, an instance of
:class:`~domain_tuple.DomainTuple` would be suitable to describe the first two
examples above. In principle, a :class:`~domain_tuple.DomainTuple`
can even be empty, which implies that the field living on it is a scalar.

A :class:`~domain_tuple.DomainTuple` supports iteration and indexing, and also
provides the properties :attr:`~domain_tuple.DomainTuple.shape` and
:attr:`~domain_tuple.DomainTuple.size` in analogy to the elementary
:class:`~domains.domain.Domain`.

An aggregation of several :class:`~domain_tuple.DomainTuple` s, each member
identified by a name, is described by the :class:`~multi_domain.MultiDomain`
class. In contrast to a :class:`~domain_tuple.DomainTuple` a
:class:`~multi_domain.MultiDomain` is a collection and does not define the
product space of its elements.  It would be the adequate space to use in the
last of the above examples.

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

Fields support a wide range of arithmetic operations, either involving
two fields of equal domains or a field and a scalar. Arithmetic operations are
performed point-wise, and the returned field has the same domain as the input field(s).
Available operators are addition ("+"), subtraction ("-"),
multiplication ("*"), division ("/"), floor division ("//") and
exponentiation ("**"). Inplace operators ("+=", etc.) are not supported.
Further, boolean operators, performing a point-wise comparison of a field with
either another field of equal domain or a scalar, are available as well. These
include equals ("=="), not equals ("!="), less ("<"), less or equal ("<="),
greater (">") and greater or equal (">=). The domain of the field returned equals
that of the input field(s), while the stored data is of boolean type.

Contractions (like summation, integration, minimum/maximum, computation of
statistical moments) can be carried out either over an entire field (producing
a scalar result) or over sub-domains (resulting in a field defined on a smaller
domain). Scalar products of two fields can also be computed easily as well.
See the documentation of :class:`~field.Field` for details.

There is also a set of convenience functions to generate fields with constant
values or fields filled with random numbers according to a user-specified
distribution: :attr:`~sugar.full`, :attr:`~sugar.from_random`.

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
multiplied, chained (via the :attr:`__call__` method or the `@` operator) and
support point-wise application of functions like :class:`exp()`, :class:`log()`,
:class:`sqrt()`, :class:`conjugate()`.


Advanced operators
------------------

NIFTy provides a library of commonly employed operators which can be used for
specific inference problems. Currently these are:

.. currentmodule:: nifty8.library

- :class:`~smooth_linear_amplitude.SLAmplitude`, which returns a smooth power spectrum.
- :class:`~inverse_gamma_operator.InverseGammaOperator`, which models point sources which are
  distributed according to a inverse-gamma distribution.
- :class:`~correlated_fields.CorrelatedField`, which models a diffuse field whose correlation
  structure is described by an amplitude operator.


Linear Operators
================

.. currentmodule:: nifty8.operators

A linear operator (represented by NIFTy8's abstract
:class:`~linear_operator.LinearOperator` class) is derived from
:class:`~operator.Operator` and can be interpreted as an (implicitly defined)
matrix. Since its operation is linear, it can provide some additional
functionality which is not available for the more generic
:class:`~operator.Operator` class.


Linear Operator basics
----------------------

There are four basic ways of applying an operator :math:`A` to a field :math:`s`:

- direct application: :math:`A(s)`
- adjoint application: :math:`A^\dagger (s)`
- inverse application: :math:`A^{-1} (s)`
- adjoint inverse application: :math:`(A^\dagger)^{-1} (s)`

Note: The inverse of the adjoint of a linear map and the adjoint of the inverse
of a linear map (if all those exist) are the same.

These different actions of a linear operator ``Op`` on a field ``f`` can be
invoked in various ways:

- direct multiplication: ``Op(f)`` or ``Op.times(f)`` or ``Op.apply(f, Op.TIMES)``
- adjoint multiplication: ``Op.adjoint_times(f)`` or ``Op.apply(f, Op.ADJOINT_TIMES)``
- inverse multiplication: ``Op.inverse_times(f)`` or ``Op.apply(f, Op.INVERSE_TIMES)``
- adjoint inverse multiplication: ``Op.adjoint_inverse_times(f)`` or ``Op.apply(f, Op.ADJOINT_INVERSE_TIMES)``

Operator classes defined in NIFTy may implement an arbitrary subset of these
four operations. This subset can be queried using the
:attr:`~linear_operator.LinearOperator.capability` property.

If needed, the set of supported operations can be enhanced by iterative
inversion methods; for example, an operator defining direct and adjoint
multiplication could be enhanced by this approach to support the complete set.
This functionality is provided by NIFTy's
:class:`~inversion_enabler.InversionEnabler` class, which is itself a linear
operator.

.. currentmodule:: nifty8.operators.operator

Direct multiplication and adjoint inverse multiplication transform a field
defined on the operator's :attr:`~Operator.domain` to one defined on the
operator's :attr:`~Operator.target`, whereas adjoint multiplication and inverse
multiplication transform from :attr:`~Operator.target` to
:attr:`~Operator.domain`.

.. currentmodule:: nifty8.operators

Operators with identical domain and target can be derived from
:class:`~endomorphic_operator.EndomorphicOperator`. Typical examples for this
category are the :class:`~scaling_operator.ScalingOperator`, which simply
multiplies its input by a scalar value, and
:class:`~diagonal_operator.DiagonalOperator`, which multiplies every value of
its input field with potentially different values.

.. currentmodule:: nifty8

Further operator classes provided by NIFTy are

- :class:`~operators.harmonic_operators.HarmonicTransformOperator` for
  transforms from a harmonic domain to its counterpart in position space, and
  their adjoint
- :class:`~operators.distributors.PowerDistributor` for transforms from a
  :class:`~domains.power_space.PowerSpace` to an associated harmonic domain, and
  their adjoint.
- :class:`~operators.simple_linear_operators.GeometryRemover`, which transforms
  from structured domains to unstructured ones. This is typically needed when
  building instrument response operators.


Syntactic sugar
---------------

NIFTy allows simple and intuitive construction of altered and combined
operators.
As an example, if ``A``, ``B`` and ``C`` are of type :class:`~operators.linear_operator.LinearOperator`
and ``f1`` and ``f2`` are of type :class:`~field.Field`, writing::

    X = A(B.inverse(A.adjoint)) + C
    f2 = X(f1)

.. currentmodule:: nifty8.operators.linear_operator

will perform the operation suggested intuitively by the notation, checking
domain compatibility while building the composed operator.
The properties :attr:`~LinearOperator.adjoint` and
:attr:`~LinearOperator.inverse` return a new operator which behaves as if it
were the original operator's adjoint or inverse, respectively.
The combined operator infers its domain and target from its constituents,
as well as the set of operations it can support.
Instantiating operator adjoints or inverses by :attr:`~LinearOperator.adjoint`
and similar methods is to be distinguished from the instant application of
operators performed by :attr:`~LinearOperator.adjoint_times` and similar
methods.


.. _minimization:


Minimization
============

Most problems in IFT are solved by (possibly nested) minimizations of
high-dimensional functions, which are often nonlinear.

.. currentmodule:: nifty8.minimization

Energy functionals
------------------

In NIFTy8 such functions are represented by objects of type
:class:`~energy.Energy`. These hold the prescription how to calculate the
function's :attr:`~energy.Energy.value`, :attr:`~energy.Energy.gradient` and
(optionally) :attr:`~energy.Energy.metric` at any given
:attr:`~energy.Energy.position` in parameter space. Function values are
floating-point scalars, gradients have the form of fields defined on the energy's
position domain, and metrics are represented by linear operator objects.

.. currentmodule:: nifty8

Energies are classes that typically have to be provided by the user when
tackling new IFT problems. An example of concrete energy classes delivered with
NIFTy8 is :class:`~minimization.quadratic_energy.QuadraticEnergy` (with
position-independent metric, mainly used with conjugate gradient minimization).

For MGVI and GeoVI, NIFTy provides
:func:`~minimization.kl_energies.SampledKLEnergy` that instantiate objects
containing the sampled estimate of the KL divergence, its gradient and the
Fisher metric. These constructors require an instance
of :class:`~operators.energy_operators.StandardHamiltonian`, an operator to
compute the negative log-likelihood of the problem in standardized coordinates
at a given position in parameter space.
Finally, the :class:`~operators.energy_operators.StandardHamiltonian`
can be constructed from the likelihood, represented by a
:class:`~operators.energy_operators.LikelihoodEnergyOperator` instance.
Several commonly used forms of the likelihoods are already provided in
NIFTy, such as :class:`~operators.energy_operators.GaussianEnergy`,
:class:`~operators.energy_operators.PoissonianEnergy`,
:class:`~operators.energy_operators.InverseGammaEnergy` or
:class:`~operators.energy_operators.BernoulliEnergy`, but the user
is free to implement any likelihood customized to the problem at hand.
The demo code `demos/getting_started_3.py` illustrates how to set up an energy
functional for MGVI and minimize it.



Iteration control
-----------------

.. currentmodule:: nifty8.minimization.iteration_controllers

Iterative minimization of an energy requires some means of
checking the quality of the current solution estimate and stopping once
it is sufficiently accurate. In case of numerical problems, the iteration needs
to be terminated as well, returning a suitable error description.

In NIFTy8, this functionality is encapsulated in the abstract
:class:`IterationController` class, which is provided with the initial energy
object before starting the minimization, and is updated with the improved
energy after every iteration. Based on this information, it can either continue
the minimization or return the current estimate indicating convergence or
failure.

Sensible stopping criteria can vary significantly with the problem being
solved; NIFTy provides a concrete sub-class of :class:`IterationController`
called :class:`GradientNormController`, which should be appropriate in many
circumstances. A full list of the available :class:`IterationController` s
in NIFTy can be found below, but users have complete freedom to implement custom
:class:`IterationController` sub-classes for their specific applications.

Minimization algorithms
-----------------------

.. currentmodule:: nifty8.minimization

All minimization algorithms in NIFTy inherit from the abstract
:class:`~minimizer.Minimizer` class, which presents a minimalistic interface
consisting only of a :meth:`~minimizer.Minimizer.__call__` method taking an
:class:`~energy.Energy` object and optionally a preconditioning operator, and
returning the energy at the discovered minimum and a status code.

For energies with a quadratic form (i.e. which can be expressed by means of a
:class:`~quadratic_energy.QuadraticEnergy` object), an obvious choice of
algorithm is the :class:`~conjugate_gradient.ConjugateGradient` minimizer.

A similar algorithm suited for nonlinear problems is provided by
:class:`~nonlinear_cg.NonlinearCG`.

Many minimizers for nonlinear problems can be characterized as

- First deciding on a direction for the next step.
- Then finding a suitable step length along this direction, resulting in the
  next energy estimate.

This family of algorithms is encapsulated in NIFTy's
:class:`~descent_minimizers.DescentMinimizer` class, which currently has three
generally usable concrete implementations:
:class:`~descent_minimizers.NewtonCG`, :class:`~descent_minimizers.L_BFGS` and
:class:`~descent_minimizers.VL_BFGS`. Of these algorithms, only
:class:`~descent_minimizers.NewtonCG` requires the energy object to provide
a :attr:`~energy.Energy.metric` property, the others only need energy values and
gradients. Further available descent minimizers are
:class:`~descent_minimizers.RelaxedNewton`
and :class:`~descent_minimizers.SteepestDescent`.

The flexibility of NIFTy's design allows using externally provided minimizers.
With only small effort, adaptors for two SciPy minimizers were written; they are
available under the names :class:`~scipy_minimizer.ScipyCG` and
:class:`~scipy_minimizer.L_BFGS_B`.


Application to operator inversion
---------------------------------

.. currentmodule:: nifty8

The machinery presented here cannot only be used for minimizing functionals
derived from IFT, but also for the numerical inversion of linear operators, if
the desired application mode is not directly available. A classical example is
the information propagator whose inverse is defined as:

:math:`D^{-1} = \left(R^\dagger N^{-1} R + S^{-1}\right)`.

It needs to be applied in forward direction in order to calculate the Wiener
filter solution, but only its inverse application is straightforward.
To use it in forward direction, we make use of NIFTy's
:class:`~operators.inversion_enabler.InversionEnabler` class, which internally
applies the (approximate) inverse of the given operator :math:`x = Op^{-1} (y)` by
solving the equation :math:`y = Op (x)` for :math:`x`.
This is accomplished by minimizing a suitable
:class:`~minimization.quadratic_energy.QuadraticEnergy`
with the :class:`~minimization.conjugate_gradient.ConjugateGradient`
algorithm. An example is provided in
:func:`~library.wiener_filter_curvature.WienerFilterCurvature`.


Posterior analysis and visualization
------------------------------------

After the minimization of an energy functional has converged, samples can be drawn
from the posterior distribution at the current position to investigate the result.
The probing module offers class called :class:`~probing.StatCalculator`
which allows to evaluate the :attr:`~probing.StatCalculator.mean` and the unbiased
variance :attr:`~probing.StatCalculator.var` of these samples.

Fields can be visualized using the :class:`~plot.Plot` class, which invokes
matplotlib for plotting.
