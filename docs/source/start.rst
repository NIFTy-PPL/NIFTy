.. _informal_label:

First steps -- An informal introduction
=======================================

NIFTy4 Tutorial
---------------

.. currentmodule:: nifty4

.. automodule:: nifty4

NIFTy4 enables the programming of grid and resolution independent algorithms.
This freedom is particularly desirable for signal inference algorithms, where
a continuous signal field is to be recovered.
It is achieved by means of an object-oriented infrastructure that comprises, among others, abstract classes for :ref:`Domains <domains>`, :ref:`Fields <fields>`, and :ref:`Operators <operators>`.
All those are covered in this tutorial.

You should be able to import NIFTy4 like this after a successful `installation <install.html>`_.

>>> import nifty4 as ift


Technical bird's eye view
.........................

The fundamental building blocks required for IFT computations are best recognized from a large distance, ignoring all technical details.

From such a perspective,

- IFT problems largely consist of *minimization* problems involving a large number of equations.
- The equations are built mostly from the application of *linear operators*, but there may also be nonlinear functions involved.
- The unknowns in the equations represent either continuous physical *fields*, or they are simply individual measured *data* points.
- The locations and volume elements attached to discretized *field* values are supplied by *domain* objects. There are many variants of such discretized *domain* supported by NIFTy4, including Cartesian and spherical geometries and their harmonic counterparts. *Fields* can live on arbitrary products of such *domains*.

In the following sections, the concepts briefly presented here will be discussed in more detail; this is done in reversed order of their introduction, to avoid forward references.


.. _domainobjects:

DomainObjects
.............

One of the fundamental building blocks of the NIFTy4 framework is the /domain/.
Its required capabilities are expressed by the abstract :py:class:`DomainObject` class.
A domain must be able to answer the following queries:

- its total number of data entries (pixels)
- the shape of the array that is supposed to hold them
- equality/unequality to another :py:class:`DomainObject` instance

.. _domains:

Unstructured domains
....................

There are domains (e.g. the data domain) which have no geometry associated to the individual data values.
In NIFTy4 they are represented by the :py:class:`UnstructuredDomain` class, which is derived from
:py:class:`DomainObject`.


Structured domains
..................

All domains defined on a geometrical manifold are derived from :py:class:`StructuredDomain` (which is in turn derived from :py:class:`DomainObject`).

In addition to the capabilities of :py:class:`DomainObject`, :py:class:`StructuredDomain` offers the following functionality:

- methods returing the pixel volume(s) and the total volume
- a :py:attr:`harmonic` property
- (iff the domain is harmonic) some methods concerned with Gaussian convolution and the absolute distances of the individual grid cells from the origin

Examples for structured domains are

- :py:class:`RGSpace` (an equidistant Cartesian grid with a user-definable number of dimensions),
- :py:class:`GLSpace` (a Gauss-Legendre grid on the sphere), and
- :py:class:`LMSpace` (a grid storing spherical harmonic coefficients).

Among these, :py:class:`RGSpace` can be harmonic or not (depending on constructor arguments), :py:class:`GLSpace` is a pure position domain (i.e. nonharmonic), and :py:class:`LMSpace` is always harmonic.

Full domains
............

A field can live on a single domain, but it can also live on a product of domains (or no domain at all, in which case it is a scalar).
The tuple of domain on which a field lives is described by the :py:class:`DomainTuple` class.
A :py:class:`DomainTuple` object can be constructed from

- a single instance of anything derived from :py:class:`DomainTuple`
- a tuple of such instances (possibly empty)
- another :py:class:`DomainTuple` object

.. _fields:

Fields
......

A :py:class:`Field` object consists of the following components:

- a domain in form of a :py:class:`DomainTuple` object
- a data type (e.g. numpy.float64)
- an array containing the actual values

Fields support arithmetic operations, contractions, etc.

.. _operators:

Linear Operators
................

A linear operator (represented by NIFTy4's abstract :py:class:`LinearOperator` class) can be interpreted as an (implicitly defined) matrix.
It can be applied to :py:class:`Field` instances, resulting in other :py:class:`Field` instances that potentially live on other domains.

There are four basic ways of applying an operator :math:`A` to a field :math:`f`:

- direct multiplication: :math:`A\cdot f`
- adjoint multiplication: :math:`A^\dagger \cdot f`
- inverse multiplication: :math:`A^{-1}\cdot f`
- adjoint inverse multiplication: :math:`(A^\dagger)^{-1}\cdot f`

(For linear operators, inverse adjoint multiplication and adjoint inverse multiplication are equivalent.)

Operator classes defined in NIFTy may implement an arbitrary subset of these four operations.
If needed, the set of supported operations can be enhanced by iterative inversion methods;
for example, an operator defining direct and adjoint multiplication, could be enhanced to support the complete set by this method.

There are two domains associated with a :py:class:`LinearOperator`: a *domain* and a *target*.
Direct multiplication and adjoint inverse multiplication transform a field living on the operator's *domain* to one living on the operator's *target*, whereas adjoint multiplication and inverse multiplication transform from *target* to *domain*.

Operators with identical domain and target can be derived from :py:class:`EndomorphicOperator`;
typical examples for this category are the :py:class:`ScalingOperator`, which simply multiplies its input by a scalar value and :py:class:`DiagonalOperator`, which multiplies every value of its input field with potentially different values.

Nifty4 allows simple and intuitive construction of combined operators.
As an example, if :math:`A`, :math:`B` and :math:`C` are of type :py:class:`LinearOperator` and :math:`f_1` and :math:`f_2` are fields, writing::
    X = A*B.inverse*A.adjoint + C
    f2 = X(f1)

will perform the operation suggested intuitively by the notation, checking domain compatibility while building the composed operator.
The combined operator infers its domain and target from its constituents, as well as the set of operations it can support.


.. _minimization:

Minimization
............

Most problems in IFT are solved by (possibly nested) minimizations of high-dimensional functions, which are often nonlinear.

In NIFTy4 such functions are represented by objects of type :py:class:`Energy`.
These hold the prescription how to calculate the function's value, gradient and (optionally) curvature at any given position.
Function values are floating-point scalars, gradients have the form of fields living on the energy's position domain, and curvatures are represented by linear operator objects.

Some examples of concrete energy classes delivered with NIFTy4 are :py:class:`QuadraticEnergy` (with position-independent curvature, mainly used with conjugate gradient minimization) and :py:class:`WienerFilterEnergy`.
Energies are classes that typically have to be provided by the user when tackling new IFT problems.

The minimization procedure can be carried out by one of several algorithms; NIFTy4 currently ships solvers based on

- the conjugate gradient method (for quadratic energies)
- the steepest descent method
- the VL-BFGS method
- the relaxed Newton method, and
- a nonlinear conjugate gradient method

