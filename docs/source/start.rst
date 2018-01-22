.. _informal_label:

First steps -- An informal introduction
=======================================

NIFTy4 Tutorial
---------------

.. currentmodule:: nifty4

.. automodule:: nifty4

NIFTy4 enables the programming of grid and resolution independent algorithms.
In particular for signal inference algorithms, where a continuous signal field is to be recovered, this freedom is desirable.
It is achieved with an object-oriented infrastructure that comprises, among others, abstract classes for :ref:`Spaces <spaces>`, :ref:`Fields <fields>`, and :ref:`Operators <operators>`.
All those are covered in this tutorial.

You should be able to import NIFTy4 like this after a successful `installation <install.html>`_.

>>> import nifty4 as ift

.. _domainobjects:

DomainObjects
.............

One of the fundamental building blocks of the NIFTy4 framework is the /domain/.
Its required capabilities are expressed by the abstract :py:class:`DomainObject` class.
A domain must be able to answer the following queries:

- its total number of data entries (pixels)
- the shape of the array that is supposed to hold them
- the pixel volume(s)
- the total volume
- equality/unequality to another :py:class:`DomainObject` instance

.. _spaces:

Unstructured spaces
...................

There are domains (e.g. the data domain) which have no geometry associated to the individual data values.
In NIFTy4 they are represented by the :py:class:`FieldArray` class, which is dreived from
:py:class:`DomainObject` and simply returns 1.0 as the volume element for every pixel and the total
number of pixels as the total volume.


Structured Spaces
.................

All domains defined on a geometrical manifold are derived from :py:class:`Space` (which is in turn derived from :py:class:`DomainObject`).

In addition to the capabilities of :py:class:`DomainObject`, :py:class:`Space` offers the following functionality:

- a :py:attr:`harmonic` property
- (iff the space is harmonic) some methods concerned with Gaussian convolution and the absolute distances of the individual grid cells from the origin

Examples for structured spaces are

- :py:class:`RGSpace` (an equidistant Cartesian grid with a user-definable number of dimensions),
- :py:class:`GLSpace` (a Gauss-Legendre grid on the sphere), and
- :py:class:`LMSpace` (a grid storing spherical harmonic coefficients).

Among these, :py:class:`RGSpace` can be harmonic or not (depending on constructor arguments), :py:class:`GLSpace` is a pure position space (i.e. nonharmonic), and :py:class:`LMSpace` is always harmonic.

Full domains
............

A field can live on a single space, but it can also live on a product of spaces (or no space at all, in which case it is a scalar).
The tuple of spaces on which a field lives is a called a *domain* in NIFTy terminology; it is described by a :py:class:`DomainTuple` object.
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


.. _minimization:

Minimization
............
