.. _informal_label:

First steps -- An informal introduction
=======================================

NIFTy4 Tutorial
---------------

.. currentmodule:: nifty4

.. automodule:: nifty4

NIFTy4 enables the programming of grid and resolution independent algorithms. In particular for signal inference algorithms, where a continuous signal field is to be recovered, this freedom is desirable. This is achieved with an object-oriented infrastructure that comprises, among others, abstract classes for :ref:`Spaces <spaces>`, :ref:`Fields <fields>`, and :ref:`Operators <operators>`. All those are covered in this tutorial.

You should be able to import NIFTy4 like this after a successful `installation <install.html>`_.

>>> import nifty4 as ift

.. _spaces:

Spaces
......

The very foundation of NIFTy4's framework are spaces, all of which are derived from the :py:class:`DomainObject` class.

A space can be either unstructured or live on a geometrical manifold; the former case is supported by objects of type :py:class:`FieldArray`, while the latter must be derived from :py:class:`Space` (both of which are in turn derived from :py:class:`DomainObject`).
Examples for structured spaces are

- :py:class:`RGSpace` (an equidistant Cartesian grid with a user-definable number of dimensions),
- :py:class:`GLSpace` (a Gauss-Legendre grid on the sphere), and
- :py:class:`LMSpace` (a grid storing spherical harmonic coefficients).


Domains
-------

A field can live on a single space, but it can also live on a product of spaces (or no space at all, in which case it is a scalar).
The set of spaces on which a field lives is a called a _domain_ in NIFTy terminology; it is described by a :py:class:`DomainTuple` object.

.. _fields:

Fields
------

A field object is specified by

- a domain in form of a :py:class:`DomainTuple` object
- a data type (e.g. numpy.float64)
- an array containing the actual values

Fields support arithmetic operations, contractions, etc.

.. _operators:

Operators
---------

