Operators
=========
Operators perform some operation on a given field. In practice an operator can 
take the form of an explicit matrix (e.g. stored in a Numpy array) or it may be 
implicitly defined as a function (e.g. an FFT operation would not be encoded in
a matrix, but performed using an FFT routine). NIFTY includes a framework for 
handling arbitrary operators, and basic methods for manipulating these 
operators. Common functions like taking traces and extracting diagonals are 
provided.

In order to have a blueprint for operators capable of handling fields, any 
application of operators is split into a general and a concrete part. The 
general part comprises the correct involvement of normalizations and 
transformations, necessary for any operator type, while the concrete part is 
unique for each operator subclass. In analogy to the field class, any operator 
instance has a set of properties that specify its domain and target as well as 
some additional flags.

Operator classes
----------------
NIFTY provides a base class for defining operators, as well as several pre-implemented operator types that are very often needed for signal inference
algorithms.

.. toctree:: 
    :maxdepth: 1

    diagonal_operator
    fft_operator
    composed_operator
    response_operator
    smoothing_operator
    projection_operator
    propagator_operator
    endomorphic_operator
    invertible_operator_mixin
    transformations

.. currentmodule:: nifty

The ``LinearOperator`` class -- The base Operator Object
--------------------------------------------------------

.. autoclass:: LinearOperator
    :show-inheritance:
    :members:
