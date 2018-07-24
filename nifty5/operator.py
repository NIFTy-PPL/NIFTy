from __future__ import absolute_import, division, print_function

import abc

import numpy as np

from .compat import *
from .utilities import NiftyMetaBase
#from ..domain_tuple import DomainTuple
#from ..multi.multi_domain import MultiDomain
from .field import Field
from .multi.multi_field import MultiField


class Linearization(object):
    def __init__(self, val, jac):
        self._val = val
        self._jac = jac

    @property
    def domain(self):
        return self._jac.domain

    @property
    def target(self):
        return self._jac.target

    @property
    def val(self):
        return self._val

    @property
    def jac(self):
        return self._jac

    def __neg__(self):
        return Linearization(-self._val, self._jac*(-1))

    def __add__(self, other):
        if isinstance(other, Linearization):
            return Linearization(self._val+other._val, self._jac+other._jac)
        if isinstance(other, (int, float, complex, Field, MultiField)):
            return Linearization(self._val+other, self._jac)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        from .operators.diagonal_operator import DiagonalOperator
        if isinstance(other, Linearization):
            d1 = DiagonalOperator(self._val)
            d2 = DiagonalOperator(other._val)
            return Linearization(self._val*other._val,
                                 self._jac*d2 + d1*other._jac)
        if isinstance(other, (int, float, complex)):
            #if other == 0:
            #    return ...
            return Linearization(self._val*other, self._jac*other)
        if isinstance(other, (Field, MultiField)):
            d2 = DiagonalOperator(other)
            return Linearization(self._val*other, self._jac*d2)
        raise TypeError

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Linearization(self._val*other, self._jac*other)
        if isinstance(other, (Field, MultiField)):
            d1 = DiagonalOperator(other)
            return Linearization(self._val*other, d1*self._jac)

    @staticmethod
    def make_var(field):
        from .operators.scaling_operator import ScalingOperator
        return Linearization(field, ScalingOperator(1., field.domain))
    @staticmethod
    def make_const(field):
        from .operators.scaling_operator import ScalingOperator
        return Linearization(field, ScalingOperator(0., {}))

class Operator(NiftyMetaBase()):
    """Transforms values living on one domain into values living on another
    domain, and can also provide the Jacobian.
    """

    def __call__(self, x):
        """Returns transformed x

        Parameters
        ----------
        x : Linearization
            input

        Returns
        -------
        Linearization
            output
        """
        raise NotImplementedError
