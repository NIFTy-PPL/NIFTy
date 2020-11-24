# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from .operators.operator import Operator
from .operators.scaling_operator import ScalingOperator
from .taylor_tensors import (TensorsLayer, TensorsChain, TensorsProd,
                             TensorsLinear, TensorsTrivial,
                             TaylorTensors)
from .sugar import makeOp
from .field import Field
from .multi_field import MultiField
from .utilities import assertEqual

class Taylor(Operator):
    """Class describing a Taylor approximation up to a given order

    This class is a generalization of Fields/MultiFields (which are Taylor
    objects with maximum order 0), and Linearizations (which are Taylor objects
    with maximum order 1).
    Eventually, Taylor objects will most likely also get a "metric" and
    "want_metric" members.

    Parameters
    ----------
    val: Field or MultiField
        The value of the zeroth order approximation
    tensors: TaylorTensors
        The tensors for higher order approximations.
    """
    def __init__(self, val, tensors):
        if not (isinstance(val, Field) or isinstance(val, MultiField)):
            raise ValueError
        if not isinstance(tensors, TaylorTensors):
            raise ValueError
        assertEqual(val.domain, tensors.target)
        self._val = val
        self._tensors = tensors

    @property
    def isTrivial(self):
        return isinstance(self.tensors, TensorsTrivial)

    @property
    def domain(self):
        return self._tensors.domain

    @property
    def target(self):
        return self._tensors.target

    @property
    def val(self):
        return self._val

    @property
    def tensors(self):
        return self._tensors

    @property
    def maxorder(self):
        return self._tensors.maxorder

    def __neg__(self):
        return self.new_from_lin(ScalingOperator(self.target,-1.)).prepend(self)

    def __add__(self, other):
        if isinstance(other, Taylor):
            assertEqual(self.maxorder, other.maxorder)
            assertEqual(self.target, other.target)
            return self.new(self.val+other.val, self.tensors + other.tensors)
        if isinstance(other, Field) or isinstance(other, MultiField):
            assertEqual(self.target, other.domain)
            return self.new(self.val+other, self.tensors)
        raise TypeError

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Taylor):
            assertEqual(self.maxorder, other.maxorder)
            assertEqual(self.target, other.target)
            return self.new(self.val-other.val, self.tensors - other.tensors)
        if isinstance(other, Field) or isinstance(other, MultiField):
            assertEqual(self.target, other.domain)
            return self.new(self.val-other, self.tensors)
        raise TypeError

    def __rsub__(self, other):
        if isinstance(other, Taylor):
            assertEqual(self.maxorder, other.maxorder)
            assertEqual(self.target, other.target)
            return self.new(other.val-self.val, other.tensors-self.tensors)
        if isinstance(other, Field) or isinstance(other, MultiField):
            assertEqual(self.target, other.domain)
            t = self.__neg__()
            return t.new(t.val+other.val, t.tensors)
        raise TypeError

    def __truediv__(self, other):
        if np.isscalar(other):
            return self.__mul__(1/other)
        return self.__mul__(other.ptw("reciprocal"))

    def __rtruediv__(self, other):
        return self.ptw("reciprocal").__mul__(other)

    def __pow__(self, power):
        if not np.isscalar(power):
            raise NotImplementedError
        return self.ptw('power', power)

    def __mul__(self, other):
        if np.isscalar(other):
            return self.new_from_lin(ScalingOperator(self.target,other)).prepend(self)
        elif isinstance(other, Taylor):
            assertEqual(self.maxorder, other.maxorder)
            assertEqual(self.target, other.target)
            return self.new_from_prod(other)
        elif isinstance(other, Field) or isinstance(other, MultiField):
            assertEqual(self.target, other.domain)
            return self.new_from_lin(makeOp(other)).prepend(self)
        else:
            raise TypeError

    def __rmul__(self, other):
        return self.__mul__(other)

    def new_from_lin(self, op):
        val = op(self.val)
        return self.new(val, TensorsLinear(op, self.maxorder))

    def new_from_prod(self, t2):
        tensors = TensorsProd(self.val, t2.val, self.tensors, t2.tensors)
        return self.new(self.val*t2.val, tensors)

    def prepend(self, old):
        if isinstance(old.tensors, TensorsTrivial):
            tensors = self.tensors
        elif isinstance(old.tensors, TensorsChain):
            tensors = old.tensors.append(self.tensors)
        elif isinstance(self.tensors, TensorsChain):
            tensors = TensorsChain((old.tensors,)+self.tensors._layers)
        else:
            tensors = TensorsChain((old.tensors, self.tensors))
        return self.new(self.val, tensors)

    def new(self, val, tensors):
        if not isinstance(tensors, TaylorTensors):
            raise ValueError
        return Taylor(val, tensors)

    def ptw(self, op, *args, **kwargs):
        tmp = self.val.ptw_with_derivs(op, self.maxorder, *args, **kwargs)
        tensors = TensorsLayer.make_diagonal(tmp[1:])
        return self.new(tmp[0], tensors).prepend(self)

    def trivial_derivatives(self):
        return Taylor.make_var(self.val, self.maxorder)

    @staticmethod
    def make_var(val, maxorder):
        # Currently maxorder 0 is defined via Field
        # Taylor objects are strictly incopatible with Linearizations
        # Will be unified in the future
        if maxorder<1:
            raise ValueError
        return Taylor(val, TensorsTrivial(val.domain, maxorder))