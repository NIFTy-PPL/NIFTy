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
from .linearization import Linearization
from .operators.scaling_operator import ScalingOperator
from .operators.operator import Operator
from .diff_tensor import (DiffTensor, assertIsinstance, assertEqual,
                          assertTrue, assertIdentical)
from .sugar import makeOp
from .field import Field
from .multi_field import MultiField


class Taylor(Operator):
    """Class describing a Taylor approximation up to a given order

    This class is a generalization of Fields/MultiFields (which are Taylor
    objects with maximum order 0), and Linearizations (which are Taylor objects
    with maximum order 1).
    Eventually, Taylor objects will most likely also get a "metric" and
    "want_metric" members.

    Parameters
    ----------
    tensors: tuple of DiffTensors
        The approximations at different orders
        - tensors[i].rank must be i+1
        - the domain of all tensors must be identical
        - the target of all tensors must be identical
    """
    def __init__(self, tensors):
        assertTrue(len(tensors)>=1)
        for i, t in enumerate(tensors):
            assertIsinstance(t, DiffTensor)
            assertEqual(i+1, t.rank)
            assertIdentical(t.domain, tensors[0].domain)
            assertIdentical(t.target, tensors[0].target)
        self._tensors = tensors

    @property
    def domain(self):
        return self._tensors[0].domain

    @property
    def target(self):
        return self._tensors[0].target

    @property
    def val(self):
        return self._tensors[0].vec

    @property
    def jac(self):
        return self._tensors[1].linop

    @property
    def tensors(self):
        return self._tensors

    @property
    def maxorder(self):
        """int: the maximum approximation order, i.e. one less than the number of stored tensors"""
        return len(self._tensors)-1

    def __neg__(self):
        return self.new_from_lin(ScalingOperator(self.target,-1.)).prepend(self)

    def __len__(self):
        return len(self._tensors)

    def __getitem__(self, i):
        return self._tensors[i]

    def __add__(self, other):
        if isinstance(other, Taylor):
            assertEqual(self.maxorder, other.maxorder)
            assertEqual(self.target, other.target)
            return self.new(tuple(aa+bb for aa,bb in zip(self.tensors,other.tensors)))
        if isinstance(other, Linearization):
            raise ValueError
        vn = (DiffTensor.makeVec(self.val+other, self.domain),)
        return self.new(vn+self.tensors[1:])
        

    def __radd__(self, other):
        if isinstance(other, Taylor):
            return self.new(tuple(aa+bb for aa,bb in zip(self.tensors,other.tensors)))
        if isinstance(other, Linearization):
            raise ValueError
        vn = (DiffTensor.makeVec(self.val+other, self.domain),)
        return self.new(vn+self.tensors[1:])

    def __sub__(self, other):
        if isinstance(other, Taylor):
            return self.new(tuple(aa-bb for aa,bb in zip(self.tensors,other.tensors)))
        if isinstance(other, Linearization):
            raise ValueError
        vn = (DiffTensor.makeVec(self.val-other, self.domain),)
        return self.new(vn+self.tensors[1:])

    def __rsub__(self, other):
        if isinstance(other, Taylor):
            return self.new(tuple(bb-aa for aa,bb in zip(self.tensors,other.tensors)))
        if isinstance(other, Linearization):
            raise ValueError
        return self.new_from_lin(ScalingOperator(self.target,-1.)).prepend(self)+other

    def __truediv__(self, other):
        if np.isscalar(other):
            return self.__mul__(1/other)
        return self.__mul__(other.ptw("reciprocal"))

    def __rtruediv__(self, other):
        return self.ptw("reciprocal").__mul__(other)

    def __pow__(self, power):
        if not np.isscalar(power):
            return NotImplemented
        return self.ptw('power', power)

    def __mul__(self, other):
        if np.isscalar(other):
            return self.new_from_lin(ScalingOperator(self.target,other)).prepend(self)
        elif isinstance(other, Taylor):
            return self.new_from_prod(other)
        elif isinstance(other, Field) or isinstance(other, MultiField):
            if other.domain != self.target:
                raise ValueError
            return self._new_from_lin(makeOp(other)).prepend(self)
        else:
            raise TypeError

    def __rmul__(self, other):
        return self.__mul__(other)

    def trivial_derivatives(self):
        return self.make_var(self.val, self.maxorder)

    def new_from_lin(self, op):
        tensors = (DiffTensor.makeVec(op(self.val), domain=op.domain),
                   DiffTensor.makeLinear(op))
        tensors += tuple(DiffTensor.makeNull(op.domain,op.target,i+1)
                    for i in range(2,self.maxorder+1))
        return self.new(tensors)

    def new_from_prod(self, t2):
        assertTrue(self.maxorder == t2.maxorder)
        return self.new(tuple(DiffTensor.makeGenLeibniz(self, t2, i) for i in range(self.maxorder+1)))

    def prepend(self, old):
        tensors = (DiffTensor.makeVec(self.val, domain=old.domain), )
        tensors += tuple(DiffTensor.makeComposed(self, old, i) for i in range(1,self.maxorder+1))
        return self.new(tensors)

    def new(self, tensors):
        return Taylor(tensors)

    def ptw(self, op, *args, **kwargs):
        tmp = self.val.ptw_with_derivs(op, self.maxorder, *args, **kwargs)
        tensors = (DiffTensor.makeVec(tmp[0],domain=self.domain), )
        tensors += tuple(DiffTensor.makeDiagonal(tm, i+2) for i,tm in enumerate(tmp[1:]))
        return self.new(tensors).prepend(self)

    @staticmethod
    def make_var(val, maxorder):
        # Currently maxorder 0 and 1 are defined via Field and Linearization
        # Will be unified in the future
        if maxorder < 2:
            raise ValueError
        tensors = (DiffTensor.makeVec(val),)
        if maxorder>0:
            tensors += (DiffTensor.makeLinear(ScalingOperator(val.domain, 1.)), )
            tensors += tuple(DiffTensor.makeNull(val.domain, val.domain, i+1)
                        for i in range(2,maxorder+1))
        return Taylor(tensors)