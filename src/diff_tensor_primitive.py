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

from functools import reduce
from .sugar import makeOp, full
from .operators.simple_linear_operators import NullOperator

class _TensorPrimitive(object):
    def __init__(self, domain, target, order):
        self._domain = domain
        self._target = target
        self._order = order

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def order(self):
        return self._order

    def getVec(self, x=()):
        raise NotImplementedError

    def getLinop(self, x=()):
        raise NotImplementedError


class LinearTensor(_TensorPrimitive):
    def __init__(self, op):
        super(LinearTensor, self).__init__(op.domain, op.target, 1)
        self._op = op

    def getLinop(self, x=()):
        if len(x) != 0:
            raise ValueError
        return self._op

    def getVec(self, x=()):
        if len(x) != 1:
            raise ValueError
        return self._op(x[0])


class DiagonalTensor(_TensorPrimitive):
    def __init__(self, vec, order):
        super(DiagonalTensor, self).__init__(vec.domain, vec.domain, order)
        self._vec = vec

    def _helper(self, x, rnk):
        if self._rank-len(x) != rnk:
            raise ValueError
        res = self._vec
        if len(x) != 0:
            res = reduce(lambda a,b:a*b, x)*res
        return res if rnk == 1 else makeOp(res)

    def getLinop(self, x=()):
        return self._helper(x, 2)

    def getVec(self, x=()):
        return self._helper(x, 1)


class NullTensor(_TensorPrimitive):
    def __init__(self, domain, target, order):
        super(NullTensor, self).__init__(domain, target, order)
        self._op, self._vec = NullOperator(self.domain, self.target), full(self.target, 0.)

    def getLinop(self, x=()):
        return self._op

    def getVec(self, x=()):
        return self._vec