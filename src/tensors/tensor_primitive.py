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

    def getVec(self, x):
        raise NotImplementedError

    def getVecAdjoint(self, y, x):
        raise NotImplementedError


class DiagonalTensorPrimitive(_TensorPrimitive):
    def __init__(self, vec, order):
        super(DiagonalTensorPrimitive, self).__init__(vec.domain, vec.domain, order)
        self._vec = vec

    def getVecAdjoint(self, y, x):
        res = self._vec
        if len(x) != 0:
            res = reduce(lambda a,b:a*b, x)*res
        return res.conjugate()*y

    def getVec(self, x):
        return reduce(lambda a,b:a*b, x)*self._vec
