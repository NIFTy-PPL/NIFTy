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
from .tensor import Tensor
from .tensor_lin import _TensorLin, _TensorLinObject
from .tensor_primitive import _TensorPrimitive, DiagonalTensorPrimitive
from ..utilities import assertEqual, assertIsinstance
from ..field import Field
from ..multi_field import MultiField


def _partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    first = collection[0]
    for smaller in _partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        yield [ [ first ] ] + smaller

def _index(p):
    for b in p:
        if 0 in b:
            return len(b)-1
    raise ValueError


class _TensorLayerLin(_TensorLin):
    def __init__(self, tensors, vecs, partitions):
        assertEqual(len(tensors), len(vecs))
        super(_TensorLayerLin, self).__init__(tensors[0].domain,
                                              tensors[0].target,
                                              len(tensors))
        for tt,vv in zip(tensors, vecs):
            assertIsinstance(tt, _TensorPrimitive)
            if not (isinstance(vv, Field) or isinstance(vv, MultiField)):
                raise ValueError
            assertEqual(self.domain, tt.domain)
            assertEqual(self.target, tt.target)
            assertEqual(self.domain, vv.domain)
        self._tensors = tensors
        self._vecs = vecs
        self._ppts = partitions

    def _apply(self, x):
        res = []
        for partitions in self._ppts:
            rest = 0.
            for p in partitions:
                v = tuple(x[len(b)-1] if 0 in b else self._vecs[len(b)-1] for b in p)
                rest = rest + self._tensors[len(p)-1].getVec(v)
            res.append(rest)
        return tuple(res)

    def _adjoint(self, x):
        res = [0.,]*self.maxorder
        for i,partitions in enumerate(self._ppts):
            rest = x[i]
            for p in partitions:
                vv = tuple(self._vecs[len(b)-1] for b in p if 0 not in b)
                ind = _index(p)
                res[ind] = res[ind] + self._tensors[len(p)-1].getVecAdjoint(rest, vv)
        return tuple(res)


class TensorLayer(Tensor):
    def __init__(self, tensors):
        self._domain = tensors[0].domain
        self._target = tensors[0].target
        for tt in tensors[1:]:
            assertEqual(self.target, tt.target)
            assertEqual(self.domain, tt.domain)
        self._maxorder = len(tensors)
        self._tensors = tensors
        self._ppts = tuple(list(_partition(list(np.arange(i+1))))
                            for i in range(self.maxorder))

    def _contract(self, inp):
        islin = isinstance(inp, _TensorLinObject)
        inpv = inp.val if islin else inp
        res = []
        for partitions in self._ppts:
            rest = 0.
            for p in partitions:
                v = tuple(inpv[len(b)-1] for b in p)
                rest = rest + self._tensors[len(p)-1].getVec(v)
            res.append(rest)
        res = tuple(res)
        if not islin:
            return res
        return inp.new_chain(res, _TensorLayerLin(self._tensors, inpv, self._ppts))

    @staticmethod
    def make_diagonal(vecs):
        tensors = tuple(DiagonalTensorPrimitive(vec,i+1) for i,vec in enumerate(vecs))
        return TensorLayer(tensors)
