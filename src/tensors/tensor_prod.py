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

from scipy.special import binom
from .tensor import Tensor
from .tensor_lin import _TensorLin, _TensorLinObject
from ..sugar import domain_union
from ..utilities import assertEqual, assertIsinstance


#TODO: Simplification of Lin should be possible
class _TensorProdLin(_TensorLin):
    def __init__(self, v1, v2, vl1, vl2):
        assertEqual(vl1.target, vl2.target)
        assertEqual(vl1.maxorder, vl2.maxorder)
        domain = domain_union((vl1.domain, vl2.domain))
        super(_TensorProdLin, self).__init__(domain, vl1.target, vl1.maxorder)
        self._vl1, self._vl2 = vl1, vl2
        self._v1, self._v2 = v1, v2

    def _apply(self, x):
        v1 = tuple(xx.extract(self._vl1.domain) for xx in x)
        v2 = tuple(xx.extract(self._vl2.domain) for xx in x)
        vl1 = (0., ) + self._vl1.times(v1)
        vl2 = (0., ) + self._vl2.times(v2)
        res = ()
        for n in range(1, self.maxorder+1):
            rr = 0.
            for i in range(1<<n):
                xx1,xx2 = 0,0
                for j in range(n-1):
                    if (i & (1<<j)):
                        xx1 += 1
                    else:
                        xx2 += 1
                if i & (1<<(n-1)):
                    tmp = self._v2[xx2]*vl1[xx1+1]
                else:
                    tmp = self._v1[xx1]*vl2[xx2+1]
                rr = rr + tmp
            res += (rr,)
        return res

    def _adjoint(self, x):
        vl1 = [0.,]*(self.maxorder+1)
        vl2 = [0.,]*(self.maxorder+1)
        for n in range(1, self.maxorder+1):
            rr = x[n-1]
            for i in range(1<<n):
                xx1,xx2 = 0,0
                for j in range(n-1):
                    if (i & (1<<j)):
                        xx1 += 1
                    else:
                        xx2 += 1
                if i & (1<<(n-1)):
                    vl1[xx1+1] = vl1[xx1+1] + self._v2[xx2].conjugate()*rr
                else:
                    vl2[xx2+1] = vl2[xx2+1] + self._v1[xx1].conjugate()*rr
        vl1 = self._vl1.adjoint(vl1[1:])
        vl2 = self._vl2.adjoint(vl2[1:])
        return tuple(a.flexible_addsub(b, False) for a,b in zip(vl1, vl2))


class TensorProd(Tensor):
    def __init__(self, val1, val2, tt1, tt2):
        assertIsinstance(tt1, Tensor)
        assertIsinstance(tt2, Tensor)
        assertEqual(tt1.target, tt2.target)
        assertEqual(val1.domain, tt2.target)
        assertEqual(val2.domain, tt2.target)
        assertEqual(tt1.maxorder, tt2.maxorder)
        self._domain = domain_union((tt1.domain, tt2.domain))
        self._target = val1.domain
        self._maxorder = tt1.maxorder
        self._tt1, self._tt2 = tt1, tt2
        self._val1, self._val2 = val1, val2

    def _contract(self, x):
        islin = isinstance(x, _TensorLinObject)
        xv = x.val if islin else x
        v1 = tuple(xx.extract(self._tt1.domain) for xx in xv)
        v2 = tuple(xx.extract(self._tt2.domain) for xx in xv)
        v1 = _TensorLinObject.make_trivial(v1, self.maxorder) if islin else v1
        v2 = _TensorLinObject.make_trivial(v2, self.maxorder) if islin else v2
        v1l = self._tt1.contract(v1)
        v2l = self._tt2.contract(v2)
        v1 = v1l.val if islin else v1l
        v2 = v2l.val if islin else v2l
        v1 = (self._val1,) + v1
        v2 = (self._val2,) + v2
        res = ()
        for n in range(1,self.maxorder+1):
            rr = 0.
            for k in range(n+1):
                rr = rr + binom(n,k)*v1[n-k]*v2[k]
            res += (rr,)
        if not islin:
            return res
        return x.new_chain(res, _TensorProdLin(v1, v2, v1l.lin, v2l.lin))
