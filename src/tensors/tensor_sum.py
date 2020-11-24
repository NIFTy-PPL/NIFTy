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

from .tensor import Tensor
from .tensor_lin import _TensorLin, _TensorLinObject
from .tensor_layer import TensorLayer
from ..sugar import full, domain_union
from ..utilities import assertEqual, assertIsinstance


class _TensorSumLin(_TensorLin):
    def __init__(self, ttlinlist, signs):
        target = ttlinlist[0].target
        maxorder = ttlinlist[0].maxorder
        for tt in ttlinlist:
            if not isinstance(tt, _TensorLin):
                raise ValueError
            if isinstance(tt, _TensorSumLin):
                raise ValueError
            assertEqual(target, tt.target)
            assertEqual(maxorder, tt.maxorder)
        #from .sugar import domain_union
        domain = domain_union([t.domain for t in ttlinlist])
        super(_TensorSumLin, self).__init__(domain, target, maxorder)
        self._ttlist = ttlinlist
        self._signs = signs

    def _apply(self, x):
        res = [0.,]*self.maxorder
        for tt,sign in zip(self._ttlist, self._signs):
            v = tuple(xx.extract(tt.domain) for xx in x)
            t = tt.times(v)
            for i, ti in enumerate(t):
                res[i] = res[i] + ti if sign else res[i] - ti
        return tuple(res)

    def _adjoint(self, x):
        res = [full(self.domain, 0.),]*self.maxorder
        for tt,sign in zip(self._ttlist, self._signs):
            t = tuple(xx if sign else -xx for xx in x)
            v = tt.adjoint(t)
            for i,vv in enumerate(v):
                res[i] = res[i].flexible_addsub(vv, False)
        return tuple(res)


class TensorSum(Tensor):
    def __init__(self, ttlist, signs):
        self._target = ttlist[0].target
        self._maxorder = ttlist[0].maxorder
        for tt in ttlist:
            if not isinstance(tt, Tensor):
                raise ValueError
            if isinstance(tt, TensorSum):
                raise ValueError
            assertEqual(self._target, tt.target)
            assertEqual(self.maxorder, tt.maxorder)
        #from .sugar import domain_union
        self._domain = domain_union([t.domain for t in ttlist])
        self._ttlist = ttlist
        self._signs = signs

    def _contract(self, x):
        islin = isinstance(x, _TensorLinObject)
        xv = x.val if islin else x
        res = [0.,]*self.maxorder
        lins = []
        for tt,sign in zip(self._ttlist, self._signs):
            v = tuple(xx.extract(tt.domain) for xx in xv)
            v = _TensorLinObject.make_trivial(v, self.maxorder) if islin else v
            tl = tt.contract(v)
            t = tl.val if islin else tl
            for i, ti in enumerate(t):
                res[i] = res[i] + ti if sign else res[i] - ti
            if islin:
                lins.append(tl.lin)
        res = tuple(res)
        if not islin:
            return res
        return x.new_chain(res, _TensorSumLin(lins, self._signs))

    def append(self, other, sign):
        assertIsinstance(other, TensorLayer)
        assertEqual(self._target, other.target)
        return TensorSum(self._ttlist+(other,), self._signs+(sign,))

    def join(self, other, sign):
        assertIsinstance(other, TensorSum)
        assertEqual(self._target, other.target)
        sgn = tuple(s if sign else not s for s in other._signs)
        return TensorSum(self._ttlist+other._ttlist, self._signs+sgn)

    def neg(self):
        return TensorSum(self._ttlist, tuple(not s for s in self._signs))
