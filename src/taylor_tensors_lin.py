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

from .diff_tensor_primitive import _TensorPrimitive
from .field import Field
from .multi_field import MultiField
from .sugar import domain_union, full
from .utilities import assertEqual, assertIsinstance

class TensorsLinObject(object):
    def __init__(self, val, lin):
        self._domain = val[0].domain
        self._val = val
        self._lin = lin

    @property
    def val(self):
        return self._val

    @property
    def lin(self):
        return self._lin

    @property
    def domain(self):
        return self._domain

    @staticmethod
    def make_trivial(val, maxorder):
        return TensorsLinObject(val, TensorsTrivialLin(val[0].domain, maxorder))

    def new_chain(self, val, lin):
        if isinstance(self.lin, _TensorsChain):
            lin = self.lin.append(lin)
        elif isinstance(self.lin, TensorsTrivialLin):
            lin = lin
        elif isinstance(lin, TensorsTrivialLin):
            lin = self.lin
        else:
            lin = _TensorsChain((self.lin,lin))
        return TensorsLinObject(val, lin)

class TensorsLin(object):
    def __init__(self, domain, target, maxorder):
        self._domain, self._target = domain, target
        self._maxorder = maxorder

    @property
    def maxorder(self):
        return self._maxorder

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    def _check_input(self, x, mode):
        dom = self._domain if mode else self._target
        assertEqual(len(x), self._maxorder)
        for xx in x:
            assertEqual(dom, xx.domain)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode:
            return self._apply(x)
        else:
            return self._adjoint(x)

    def times(self, x):
        return self.apply(x, True)

    def adjoint(self, x):
        return self.apply(x, False)

    def _apply(self, x):
        raise NotImplementedError

    def _adjoint(self, x):
        raise NotImplementedError

class _TensorsChain(TensorsLin):
    def __init__(self, ops):
        domain = ops[0].domain
        target = ops[-1].target
        super(_TensorsChain,self).__init__(domain, target, ops[-1].maxorder)
        for i in range(len(ops)-1):
            assertEqual(ops[i].target, ops[i+1].domain)
            assertEqual(ops[i].maxorder, self.maxorder)
        self._ops = ops

    def _apply(self, x):
        for op in self._ops:
            x = op.times(x)
        return x

    def _adjoint(self, x):
        for op in reversed(self._ops):
            x = op.adjoint(x)
        return x

    def append(self, op):
        return _TensorsChain(self._ops + (op,))

class TensorsLinearLin(TensorsLin):
    def __init__(self, op, maxorder):
        super(TensorsLinearLin,self).__init__(op.domain, op.target, maxorder)
        self._op = op

    def _apply(self, x):
        return tuple(self._op(xx) for xx in x)

    def _adjoint(self, x):
        return tuple(self._op.adjoint(xx) for xx in x)

class TensorsTrivialLin(TensorsLin):
    def __init__(self, domain, maxorder):
        super(TensorsTrivialLin, self).__init__(domain, domain, maxorder)

    def _apply(self, x):
        return x

    def _adjoint(self, x):
        return x

def _index(p):
    for b in p:
        if 0 in b:
            return len(b)-1
    raise ValueError

class TensorsLayerLin(TensorsLin):
    def __init__(self, tensors, vecs, partitions):
        assertEqual(len(tensors), len(vecs))
        super(TensorsLayerLin, self).__init__(tensors[0].domain,
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

class TensorsSumLin(TensorsLin):
    def __init__(self, ttlinlist, signs):
        target = ttlinlist[0].target
        maxorder = ttlinlist[0].maxorder
        for tt in ttlinlist:
            if not isinstance(tt, TensorsLin):
                raise ValueError
            if isinstance(tt, TensorsSumLin):
                raise ValueError
            assertEqual(target, tt.target)
            assertEqual(maxorder, tt.maxorder)
        from .sugar import domain_union
        domain = domain_union([t.domain for t in ttlinlist])
        super(TensorsSumLin, self).__init__(domain, target, maxorder)
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

class TensorsProdLin(TensorsLin):
    def __init__(self, v1, v2, vl1, vl2):
        assertEqual(vl1.target, vl2.target)
        assertEqual(vl1.maxorder, vl2.maxorder)
        domain = domain_union((vl1.domain, vl2.domain))
        super(TensorsProdLin, self).__init__(domain, vl1.target, vl1.maxorder)
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
