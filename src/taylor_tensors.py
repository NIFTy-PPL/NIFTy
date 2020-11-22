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
"""
Some general remarks on the special type of tensors we need for higher order
taylor expansions (We ignore the notion of dual spaces here):

Consider a function f that maps from space A to space B:

The first derivative is a Linear operator that maps from A to B, it can be
denoted as a 2nd order tensor of the form J_{i,j} where i labels the
coordinates of A and j the coords of B. The comma "," indicates derivatives,
i.E. everything to the right of "," is an index arising from taking a
derivative.

A higher order derivative can be denoted in the same manner, e.g. J_{i,jk}
denotes the second derivative of f and is a 3rd order tensor.
It is a Multilinear operator that maps from AxA to B, i.E. it takes two vectors
in A and maps to a vector in B. Higher orders are constructed accordingly.

Some important properties:
All input vectors have to live on the same space A. In order to evaluate a
Taylor expansion, aditionally all input vectors are the same.

The differential tensors are symmetric w.r.t. the latter indices, i.E. the 
indices produced by the derivatives. Therefore
    J_{i,jk} a_j b_k = J_{i,kj} b_k a_j
where we sum over repeated indices. This means that the order in which we
contract does not matter. As a consequence partial contraction leads to a
unique new tensor irrespective of how we contracted:
    B_{ik} := J_{i,jk} a_j = J_{i,kj} a_j
This leads to the special case where we can form a new Linear operator from a
nth order diff tensor by contracting with n-1 vectors. This new
operator then has the same capabilities as usual Linear operators.
"""
import numpy as np
from scipy.special import binom
from .diff_tensor_primitive import LinearTensor, NullTensor, DiagonalTensor
from .operators.scaling_operator import ScalingOperator


class TaylorTensors(object):
    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def maxorder(self):
        return self._maxorder

    def __add__(self, other):
        #TODO checks
        if isinstance(self, TensorsSum):
            if isinstance(other, TensorsSum):
                return self.join(other, True)
            return self.append(other, True)
        if isinstance(other, TensorsSum):
            return other.append(self, True)
        return TensorsSum((self, other), (True, True))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(self, TensorsSum):
            if isinstance(other, TensorsSum):
                return self.join(other, False)
            return self.append(other, False)
        if isinstance(other, TensorsSum):
            return other.neg.append(self, True)
        return TensorsSum((self, other), (True, False))

    def __rsub__(self, other):
        if isinstance(self, TensorsSum):
            if isinstance(other, TensorsSum):
                return other.join(self, False)
            return self.neg.append(other, True)
        if isinstance(other, TensorsSum):
            return other.append(self, False)
        return TensorsSum((self, other), (False, True))

    def _check_input(self, x):
        if x.domain != self.domain:
            raise ValueError

    def getVecs(self, x):
        self._check_input(x)
        return self._getVecs(x)

    def getLin(self, x):
        self._check_input(x)
        return self._getLin(x)

    def _getVecs(self, x):
        raise NotImplementedError

    def _getLin(self, x):
        raise NotImplementedError


class TensorsLayer(TaylorTensors):
    def __init__(self, tensors):
        #TODO checks
        self._domain = tensors[0].domain
        self._target = tensors[0].target
        self._maxorder = len(tensors)
        self._tensors = tensors

    @property
    def tensors(self):
        return self._tensors

    def _getVecs(self, x):
        return tuple((tt.getVec(x=(x,)*(i+1)) for i,tt in enumerate(self._tensors)))

    def _getLin(self, x):
        return tuple((tt.getLinOp(x=(x,)*i) for i,tt in enumerate(self._tensors)))

    @staticmethod
    def make_trivial(domain, maxorder):
        return TensorsLayer.make_linear(ScalingOperator(domain,1.), maxorder)

    @staticmethod
    def make_diagonal(vecs):
        tensors = tuple(DiagonalTensor(vec,i+1) for i,vec in enumerate(vecs))
        return TensorsLayer(tensors)

    @staticmethod
    def make_linear(op, maxorder):
        tensors = (LinearTensor(op),)
        tensors += tuple(NullTensor(op.domain,op.target,o) for o in range(2,maxorder+1))
        return TensorsLayer(tensors)
        

class TensorsSum(TaylorTensors):
    def __init__(self, ttlist, signs):
        #TODO checks
        from .sugar import domain_union
        self._domain = domain_union((t.domain for t in ttlist))
        self._target = ttlist[0].target
        self._maxorder = ttlist[0].maxorder
        self._ttlist = ttlist
        self._signs = signs

    def _getVecs(self, x):
        res = (0.,)*len(self.maxorder)
        for tt,sign in zip(self._ttlist, self._signs):
            t = tt.getVecs(x)
            for r,ti in zip(res,t):
                r = r + ti if sign else r - ti
        return res

    def append(self, other, sign):
        #TODO checks
        return TensorsSum(self._ttlist+(other,), self._signs+(sign,))

    def join(self, other, sign):
        #TODO checks
        sgn = tuple(s if sign else not s for s in other._signs)
        return TensorsSum(self._ttlist+other._ttlist, self._signs+sgn)

    def neg(self):
        return TensorsSum(self._ttlist, tuple(not s for s in self._signs))

class TensorsProd(TaylorTensors):
    def __init__(self, val1, val2, tt1, tt2):
        #TODO checks
        from .sugar import domain_union
        self._domain = domain_union((tt1.domain, tt2.domain))
        self._target = val1.domain
        self._maxorder = tt1.maxorder
        self._tt1, self._tt2 = tt1, tt2
        self._val1, self._val2 = val1, val2

    def _getVecs(self, x):
        v1 = self._tt1.getVecs(x.extract(self._tt1.domain))
        v2 = self._tt2.getVecs(x.extract(self._tt2.domain))
        v1 = (self._val1,)+v1
        v2 = (self._val2,)+v2
        res = ()
        for n in range(1,self.maxorder):
            rr = 0.
            for k in range(n+1):
                rr = rr + binom(n,k)*v1[n-k]*v2[k]
            res += (rr,)
        return res

def _partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in _partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

def _all_partitions_nontrivial(n, new):
    pps = list(_partition(list(np.arange(n))))
    i,nm = 0,len(pps)
    while i<nm:
        if new[len(pps[i])-1].isNullTensor:
            del pps[i]
            nm -=1
        else:
            i+=1
    return pps

def _contract(layer, inp):
    res = ()
    for i in range(layer.maxorder):
        partitions = _all_partitions_nontrivial(i+1, layer.tensors)
        rest = 0.
        for p in partitions:
            rest = rest + layer.tensors[len(p)-1].getVec((inp[len(b)-1] for b in p))
        res += (rest, )
    return res

class TensorsChain(TaylorTensors):
    def __init__(self, layers):
        #TODO checks
        self._domain = layers[0].domain
        self._target = layers[-1].target
        self._maxorder = layers[0].maxorder
        self._layers = layers

    def append(self, tensors):
        #TODO checks
        return TensorsChain(self._layers + (tensors,))

    def _getVecs(self, x):
        r = self._layers[0].getVecs(x)
        for ll in self._layers[1:]:
            r = _contract(ll, r)
        return r

if __name__ == '__main__':
    import nifty7 as ift
    import timeit
    sp = ift.RGSpace(32)
    n=3
    derivs = tuple(ift.from_random(sp) for i in range(n))
    tts = tuple(ift.DiffTensor.makeDiagonal(derivs[i], i+2) for i in range(n))
    derivs2 = tuple(ift.from_random(sp) for i in range(n))
    tts2 = tuple(ift.DiffTensor.makeDiagonal(derivs2[i], i+2) for i in range(n))
    t1 = TensorsLayer.make_diagonal(tts)
    t2 = TensorsLayer(tts2)
    linop = ift.HarmonicTransformOperator(sp.get_default_codomain()).adjoint
    tts3 = (ift.DiffTensor.makeLinear(linop),) + tuple(ift.DiffTensor.makeNull(linop.domain,linop.target,i+3) for i in range(n-1))
    t3 = TensorsLayer(tts3)
    
    tv1 = ift.DiffTensor.makeVec(ift.from_random(t1.target))
    tl1 = ift.Taylor((tv1,)+tts)
    tv2 = ift.DiffTensor.makeVec(ift.from_random(t2.target))
    tl2 = ift.Taylor((tv2,)+tts2)
    tv3 = ift.DiffTensor.makeVec(ift.from_random(t3.target))
    tl3 = ift.Taylor((tv3,)+tts3)
    

    ch = TensorsChain([t1,t2,t3])
    
    x = ift.from_random(ch.domain)
    t0 = timeit.default_timer()
    res = ch.getVecs(x)
    t1 = timeit.default_timer()
    print(t1-t0)

    d1 = linop(derivs2[0]*derivs[0]*x)
    ift.extra.assert_allclose(d1,res[0])
    d2 = linop(derivs2[1]*(derivs[0]*x)**2 + derivs2[0]*derivs[1]*x**2)
    ift.extra.assert_allclose(d2,res[1])
    d3 = linop(derivs2[2]*(derivs[0]*x)**3+derivs2[1]*(3.*derivs[1]*x**2*derivs[0]*x)+derivs2[0]*derivs[2]*x**3)
    ift.extra.assert_allclose(d3,res[2])
    