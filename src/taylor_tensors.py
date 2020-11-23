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
from .sugar import full

def assertIsinstance(t1, t2):
    if not isinstance(t1, t2):
        print("Error: {} is not an instance of {}".format(t1, t2))
        raise TypeError("type mismatch")


def assertIdentical(t1, t2):
    if t1 is not t2:
        raise ValueError("objects are not identical")


def assertEqual(t1, t2):
    if t1 != t2:
        print("Error: {} is not equal to {}".format(t1, t2))
        raise ValueError("objects are not equal")

def assertTrue(t1):
    if not t1:
        raise ValueError("assertion failed")

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
        assertEqual(self.target, other.target)
        assertEqual(self.maxorder, other.maxorder)
        assertIsinstance(other, TaylorTensors)
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
        assertEqual(self.target, other.target)
        assertEqual(self.maxorder, other.maxorder)
        assertIsinstance(other, TaylorTensors)
        if isinstance(self, TensorsSum):
            if isinstance(other, TensorsSum):
                return self.join(other, False)
            return self.append(other, False)
        if isinstance(other, TensorsSum):
            return other.neg.append(self, True)
        return TensorsSum((self, other), (True, False))

    def __rsub__(self, other):
        assertEqual(self.target, other.target)
        assertEqual(self.maxorder, other.maxorder)
        assertIsinstance(other, TaylorTensors)
        if isinstance(self, TensorsSum):
            if isinstance(other, TensorsSum):
                return other.join(self, False)
            return self.neg.append(other, True)
        if isinstance(other, TensorsSum):
            return other.append(self, False)
        return TensorsSum((self, other), (False, True))

    def _check_input(self, x):
        for xx in x:
            if xx.domain != self.domain:
                raise ValueError

    def getVecs(self, x):
        r = (x,)+(full(self.domain,0.),)*(self.maxorder-1)
        return self.contract(r)

    def getLin(self, x):
        raise NotImplementedError

    def contract(self, x):
        self._check_input(x)
        return self._contract(x)

    def _contract(self, x):
        raise NotImplementedError


    @property
    def isTrivial(self):
        if not isinstance(self, TensorsLayer):
            return False
        if not isinstance(self.tensors[0], LinearTensor):
            return False
        if not isinstance(self.tensors[0]._op, ScalingOperator):
            return False
        if not self.tensors[0]._op._factor == 1:
            return False
        for t in self.tensors[1:]:
            if not isinstance(t, NullTensor):
                return False
        return True

def _partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in _partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        yield [ [ first ] ] + smaller

def _all_partitions_nontrivial(n, new):
    pps = list(_partition(list(np.arange(n))))
    i,nm = 0,len(pps)
    while i<nm:
        if isinstance(new[len(pps[i])-1], NullTensor):
            del pps[i]
            nm -=1
        else:
            i+=1
    return pps


class TensorsLayer(TaylorTensors):
    def __init__(self, tensors):
        self._domain = tensors[0].domain
        self._target = tensors[0].target
        for tt in tensors[1:]:
            assertEqual(self.target, tt.target)
            assertEqual(self.domain, tt.domain)
        self._maxorder = len(tensors)
        self._tensors = tensors
        self._ppts = tuple(_all_partitions_nontrivial(i+1, self.tensors)
                            for i in range(self.maxorder))

    @property
    def tensors(self):
        return self._tensors

    def _contract(self, inp):
        res = []
        for partitions in self._ppts:
            rest = 0.
            for p in partitions:
                v = tuple(inp[len(b)-1] for b in p)
                rest = rest + self.tensors[len(p)-1].getVec(v)
            res.append(rest)
        return tuple(res)

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
        self._target = ttlist[0].target
        for tt in ttlist:
            if not isinstance(tt, TaylorTensors):
                raise ValueError
            if isinstance(tt, TensorsSum):
                raise ValueError
            assertEqual(self._target, tt.target)
        from .sugar import domain_union
        self._domain = domain_union([t.domain for t in ttlist])
        self._maxorder = ttlist[0].maxorder
        self._ttlist = ttlist
        self._signs = signs

    def _contract(self, x):
        res = [0.,]*self.maxorder
        for tt,sign in zip(self._ttlist, self._signs):
            v = tuple(xx.extract(tt.domain) for xx in x)
            t = tt.contract(v)
            for i, ti in enumerate(t):
                res[i] = res[i] + ti if sign else res[i] - ti
        return tuple(res)

    def append(self, other, sign):
        assertIsinstance(other, TensorsLayer)
        assertEqual(self._target, other.target)
        return TensorsSum(self._ttlist+(other,), self._signs+(sign,))

    def join(self, other, sign):
        assertIsinstance(other, TensorsSum)
        assertEqual(self._target, other.target)
        sgn = tuple(s if sign else not s for s in other._signs)
        return TensorsSum(self._ttlist+other._ttlist, self._signs+sgn)

    def neg(self):
        return TensorsSum(self._ttlist, tuple(not s for s in self._signs))

class TensorsProd(TaylorTensors):
    def __init__(self, val1, val2, tt1, tt2):
        assertIsinstance(tt1, TaylorTensors)
        assertIsinstance(tt2, TaylorTensors)
        assertEqual(tt1.target, tt2.target)
        assertEqual(val1.domain, tt2.target)
        assertEqual(val2.domain, tt2.target)
        from .sugar import domain_union
        self._domain = domain_union((tt1.domain, tt2.domain))
        self._target = val1.domain
        self._maxorder = tt1.maxorder
        self._tt1, self._tt2 = tt1, tt2
        self._val1, self._val2 = val1, val2

    def _contract(self, x):
        v1 = tuple(xx.extract(self._tt1.domain) for xx in x)
        v2 = tuple(xx.extract(self._tt2.domain) for xx in x)
        v1 = (self._val1,) + self._tt1.contract(v1)
        v2 = (self._val2,) + self._tt2.contract(v2)
        res = ()
        for n in range(1,self.maxorder+1):
            rr = 0.
            for k in range(n+1):
                rr = rr + binom(n,k)*v1[n-k]*v2[k]
            res += (rr,)
        return res


class TensorsChain(TaylorTensors):
    def __init__(self, layers):
        mylayers = []
        for l in layers:
            if not isinstance(l, TaylorTensors):
                raise ValueError
            if isinstance(l, TensorsChain):
                raise ValueError
            if not l.isTrivial:
                mylayers.append(l)
        self._domain = mylayers[0].domain
        self._target = mylayers[-1].target
        self._maxorder = mylayers[0].maxorder
        self._layers = tuple(mylayers)

    def append(self, tensors):
        assertEqual(self.target, tensors.domain)
        if isinstance(tensors, TensorsChain):
            return TensorsChain(self._layers + tensors._layers)
        elif not isinstance(tensors, TaylorTensors):
            raise ValueError
        return TensorsChain(self._layers + (tensors,))

    def _contract(self, x):
        for ll in self._layers:
            x = ll.contract(x)
        return x
