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
derivatives (We ignore the notion of dual spaces here):

Consider a function f that maps from space A to space B:

The first derivative is a Linear operator that maps from A to B, it can be
denoted as a 2nd order tensor of the form J_{i,j} where i labels the
coordinates of A and j the coords of B. The comma "," indicates derivatives,
i.E. everything to the right of "," is an index arising from taking a
derivative. (This is a common way of notation in tensor calculus).

A higher order derivative can be denoted in the same manner, e.g. J_{i,jk}
denotes the second derivative of f and is a 3rd order tensor.
It is a Multilinear operator that maps from AxA to B, i.E. it takes two vectors
in A and maps to a vector in B. Higher orders are constructed accordingly.

Some important properties:
All input vectors have to live on the same space A. 

The differential tensors are symmetric w.r.t. the latter indices, i.E. the 
indices produced by the derivatives. Therefore
    J_{i,jk} a_j b_k = J_{i,kj} b_k a_j
where we sum over repeated indices. This means that the order in which we
contract does not matter. As a consequence partial contraction leads to a
unique new tensor irrespective of how we contracted:
    B_{ik} := J_{i,jk} a_j = J_{i,kj} a_j
This leads to the special case where we can form a new Linear operator (Tensor
of 2nd order) from a n+1 order tensor by contracting with n-1 vectors. This new
operator then has the same capabilities as usual Linear operators.
"""

import numpy as np
from functools import reduce
from .sugar import makeOp, domain_union, full, makeField
from .field import Field
from .multi_field import MultiField
from .operators.scaling_operator import ScalingOperator
from .operators.diagonal_operator import DiagonalOperator
from .operators.simple_linear_operators import NullOperator



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


class _DiffTensorImpl(object):
    def __init__(self, dom, tgt, rank):
        self._domain = dom
        self._target = tgt
        self._rank = rank

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def rank(self):
        return self._rank

    def getVec(self, x=()):
        raise NotImplementedError

    def getLinop(self, x=()):
        raise NotImplementedError


class DiffTensor(_DiffTensorImpl):
    """Should not be called directly by the user."""
    def __init__(self, impl, arg=()):
        assertIsinstance(impl, _DiffTensorImpl)
        self._impl = impl
        arg = tuple(arg)
        if len(arg) >= self._impl.rank:
            raise ValueError("too many contractions")
        for f in arg:
            assertIsinstance(f, (Field, MultiField))
            assertIdentical(f.domain, self._impl.domain)
        self._arg = arg

    @property
    def isNullTensor(self):
        return isinstance(self._impl, NullTensor)

    @property
    def isLinearTensor(self):
        return isinstance(self._impl, LinearTensor)

    @property
    def isDiagonalTensor(self):
        return isinstance(self._impl, DiagonalTensor)

    @property
    def isComposedTensor(self):
        return isinstance(self._impl, ComposedTensor)

    @staticmethod
    def makeDiagonal(vec, rank):
        return DiffTensor(DiagonalTensor(vec, rank))

    @staticmethod
    def makeNull(domain, target, rank):
        return DiffTensor(NullTensor(domain, target, rank))

    @staticmethod
    def makeGenLeibniz(t1, t2, order_derivative):
        if order_derivative<0:
            raise ValueError
        if order_derivative==0:
            return DiffTensor.makeVec(t1[0].vec*t2[0].vec,
                                      domain=domain_union((t1.domain,t2.domain)))
        return DiffTensor(GenLeibnizTensor(t1,t2,order_derivative))

    @staticmethod
    def makeComposed(new, old, order_derivative):
        if order_derivative < 1:
            raise ValueError
        if old.istrivial:
            return new[order_derivative]
        if new.isDiagonal and old.isDiagonal:
            return ComposedTensor.simplify_for_diagonal(new[1:order_derivative+1],
                                                        old[1:order_derivative+1],
                                                        order_derivative)
        if new.isLinear and old.isDiagonal:
            if isinstance(new[1]._impl._op, DiagonalOperator):
                return ComposedTensor.simplify_for_diagonal(new[1:order_derivative+1],
                                                            old[1:order_derivative+1],
                                                            order_derivative,
                                                            diag = True)
            if isinstance(new[1]._impl._op, ScalingOperator):
                return ComposedTensor.simplify_for_diagonal(new[1:order_derivative+1],
                                                            old[1:order_derivative+1],
                                                            order_derivative,
                                                            scaling = True)
        return DiffTensor(ComposedTensor(new, old, order_derivative))

    @staticmethod
    def makeVec(vec, domain=None):
        return DiffTensor(VecTensor(vec, domain=domain))

    @staticmethod
    def makeLinear(op):
        return DiffTensor(LinearTensor(op))

    def __add__(self, other):
        assertIsinstance(other, _DiffTensorImpl)
        return DiffTensor(SumTensor((self, other), (True,True)))

    def __sub__(self, other):
        assertIsinstance(other, _DiffTensorImpl)
        return DiffTensor(SumTensor((self, other), (True,False)))

    def __rsub__(self, other):
        assertIsinstance(other, _DiffTensorImpl)
        return DiffTensor(SumTensor((self, other), (False,True)))


    @property
    def domain(self):
        return self._impl.domain

    @property
    def target(self):
        return self._impl.target

    @property
    def rank(self):
        """Returns the tensor's rank."""
        return self._impl.rank-len(self._arg)

    def getVec(self, x=()):
        """Contracts the tensor with the fields in `x` and returns the
        resulting rank-1 tensor as a Field/MultiField.
        If the rank is not 1 after contraction with all members of `x`,
        an exception is raised."""
        return self._impl.getVec(tuple(x)+self._arg)

    def getLinop(self, x=()):
        """Contracts the tensor with the fields in `x` and returns the
        resulting rank-2 tensor as a LinearOperator.
        If the rank is not 2 after contraction with all members of `x`,
        an exception is raised."""
        return self._impl.getLinop(tuple(x)+self._arg)

    @property
    def vec(self):
        """Shorthand for getVec(())."""
        return self.getVec()

    @property
    def linop(self):
        """Shorthand for getLinop(())."""
        return self.getLinop()

    def contract(self, x):
        """Contracts the tensor with the fields in `x` and returns a new tensor
        with a rank of self.rank-len(x).
        If the resulting rank is lower than 1, an exception is raised."""
        return DiffTensor(self._impl, tuple(x)+self._arg)


class VecTensor(_DiffTensorImpl):
    def __init__(self, vec, domain=None):
        if domain is None:
            domain = vec.domain
        super(VecTensor, self).__init__(domain, vec.domain, 1)
        self._vec = vec

    def getVec(self, x=()):
        if len(x) != 0:
            raise ValueError
        return self._vec


class LinearTensor(_DiffTensorImpl):
    def __init__(self, op):
        super(LinearTensor, self).__init__(op.domain, op.target, 2)
        self._op = op

    def getLinop(self, x=()):
        if len(x) != 0:
            raise ValueError
        return self._op

    def getVec(self, x=()):
        if len(x) != 1:
            raise ValueError
        return self._op(x[0])


class DiagonalTensor(_DiffTensorImpl):
    def __init__(self, vec, rank):
        super(DiagonalTensor, self).__init__(vec.domain, vec.domain, rank)
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


class NullTensor(_DiffTensorImpl):
    def __init__(self, domain, target, rank):
        super(NullTensor, self).__init__(domain, target, rank)
        self._op, self._vec = NullOperator(self.domain, self.target), full(self.target, 0.)

    def getLinop(self, x=()):
        return self._op

    def getVec(self, x=()):
        return self._vec


class SumTensor(_DiffTensorImpl):
    def __init__(self, tlist, sign):
        from .sugar import domain_union
        for ts in tlist:
            assertIsinstance(ts, _DiffTensorImpl)
            assertTrue(ts.target == tlist[0].target)
            assertTrue(ts.rank == tlist[0].rank)
        dom = domain_union([t.domain for t in tlist])
        super(SumTensor, self).__init__(dom, tlist[0].target, tlist[0].rank)
        self._tlist = tlist
        self._sign = sign

    def getVec(self, x=()):
        if self._rank-len(x) != 1:
            raise ValueError
        res = 0.
        for tli, sign in zip(self._tlist, self._sign):
            tm = tli.getVec((xj.extract(tli.domain) for xj in x))
            res = res + tm if sign else res - tm
        return res

    def getLinop(self, x=()):
        if self._rank-len(x) != 2:
            raise ValueError
        res = None
        for tli, sign in zip(self._tlist, self._sign):
            tmp = tli.getLinop((xj.extract(tli.domain) for xj in x))
            tmp = tmp if sign else -tmp
            res = tmp if res is None else res + tmp
        return res


class GenLeibnizTensor(_DiffTensorImpl):
    def __init__(self, t1, t2, order_derivative):
        from .taylor import Taylor
        assertIsinstance(t1, Taylor)
        assertIsinstance(t2, Taylor)
        assertTrue(t1.maxorder >= order_derivative)
        assertTrue(t2.maxorder >= order_derivative)
        dom = domain_union((t1.domain, t2.domain))
        self._t1 = t1
        self._t2 = t2
        super(GenLeibnizTensor, self).__init__(dom, t1.target, order_derivative+1)

    def getVec(self, x=()):
        if self._rank-len(x) != 1:
            raise ValueError
        x1 = [xj.extract(self._t1.domain) for xj in x]
        x2 = [xj.extract(self._t2.domain) for xj in x]
        ord = self._rank-1
        res = None
        for i in range(1<<ord):
            # (i & (1<<j)) is True iff the j-th bit is set in i
            xx1 = [x1[j] for j in range(ord) if (i & (1<<j))]
            xx2 = [x2[j] for j in range(ord) if not (i & (1<<j))]
            if not (self._t1[len(xx1)].isNullTensor or
                    self._t2[len(xx2)].isNullTensor):
                tmp = (self._t1[len(xx1)].getVec(xx1)*
                       self._t2[len(xx2)].getVec(xx2))
                res = tmp if res is None else res+tmp
        return res if res is not None else full(self.target, 0.)

    def getLinop(self, x=()):
        if self._rank-len(x) != 2:
            raise ValueError
        x1 = [xj.extract(self._t1.domain) for xj in x]
        x2 = [xj.extract(self._t2.domain) for xj in x]
        ord = self._rank-1
        res = None
        for i in range(1<<ord):
            # (i & (1<<j)) is True iff the j-th bit is set in i
            xx1 = [x1[j] for j in range(ord-1) if (i & (1<<j))]
            xx2 = [x2[j] for j in range(ord-1) if not (i & (1<<j))]
            if i & (1<<(ord-1)):
                if not (self._t1[len(xx1)+1].isNullTensor or
                        self._t2[len(xx2)].isNullTensor):
                    tmp = (makeOp(self._t2[len(xx2)].getVec(xx2))@
                           self._t1[len(xx1)+1].getLinop(xx1))
                    res = tmp if res is None else res+tmp
            else:
                if not (self._t1[len(xx1)].isNullTensor or
                        self._t2[len(xx2)+1].isNullTensor):
                    tmp = (makeOp(self._t1[len(xx1)].getVec(xx1))@
                           self._t2[len(xx2)+1].getLinop(xx2))
                    res = tmp if res is None else res+tmp
        return res if res is not None else NullOperator(self.domain, self.target)


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


class ComposedTensor(_DiffTensorImpl):
    """Implements a generalization of the chain rule for higher derivatives.
    """
    def __init__(self, new, old, order_derivative, _internal_call=False):
        if _internal_call:
            assertEqual(len(old), order_derivative)
            assertEqual(len(new), order_derivative)
            self._old, self._new = old, new
        else:
            from .taylor import Taylor
            assertIsinstance(old, Taylor)
            assertIsinstance(new, Taylor)
            assertTrue(old.maxorder >= order_derivative)
            assertTrue(new.maxorder >= order_derivative)
            assertIdentical(old.target, new.domain)
            self._old = old[1:order_derivative+1]
            self._new = new[1:order_derivative+1]
        super(ComposedTensor, self).__init__(old[0].domain, new[0].target, order_derivative+1)
        self._partitions = _all_partitions_nontrivial(order_derivative, self._new)

    def getVec(self, x):
        if len(x) != self._rank-1:
            raise ValueError
        res = None
        for p in self._partitions:
            rr = (self._old[len(b)-1].getVec((x[ind] for ind in b)) for b in p)
            res = self._new[len(p)-1].getVec(rr) if res is None else res + self._new[len(p)-1].getVec(rr)
        return res if res is not None else full(self.target, 0.)

    def getLinop(self, x):
        if len(x) != self._rank-2:
            raise ValueError
        res = None
        for p in self._partitions:
            rr = ()
            for b in p:
                if self._rank-2 not in b:
                    rr += (self._old[len(b)-1].getVec((x[ind] for ind in b)), )
                else:
                    tm = self._old[len(b)-1].getLinop((x[ind] for ind in b if ind != self._rank-2))
            r = self._new[len(p)-1].getLinop(rr)@tm
            res = r if res is None else res+r
        return res if res is not None else NullOperator(self.domain, self.target)

    @staticmethod
    def simplify_for_diagonal(new, old, n, diag=False, scaling=False):
        pps = _all_partitions_nontrivial(n, new)
        re = 0.
        for p in pps:
            rr = reduce(lambda x,y: x*y, [old[len(b)-1]._impl._vec for b in p])
            if diag:
                re = re + makeField(rr.domain,new[len(p)-1]._impl._op._ldiag)*rr
            elif scaling:
                re = re + new[len(p)-1]._impl._op._factor*rr
            else:
                re = re + new[len(p)-1]._impl._vec*rr
        return DiffTensor.makeDiagonal(re, n+1)
