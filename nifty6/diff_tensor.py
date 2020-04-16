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
from itertools import product
from scipy.special import factorial
from .sugar import makeOp
from .field import Field
from .multi_field import MultiField


def assertIsinstance(t1, t2):
    if not isinstance(t1, t2):
        raise TypeError("type mismatch")


def assertIdentical(t1, t2):
    if t1 is not t2:
        raise ValueError("objects are not identical")


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

    @staticmethod
    def makeDiagonal(vec, rank):
        return DiffTensor(DiagonalTensor(vec, rank))

    def __add__(self, other):
        assertIsinstance(other, _DiffTensorImpl)
        return DiffTensor(SumTensor((self, other)))

#     @staticmethod
#     def makeSum
#     def makeGenLeibniz
#     def makeComposed

    @property
    def domain(self):
        return self._impl.domain

    @property
    def target(self):
        return self._impl.target

    @property
    def rank(self):
        return self._impl.rank-len(self._arg)

    def getVec(self, x=()):
        return self._impl.getVec(x+self._arg)

    def getLinop(self, x=()):
        return self._impl.getLinop(x+self._arg)

    @property
    def vec(self):
        return self.getVec()

    @property
    def linop(self):
        return self.getLinop()

    def contract(self, x):
        return DiffTensor(self._impl, x+self._arg)


# class DiffTensorRank1(DiffTensor):
#     def __init__(self, vec):
#         super(DiffTensorRank1, self).__init__(vec.domain, vec.domain, 1)
#         self._vec = vec
#
#     @property
#     def vec(self):
#         return self._vec
#
#     def _contract(self, x):
#         # must not arrive here
#         raise ValueError
#
#
# class DiffTensorRank2(DiffTensor):
#     def __init__(self, op):
#         super(DiffTensorRank2, self).__init__(op.domain, op.target, 2)
#         self._op = op
#
#     @property
#     def linop(self):
#         return self._op
#
#     def _contract(self,x):
#         return DiffTensorRank1(self._op(x[0]))


class DiagonalTensor(_DiffTensorImpl):
    def __init__(self, vec, rank):
        super(DiagonalTensor, self).__init__(vec.domain, vec.domain, rank)
        self._vec = vec

    def _helper(self, x, rnk):
        if self._rank-len(x) != rnk:
            raise ValueError
        res = self._vec
        for xx in x:
            res = res*xx
        return res if rnk == 1 else makeOp(res)

    def getLinop(self, x=()):
        return self._helper(x, 2)

    def getVec(self, x=()):
        return self._helper(x, 1)


class SumTensor(_DiffTensorImpl):
    def __init__(self, tlist):
        from .sugar import domain_union
        for ts in tlist:
            assert isinstance(ts, _DiffTensorImpl)
            assert ts.target == tlist[0].target
            assert ts.rank == tlist[0].rank
            print("xxx", ts, ts.rank)
        dom = domain_union([t.domain for t in tlist])
        super(SumTensor, self).__init__(dom, tlist[0].target, tlist[0].rank)
        self._tlist = tlist

    def getVec(self, x=()):
        if self._rank-len(x) != 1:
            raise ValueError
        res = 0.
        for tli in self._tlist:
            res = res + tli.getVec(tuple(xj.extract(tli.domain) for xj in x))
        return res

    def getLinop(self, x=()):
        if self._rank-len(x) != 2:
            raise ValueError
        res = None
        for tli in self._tlist:
            tmp = tli.getLinop(tuple(xj.extract(tli.domain) for xj in x))
            res = tmp if res is None else res + tmp
        return res


# unfinished code from here on ...

class GenLeibnizTensor(_DiffTensorImpl):
    def __init__(self, t1, t2):
# t1 and t2 are Taylor objects ... exact interface TBD
        assert len(t1) == len(t2)
        dom = ift.domain_union(t1.domain, t22.domain)
        self._t1 = t1
        self._t22 = t2
        super(GenLeibnizTensor, self).__init__(dom, t1.target, len(t1)+1)
        lst = tuple(np.array(i) for i in product([0, 1], repeat=len(t1)))
        self._id1 = [np.where(inds == 1)[0] for inds in lst]
        self._id2 = [np.where(inds == 0)[0] for inds in lst]

    def getVec(self, x=()):
        if self._rank-len(x) != 1:
            raise ValueError
        x1 = [xj.extract(self._lin1.domain) for xj in x]
        x2 = [xj.extract(self._lin2.domain) for xj in x]
        res = None
        for id1, id2 in zip(self._id1, self._id2):
            xx1 = [x1[inp] for inp in id1] #FIXME: xx1 = x1[inds] does not work for lists?
            xx2 = [x2[inp] for inp in id2]
            tmp = self._lin1[len(xx1)].getVec(xx1)*self._lin2[len(xx2)].getVec(xx2)
            res = tmp if res is None else res+tmp
        return res

    def getLinop(self, x=()):
        if self.rank-len(x) != 2:
            raise ValueError
        x1 = [xj.extract(self._lin1.domain) for xj in x]
        x2 = [xj.extract(self._lin2.domain) for xj in x]
        res = None
        for id1, id2 in zip(self._id1, self._id2):
            xx1 = [x1[inp] for inp in id1]
            xx2 = [x2[inp] for inp in id2]
            tmp = self._t1[len(xx1)].getLinop(xx1)@self._t2[len(xx2)].getLinop(xx2)
            res = tmp if res is None else res+tmp
        return res

def _constraint(lst):
    return len(lst) == sum([v*(i+1) for i,v in enumerate(lst)])

def _multifact(lst,idx,n):
    res = factorial(n)
    ls = lst[idx]
    res /= np.prod(factorial(ls)*factorial(np.array(idx)+1.)**ls)
    return res

def _get_all_comb(n):
    mv = n//(np.arange(n)+1)
    lst = [np.array(i) for i in product(*(range(i+1) for i in mv)) if _constraint(i)]
    idx, coeff = [], []
    for ll in lst:
        idx.append(np.where(np.array(ll)!=0)[0])
        coeff.append(_multifact(ll, idx[-1],n))
    return lst, idx, coeff

def _mysum(a,b):
    if isinstance(a, list) or isinstance(b,list):
        return [aa+bb for aa,bb in zip(a,b)]
    return a+b

class ComposedTensor(_DiffTensorImpl):
    """Implements a generalization of the chain rule for higher derivatives
    based on the Faà di Bruno's formula.
    """
    def __init__(self, old, new):
        assert len(old) == len(new)
        super(ComposedTensor, self).__init__(old[0].domain, new[0].target, len(new)+1)
        for o, n in zip(old, new):
            assert o.domain == self._domain
            if n is not None:
                assert n.target == self.target
                assert o.target == n.domain
        lst, self._idx, self._coeff = _get_all_comb(self._rank-1)
        self._newidx = []
        self._new = new
        self._newmap = []
        self._oldmap = []
        i, nmax = 0, len(lst)
        while i < nmax:
            rr = lst[i].sum()-1
            if new[rr] is None:
                del lst[i], self._idx[i], self._coeff[i]
                nmax -= 1
            else:
                self._newidx.append(rr)
                self._newmap.append(new[rr])
                mi = []
                for a in self._idx[i]:
                    mi = mi + [old[a] ,]*lst[i][a]
                self._oldmap.append(mi)
                i += 1

    def getVec(self, x):
        res = [None, ]*(self._nrank-1)
        for oldmap, newidx, coeff in zip(self._oldmap, self._newidx, self._coeff):
            tm = []
            cnt = 0
            for op in oldmap:
                sl = x[cnt:(cnt+op.nderiv)]
                cnt += op.rank-1
                tm.append(coeff*op(sl))
            tm = tm[0] if len(tm)==1 else tm
            res[newidx] = tm if res[newidx] is None else _mysum(res[newidx],tm)
        res = sum([m(rr) for m,rr in zip(self._new, res) if rr is not None])
        return res

    def _contract_to_one(self, y):
        x = y + [None,]
        tms = [None, ]*self._nderiv
        rrs = [None, ]*self._nderiv
        for oldmap, newidx, coeff in zip(self._oldmap, self._newidx, self._coeff):
            tm = []
            rr = None
            cnt = 0
            for op in oldmap:
                if isinstance(op, ift.LinearOperator):
                    if x[cnt] is None:
                        rr = op
                    else:
                        tm.append(op(x[cnt]))
                    cnt +=1
                else:
                    sl = x[cnt:(cnt+op.nderiv)]
                    if sl[-1] is None:
                        rr = op.contract_to_one(sl[:-1])
                    else:
                        tm.append(op(sl))
                    cnt += op.nderiv
            rr = coeff*rr
            tms[newidx] = tm if tms[newidx] is None else _mysum(tms[newidx],tm)
            rrs[newidx] = rr if rrs[newidx] is None else rrs[newidx]+rr
        res = None
        for newmap, tm, rr in zip(self._new, tms, rrs):
            if tm is not None:
                if len(tm)==0:
                    rr = newmap@rr
                else:
                    rr = newmap.contract_to_one(tm)@rr
                res = rr if res is None else res+rr
        return res
