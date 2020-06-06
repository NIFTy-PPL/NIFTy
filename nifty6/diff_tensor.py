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
from .sugar import makeOp, domain_union, full
from .field import Field
from .multi_field import MultiField
from .operators.operator import Operator
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
            return DiffTensor.makeDiagonal(t1[0].vec*t2[0].vec, 1)
        return DiffTensor(GenLeibnizTensor(t1,t2,order_derivative))

    @staticmethod
    def makeComposed(new, old, order_derivative):
        if order_derivative < 1:
            raise ValueError
        return DiffTensor(ComposedTensor(new, old, order_derivative))
        

    def __add__(self, other):
        assertIsinstance(other, _DiffTensorImpl)
        return DiffTensor(SumTensor((self, other)))

#     @staticmethod
#     def makeSum

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
        return DiffTensor(self._impl, x+self._arg)


class Taylor(Operator):
    """Class describing a Taylor approximation up to a given order

    This class is a generalization of Fields/MultiFields (which are Taylor
    objects with maximum order 0), and Linearizations (which are Taylor objects
    with maximum order 1).
    Eventually, Taylor objects will most likely also get a "metric" and
    "want_metric" members.

    Parameters
    ----------
    tensors: tuple of DiffTensors
        The approximations at different orders
        - tensors[i].rank must be i+1
        - the domain of all tensors must be identical
        - the target of all tensors must be identical
    """
    def __init__(self, tensors):
        assertTrue(len(tensors)>=1)
        for i, t in enumerate(tensors):
            assertIsinstance(t, DiffTensor)
            assertEqual(i+1, t.rank)
            assertIdentical(t.domain, tensors[0].domain)
            assertIdentical(t.target, tensors[0].target)
        self._tensors = tensors

    @property
    def domain(self):
        return self._tensors[0].domain

    @property
    def target(self):
        return self._tensors[0].target

    @property
    def val(self):
        return self._tensors[0].vec

    @property
    def jac(self):
        return self._tensors[1].linop

    @property
    def tensors(self):
        return self._tensors

    @property
    def maxorder(self):
        """int: the maximum approximation order, i.e. one less than the number of stored tensors"""
        return len(self._tensors)-1

    def __len__(self):
        return len(self._tensors)

    def __getitem__(self, i):
        return self._tensors[i]

    @staticmethod
    def make_var(val, maxorder):
        if maxorder < 2:
            raise ValueError
        jacs = [DiffTensor.makeDiagonal(val,1)]
        jacs.append(DiffTensor.makeDiagonal(full(val.domain,1.),2))
        jacs += [DiffTensor.makeNull(val.domain, val.domain, i+1) for i in range(2,maxorder+1)]
        return Taylor(jacs)

    def new_from_prod(self, t2):
        assert self.maxorder == t2.maxorder
        return Taylor([DiffTensor.makeGenLeibniz(self, t2, i) for i in range(self.maxorder+1)])

    def new(self, new):
        res = [DiffTensor.makeDiagonal(new[0].vec,1),]
        res += [DiffTensor.makeComposed(new, self, i) for i in range(1, self.maxorder+1)]
        return Taylor(res)

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
        return self._op(x)

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


class NullTensor(_DiffTensorImpl):
    def __init__(self, domain, target, rank):
        super(NullTensor, self).__init__(domain, target, rank)

    def getLinop(self, x=()):
        return NullOperator(self.domain, self.target)

    def getVec(self, x=()):
        return full(self.target, 0.)


class SumTensor(_DiffTensorImpl):
    def __init__(self, tlist):
        from .sugar import domain_union
        for ts in tlist:
            assert isinstance(ts, _DiffTensorImpl)
            assert ts.target == tlist[0].target
            assert ts.rank == tlist[0].rank
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
    def __init__(self, t1, t2, order_derivative):
        assertIsinstance(t1, Taylor)
        assertIsinstance(t2, Taylor)
        assert(t1.maxorder >= order_derivative)
        assert(t2.maxorder >= order_derivative)
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
            if not (isinstance(self._t1[len(xx1)]._impl, NullTensor) or
                    isinstance(self._t2[len(xx2)]._impl, NullTensor)):
                tmp = (self._t1[len(xx1)].getVec(xx1)*
                       self._t2[len(xx2)].getVec(xx2))
                res = tmp if res is None else res+tmp
        if res is None:
            return full(self.target, 0.)
        return res

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
                if not (isinstance(self._t1[len(xx1)+1]._impl, NullTensor) or
                        isinstance(self._t2[len(xx2)]._impl, NullTensor)):
                    tmp = (makeOp(self._t2[len(xx2)].getVec(xx2))@
                           self._t1[len(xx1)+1].getLinop(xx1))
                    res = tmp if res is None else res+tmp
            else:
                if not (isinstance(self._t1[len(xx1)]._impl, NullTensor) or
                        isinstance(self._t2[len(xx2)+1]._impl, NullTensor)):
                    tmp = (makeOp(self._t1[len(xx1)].getVec(xx1))@
                           self._t2[len(xx2)+1].getLinop(xx2))
                    res = tmp if res is None else res+tmp
        if res is None:
            return NullOperator(self.domain, self.target)
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
    coeff = []
    for ll in lst:
        idx = np.where(np.array(ll)!=0)[0]
        coeff.append(_multifact(ll, idx, n))
    return lst, coeff

def _mysum(a,b):
    return [aa+bb for aa,bb in zip(a,b)]

class ComposedTensor(_DiffTensorImpl):
    """Implements a generalization of the chain rule for higher derivatives
    based on the Faà di Bruno's formula.
    """
    def __init__(self, new, old, order_derivative):
        assertIsinstance(old, Taylor)
        assertIsinstance(new, Taylor)
        assert(old.maxorder >= order_derivative)
        assert(new.maxorder >= order_derivative)
        assertIdentical(old.target, new.domain)
        super(ComposedTensor, self).__init__(old.domain, new.target, order_derivative+1)
        lst, self._coeff = _get_all_comb(order_derivative)
        self._oldidx, self._newidx = [], []
        self._old = old
        self._new = new
        #for lsti in lst:
        #    rr = lsti.sum()-1
        #    self._newidx.append(rr)
        #    mi = [oi+1 for oi, l in enumerate(lsti) for cnt in range(l)]
        #    self._oldidx.append(mi)
        #print(self._newidx)
        #print(self._oldidx)

        i, nmax = 0, len(lst)
        while i<nmax:
            rr = lst[i].sum()-1
            if isinstance(self._new[rr]._impl, NullTensor):
                del lst[i], self._coeff[i]
                nmax -=1
            else:
                self._newidx.append(rr)
                mi = [oi+1 for oi, l in enumerate(lst[i]) for cnt in range(l)]
                self._oldidx.append(mi)
                i += 1

    def getVec(self, x):
        res = [None, ]*(self._rank-1)
        for oldidx, newidx, coeff in zip(self._oldidx, self._newidx, self._coeff):
            tm = []
            cnt = 0
            for i in oldidx:
                op = self._old[i]
                sl = x[cnt:(cnt+op.rank-1)]
                cnt += op.rank-1
                tm.append(coeff*op.getVec(sl))
            res[newidx] = tm if res[newidx] is None else _mysum(res[newidx],tm)
        res = sum([m.getVec(rr) for m,rr in zip(self._new[1:self._rank], res) if rr is not None])
        return res

    def getLinop(self, y):
        nderiv = self.rank-1
        x = y + (None,)
        tms = [None, ]*nderiv
        rrs = [None, ]*nderiv
        for oldidx, newidx, coeff in zip(self._oldidx, self._newidx, self._coeff):
            tm = [] # list of linops
            rr = None # linop
            cnt = 0
            for i in oldidx:
                op = self._old[i]
                if op.rank == 1:
                    raise ValueError
                    rr = 0.
                elif op.rank == 2:
                    if x[cnt] is None:
                        rr = op.getLinop()
                    else:
                        tm.append(op.getVec((x[cnt],)))
                    cnt +=1
                else:
                    sl = x[cnt:(cnt+(op.rank-1))]
                    if sl[-1] is None:
                        rr = op.getLinop(sl[:-1])
                    else:
                        tm.append(op.getVec((sl,)))
                    cnt += op.rank-1
            rr = coeff*rr
            tms[newidx] = tm if tms[newidx] is None else _mysum(tms[newidx],tm)
            rrs[newidx] = rr if rrs[newidx] is None else rrs[newidx]+rr
        res = None
        for newmap, tm, rr in zip(self._new[1:self._rank], tms, rrs):
            if tm is not None:
                rr = newmap.getLinop(tuple(tm))@rr
                res = rr if res is None else res+rr
        return res
