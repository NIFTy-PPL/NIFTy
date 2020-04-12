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
import nifty6 as ift
import numpy as np
from itertools import product
from scipy.special import factorial

class DiffTensor:
    @property
    def domain(self):
        return self._domain
    @property
    def target(self):
        return self._target
    @property
    def nderiv(self):
        return self._nderiv

    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        assert len(x) == self._nderiv
        for xx in x:
            assert xx.domain == self._domain
        return self._apply(x)

    def partial_contract(self, x):
        if len(x) > self._nderiv:
            raise ValueError("Too many indices to contract")
        if len(x) == self._nderiv:
            return self.apply(x)
        if len(x) == self._nderiv-1:
            return self.contract_to_one(x)
        return PartialContractedTensor(self, x)

    def contract_to_one(self, y):
        assert len(y) == self._nderiv-1
        for yy in y:
            assert yy.domain == self._domain
        return self._contract_to_one(y)
    
    def _apply(self, x):
        raise NotImplementedError
    
    def _contract_to_one(self,y):
        raise NotImplementedError

class PartialContractedTensor(DiffTensor):
    def __init__(self, tensor, y):
        self._tensor = tensor
        self._domain = tensor.domain
        self._target = tensor.target
        self._nderiv = tensor.nderiv-len(y)
        self._y = y
    def _apply(self, x):
        return self._tensor(x+self._y)

    def _contract_to_one(self, y):
        self._tensor.contract_to_one(y+self._y)

    def partial_contract(self, x):
        if len(x) > self._nderiv:
            raise ValueError("Too many indices to contract")
        if len(x) == self._nderiv:
            return self(x)
        if len(x) == self._nderiv-1:
            return self.contract_to_one(x)
        return PartialContractedTensor(self._tensor, x+self._y)

class DiagonalTensor(DiffTensor):
    def __init__(self, vec, n):
        self._domain = self._target = vec.domain
        self._vec = vec
        self._nderiv = n
    def _apply(self, x):
        res = 1.
        for xx in x:
            res = res*xx
        return self._vec*res
    def _contract_to_one(self, y):
        return ift.makeOp(self._apply(y))

class SumTensor(DiffTensor):
    def __init__(self, tlist):
        self._tlist = tlist
        self._target = tlist[0].target
        self._nderiv = tlist[0].nderiv
        for ts in tlist:
            assert self._target == ts.target
            assert self._nderiv == ts.nderiv
        dom = ift.domain_union([op.domain for op in tlist])
        self._domain = dom

    def _apply(self, x):
        res = 0.
        for tli in self._tlist:
            res = res + tli([xj.extract(tli.domain) for xj in x])
        return res

    def _contract_to_one(self, y):
        res = None
        for tli in self._tlist:
            tmp = tli.contract_to_one([yj.extract(tli.domain) for yj in y])
            res = tmp if res is None else res + tmp
        return res

class GenLeibnizTensor(DiffTensor):
    def __init__(self, j1, j2, v1, v2):
        self._domain = ift.domain_union([j1[0].domain, j2[0].domain])
        self._target = j1[0].target
        self._nderiv = len(j1)
        assert len(j1) == len(j2)
        for f1, f2 in zip(j1, j2):
            assert f1 is None or f1.target == self._target
            assert f2 is None or f2.target == self._target
        assert v1.domain == self._target
        assert v2.domain == self._target
        self._j1 = [v1, ] + j1
        self._j2 = [v2, ] + j2

    def _apply(self, x):
        x1 = [xj.extract(self._j1[1].domain) for xj in x]
        x2 = [xj.extract(self._j2[1].domain) for xj in x]
        lst = list(product([0, 1], repeat=self._nderiv))
        res = 0.
        for inds in lst:
            inds = np.array(inds)
            inds2 = np.ones(self._nderiv) - inds
            xx1 = [x1[inp] for inp in np.where(inds == 1)[0]] #FIXME: xx1 = x1[inds] does not work for lists?
            xx2 = [x2[inp] for inp in np.where(inds2 == 1)[0]]
            l1 = len(xx1)
            if l1 == 0:
                tm = self._j1[0]
            elif l1 == 1:
                tm = self._j1[1](xx1[0])
            else:
                tm = self._j1[l1](xx1)
            l2 = len(xx2)
            if l2 == 0:
                tm = tm*self._j2[0]
            elif l2 == 1:
                tm = tm*self._j2[1](xx2[0])
            else:
                tm = tm*self._j2[l2](xx2)
            res = res + tm
        return res

    def _contract_to_one(self, x):
        x1 = [xj.extract(self._j1[1].domain) for xj in x]
        x2 = [xj.extract(self._j2[1].domain) for xj in x]
        x1.append(None)
        x2.append(None)
        lst = list(product([0, 1], repeat=self._nderiv))
        res = None
        for inds in lst:
            inds = np.array(inds)
            inds2 = np.ones(self._nderiv) - inds
            xx1 = [x1[inp] for inp in np.where(inds == 1)[0]]
            xx2 = [x2[inp] for inp in np.where(inds2 == 1)[0]]
            l1 = len(xx1)
            if l1 == 0:
                tm = self._j1[0]
            elif l1 == 1:
                tm = self._j1[1] if xx1[0] is None else self._j1[1](xx1[0])
            else:
                if xx1[-1] is None:
                    tm = self._j1[l1].contract_to_one(xx1[:-1])
                else:
                    tm = self._j1[l1](xx1)
            l2 = len(xx2)
            if l2 == 0:
                tmm = self._j2[0]
            elif l2 == 1:
                tmm = self._j2[1] if xx2[0] is None else self._j2[1](xx2[0])
            else:
                if xx2[-1] is None:
                    tmm = self._j2[l2].contract_to_one(xx2[:-1])
                else:
                    tmm = self._j2[l2](xx2)
            if isinstance(tm, (ift.Field, ift.MultiField)):
                tm = ift.makeOp(tm)@tmm
            else:
                tm = ift.makeOp(tmm)@tm
            res = tm if res is None else res + tm
        return res

def _constraint(lst):
    res = 0
    for n in range(len(lst)):
        res += lst[n]*(n+1)
    return res == len(lst)

def _multifact(lst,idx):
    res = factorial(len(lst))
    for ind in idx:
        res /= factorial(lst[ind])*factorial(ind)**lst[ind]
    return res

def _get_all_comb(n):
    maxvals = []
    for i in range(n):
        maxvals.append(n//(i+1))
    lst = [i for i in product(*(range(i+1) for i in maxvals)) if _constraint(i)]
    idx, coeff = [], []
    for ll in lst:
        idx.append(np.where(np.array(ll)!=0)[0])
        coeff.append(_multifact(ll, idx[-1]))
    return lst, idx, coeff

class ComposedTensor(DiffTensor):
    """Implements a generalization of the chain rule for higher derivatives
    based on the Faà di Bruno's formula.
    """
    def __init__(self, old, new):
        assert len(old) == len(new)
        self._nderiv = len(new)
        self._domain = old[0].domain
        self._target = new[0].target
        for i in range(self._nderiv):
            assert old[i].domain == self._domain
            if new[i] is not None:
                assert new[i].target == self.target
                assert old[i].target == new[i].domain
        self._old = old
        self._new = new
        self._lst, self._idx, self._coeff = _get_all_comb(self._nderiv)
        self._newmap = []
        i, nmax = 0, len(self._lst)
        while i < nmax:
            rr = 0
            for ll in self._lst[i]:
                rr += ll
            rr -= 1
            if self._new[rr] == None:
                del self._lst[i]
                del self._idx[i]
                del self._coeff[i]
                nmax -= 1
            else:
                self._newmap.append(rr)
                i += 1

    def _apply(self, x):
        res = None
        for i in range(len(self._lst)):
            tm = []
            cnt = 0
            for a in self._idx[i]:
                op = self._old[a]
                for _ in range(self._lst[i][a]):
                    sl = x[cnt:(cnt+a+1)]
                    sl = sl[0] if len(sl)==1 else sl
                    tm.append(op(sl))
                    cnt += a+1
            tm = tm[0] if len(tm)==1 else tm
            rr = self._new[self._newmap[i]](tm)
            res = rr if res is None else res+rr
        return res

    def _contract_to_one(self, y):
        x = y + [None,]
        res = None
        for i in range(len(self._lst)):
            tm = []
            rr = None
            cnt = 0
            for a in self._idx[i]:
                op = self._old[a]
                for _ in range(self._lst[i][a]):
                    sl = x[cnt:(cnt+a+1)]
                    if sl[-1] is None:
                        rr = op if len(sl)==1 else op.contract_to_one(sl[:-1])
                    else:
                        sl = sl[0] if len(sl)==1 else sl
                        tm.append(op(sl))
                    cnt += a+1
            
            if len(tm)==0:
                rr = self._new[self._newmap[i]]@rr
            else:
                rr = self._new[self._newmap[i]].contract_to_one(tm)@rr
            res = rr if res is None else res+rr
        return res
