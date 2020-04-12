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
import scipy.special as sp
import numpy as np
import itertools

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
        l= [yj.extract(self._tlist[0].domain) for yj in y]
        res = self._tlist[0].contract_to_one(l)
        for tli in self._tlist[1:]:
            l = [yj.extract(tli.domain) for yj in y]
            res = res + tli.contract_to_one(l)
        return res

class GenLeibnizTensor(DiffTensor):
    def __init__(self, j1, j2, v1, v2):
        self._domain = ift.domain_union([j1[0].domain, j2[0].domain])
        self._target = j1[0].target
        self._nderiv = len(j1)
        assert len(j1) == len(j2)
        for i in range(len(j1)):
            assert j1[i] is None or j1[i].target == self._target
            assert j2[i] is None or j2[i].target == self._target
        assert v1.domain == self._target
        assert v2.domain == self._target
        self._j1 = [v1, ] + j1
        self._j2 = [v2, ] + j2

    def _apply(self, x):
        x1 = [xj.extract(self._j1[1].domain) for xj in x]
        x2 = [xj.extract(self._j2[1].domain) for xj in x]
        lst = list(itertools.product([0, 1], repeat=self._nderiv))
        res = 0.
        for inds in lst:
            inds = np.array(inds)
            inds2 = np.ones(self._nderiv) - inds
            inds = np.where(inds == 1)[0]
            inds2 = np.where(inds2 == 1)[0]
            xx1 = [x1[inp] for inp in inds] # FIXME: just xx1 = x1[inds] ?
            xx2 = [x2[inp] for inp in inds2] 
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

class ComposedTensor(DiffTensor):
    """Implements a generalization of the chain rule for higher derivatives.
    Currently only supports max. 4th derivative. However there exists a
    (somewhat complicated) closed form solution known as:
    "Faà di Bruno's formula".
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

    def _apply(self, x):
        if self._nderiv == 2:
            tm = self._old[1](x)
            res = self._new[0](tm)
            if self._new[1] is not None:
                res = res + self._new[1]([self._old[0](x[0]), self._old[0](x[1])])
            return res

        if self._nderiv == 3:
            tm = self._old[2](x)
            res = self._new[0](tm)
            if self._new[1] != None:
                dx1 = self._old[0](x[0])
                tm = self._old[1](x[1:])
                res = res + 3.*self._new[1]([dx1, tm])
            if self._new[2] != None:
                dx2 = self._old[0](x[1])
                dx3 = self._old[0](x[2])
                res = res + self._new[2]([dx1, dx2, dx3])
            return res

        if self._nderiv == 4:
            tm = self._old[3](x)
            res = self._new[0](tm)
            if self._new[1] != None:
                dx1 = self._old[0](x[0])
                tm = self._old[2](x[1:])
                tm2 = self._old[1](x[:2])
                dx34 = self._old[1](x[2:])
                lst = [3.*tm2+4.*dx1, 3.*dx34+4.*tm]
                res = res + self._new[1](lst)
            if self._new[2] != None:
                dx2 = self._old[0](x[1])
                res = res + 6.*self._new[2]([dx1, dx2, dx34])
            if self._new[3] != None:
                dx3 = self._old[0](x[2])
                dx4 = self._old[0](x[3])
                res = res + self._new[3]([dx1, dx2, dx3, dx4])
            return res

        if self._nderiv == 1 or self._nderiv > 4:
            raise NotImplementedError

    def _contract_to_one(self, y):
        if self._nderiv == 2:
            tm = self._old[1].contract_to_one(y)
            res = self._new[0](tm)
            if self._new[1] != None:
                tm = self._old[0](y[0])
                res = res + self._new[1].contract_to_one([tm,]) @ self._old[0]
            return res

        if self._nderiv == 3:
            tm = self._old[2].contract_to_one(y)
            res = self._new[0](tm)
            if self._new[1] != None:
                tm = self._old[1](y)
                res = res + 3.*self._new[1].contract_to_one([tm,])@self._old[0]
            if self._new[2] != None:
                dx1 = self._old[0](y[0])
                dx2 = self._old[0](y[1])
                res = res + self._new[2].contract_to_one([dx1, dx2])@self._old[0]
            return res

        if self._nderiv == 4:
            tm = self._old[3].contract_to_one(y)
            res = self._new[0] @ tm
            if self._new[1] != None:
                tm = self._old[2](y)
                dy23 = self._old[1](y[1:])
                re = self._new[1].contract_to_one([3.*dy23+4.*tm, ])
                re = re @ (3.*self._old[1].contract_to_one([y[0],])+4.*self._old[0])
                res = res + re
            if self._new[2] != None:
                dy1 = self._old[0](y[0])
                res = res + 6.*self._new[2].contract_to_one([dy1,dy23])@self._old[0]
            if self._new[3] != None:
                dy2 = self._old[0](y[1])
                dy3 = self._old[0](y[2])
                res = res + self._new[3].contract_to_one([dy1,dy2,dy3])@self._old[0]
            return res

        if self._nderiv == 1 or self._nderiv > 4:
            raise NotImplementedError
