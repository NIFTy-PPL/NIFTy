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
from ..sugar import full
from ..utilities import assertEqual, assertIsinstance
from .tensor_lin import _TensorLinObject


class Tensor(object):
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
        assertIsinstance(other, Tensor)
        from .tensor_sum import TensorSum
        if isinstance(self, TensorSum):
            if isinstance(other, TensorSum):
                return self.join(other, True)
            return self.append(other, True)
        if isinstance(other, TensorSum):
            return other.append(self, True)
        return TensorSum((self, other), (True, True))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        assertEqual(self.target, other.target)
        assertEqual(self.maxorder, other.maxorder)
        assertIsinstance(other, Tensor)
        from .tensor_sum import TensorSum
        if isinstance(self, TensorSum):
            if isinstance(other, TensorSum):
                return self.join(other, False)
            return self.append(other, False)
        if isinstance(other, TensorSum):
            return other.neg.append(self, True)
        return TensorSum((self, other), (True, False))

    def __rsub__(self, other):
        assertEqual(self.target, other.target)
        assertEqual(self.maxorder, other.maxorder)
        assertIsinstance(other, Tensor)
        from .tensor_sum import TensorSum
        if isinstance(self, TensorSum):
            if isinstance(other, TensorSum):
                return other.join(self, False)
            return self.neg.append(other, True)
        if isinstance(other, TensorSum):
            return other.append(self, False)
        return TensorSum((self, other), (False, True))

    def _check_input(self, x):
        if isinstance(x, _TensorLinObject):
            assertEqual(x.domain, self.domain)
        else:
            for xx in x:
                if xx.domain != self.domain:
                    raise ValueError

    def getVecs(self, x):
        r = (x,)+(full(self.domain,0.),)*(self.maxorder-1)
        return self.contract(r)

    def getLins(self, x):
        r = (x,)+(full(self.domain,0.),)*(self.maxorder-1)
        r = _TensorLinObject.make_trivial(r, self.maxorder)
        return self.contract(r)

    def contract(self, x):
        self._check_input(x)
        return self._contract(x)

    def _contract(self, x):
        raise NotImplementedError


class TrivialTensor(Tensor):
    def __init__(self, domain, maxorder):
        self._domain = self._target = domain
        self._maxorder = maxorder

    def _contract(self, inp):
        if not isinstance(inp, _TensorLinObject):
            return inp
        tm = _TensorLinObject.make_trivial(inp.val, self.maxorder)
        return inp.new_chain(tm.val, tm.lin)


class TensorChain(Tensor):
    def __init__(self, layers):
        mylayers = []
        for l in layers:
            if not isinstance(l, Tensor):
                raise ValueError
            if isinstance(l, TensorChain):
                raise ValueError
            if not isinstance(l, TrivialTensor):
                mylayers.append(l)
        self._domain = mylayers[0].domain
        self._target = mylayers[-1].target
        self._maxorder = mylayers[0].maxorder
        self._layers = tuple(mylayers)

    def append(self, tensors):
        assertEqual(self.target, tensors.domain)
        if isinstance(tensors, TensorChain):
            return TensorChain(self._layers + tensors._layers)
        elif not isinstance(tensors, Tensor):
            raise ValueError
        return TensorChain(self._layers + (tensors,))

    def _contract(self, x):
        for ll in self._layers:
            x = ll.contract(x)
        return x
