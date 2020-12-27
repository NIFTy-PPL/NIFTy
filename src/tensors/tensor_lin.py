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

from ..utilities import assertEqual


class _TensorLinObject(object):
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
        return _TensorLinObject(val, _TrivialTensorLin(val[0].domain, maxorder))

    def new_chain(self, val, lin):
        if isinstance(self.lin, _TensorChainLin):
            lin = self.lin.append(lin)
        elif isinstance(self.lin, _TrivialTensorLin):
            lin = lin
        elif isinstance(lin, _TrivialTensorLin):
            lin = self.lin
        else:
            lin = _TensorChainLin((self.lin,lin))
        return _TensorLinObject(val, lin)


class _TensorLin(object):
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


class _TrivialTensorLin(_TensorLin):
    def __init__(self, domain, maxorder):
        super(_TrivialTensorLin, self).__init__(domain, domain, maxorder)

    def _apply(self, x):
        return x

    def _adjoint(self, x):
        return x


class _TensorChainLin(_TensorLin):
    def __init__(self, ops):
        domain = ops[0].domain
        target = ops[-1].target
        super(_TensorChainLin,self).__init__(domain, target, ops[-1].maxorder)
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
        return _TensorChainLin(self._ops + (op,))