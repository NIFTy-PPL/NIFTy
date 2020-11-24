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

from .tensor import Tensor
from .tensor_lin import _TensorLin, _TensorLinObject


class _LinearTensorLin(_TensorLin):
    def __init__(self, op, maxorder):
        super(_LinearTensorLin,self).__init__(op.domain, op.target, maxorder)
        self._op = op

    def _apply(self, x):
        return tuple(self._op(xx) for xx in x)

    def _adjoint(self, x):
        return tuple(self._op.adjoint(xx) for xx in x)


class LinearTensor(Tensor):
    def __init__(self, op, maxorder):
        self._domain = op.domain
        self._target = op.target
        self._op = op
        self._maxorder = maxorder

    def _contract(self, inp):
        islin = isinstance(inp, _TensorLinObject)
        r = inp.val if islin else inp
        r = tuple(self._op(ii) for ii in r)
        if not islin:
            return r
        return inp.new_chain(r, _LinearTensorLin(self._op, self.maxorder))
