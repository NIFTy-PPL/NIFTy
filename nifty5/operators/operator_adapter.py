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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from .linear_operator import LinearOperator
import numpy as np


class OperatorAdapter(LinearOperator):
    """Class representing the inverse and/or adjoint of another operator.

    Objects of this class are created internally by `LinearOperator` whenever
    the inverse and/or adjoint of an already existing operator object is
    requested via the `LinearOperator` attributes `inverse`, `adjoint` or
    `_flip_modes()`.

    Users should never have to create instances of this class directly.

    Parameters
    ----------
    op : LinearOperator
        The operator on which the adapter will act
    op_transform : int
        1) adjoint
        2) inverse
        3) adjoint inverse
    """

    def __init__(self, op, op_transform):
        super(OperatorAdapter, self).__init__()
        self._op = op
        self._trafo = int(op_transform)
        if self._trafo < 1 or self._trafo > 3:
            raise ValueError("invalid operator transformation")

    @property
    def domain(self):
        return self._op._dom(1 << self._trafo)

    @property
    def target(self):
        return self._op._tgt(1 << self._trafo)

    @property
    def capability(self):
        return self._capTable[self._trafo][self._op.capability]

    def _flip_modes(self, trafo):
        newtrafo = trafo ^ self._trafo
        return self._op if newtrafo == 0 \
            else OperatorAdapter(self._op, newtrafo)

    def apply(self, x, mode):
        return self._op.apply(x,
                              self._modeTable[self._trafo][self._ilog[mode]])

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        if self._trafo & self.INVERSE_BIT:
            return self._op.draw_sample(not from_inverse, dtype)
        return self._op.draw_sample(from_inverse, dtype)
