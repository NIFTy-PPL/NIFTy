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

from .multi_linear_operator import MultiLinearOperator


class MultiChainOperator(MultiLinearOperator):
    """Class representing chains of multi-operators."""

    def __init__(self, ops, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        super(MultiChainOperator, self).__init__()
        self._ops = ops
        self._capability = self._all_ops
        for op in ops:
            self._capability &= op.capability

    @staticmethod
    def make(ops):
        ops = tuple(ops)
        if len(ops) == 1:
            return ops[0]
        return MultiChainOperator(ops, _callingfrommake=True)

    @property
    def domain(self):
        return self._ops[-1].domain

    @property
    def target(self):
        return self._ops[0].target

    def _flip_modes(self, trafo):
        ADJ = self.ADJOINT_BIT
        INV = self.INVERSE_BIT

        if trafo == 0:
            return self
        if trafo == ADJ or trafo == INV:
            return self.make([op._flip_modes(trafo)
                              for op in reversed(self._ops)])
        if trafo == ADJ | INV:
            return self.make([op._flip_modes(trafo) for op in self._ops])
        raise ValueError("invalid operator transformation")

    @property
    def capability(self):
        return self._capability

    def apply(self, x, mode):
        self._check_mode(mode)
        t_ops = self._ops if mode & self._backwards else reversed(self._ops)
        for op in t_ops:
            x = op.apply(x, mode)
        return x
