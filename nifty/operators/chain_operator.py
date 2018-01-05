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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from .linear_operator import LinearOperator


class ChainOperator(LinearOperator):
    def __init__(self, op1, op2):
        super(ChainOperator, self).__init__()
        if op2.target != op1.domain:
            raise ValueError("domain mismatch")
        self._capability = op1.capability & op2.capability
        op1 = op1._ops if isinstance(op1, ChainOperator) else (op1,)
        op2 = op2._ops if isinstance(op2, ChainOperator) else (op2,)
        self._ops = op1 + op2

    @property
    def domain(self):
        return self._ops[-1].domain

    @property
    def target(self):
        return self._ops[0].target

    @property
    def capability(self):
        return self._capability

    def apply(self, x, mode):
        self._check_mode(mode)
        t_ops = self._ops if mode & self._backwards else reversed(self._ops)
        for op in t_ops:
            x = op.apply(x, mode)
        return x
