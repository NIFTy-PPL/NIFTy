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
        self._op1 = op1
        self._op2 = op2

    @property
    def domain(self):
        return self._op2.domain

    @property
    def target(self):
        return self._op1.target

    @property
    def capability(self):
        return self._op1.capability & self._op2.capability

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES or mode == self.ADJOINT_INVERSE_TIMES:
            return self._op1.apply(self._op2.apply(x, mode), mode)
        return self._op2.apply(self._op1.apply(x, mode), mode)
