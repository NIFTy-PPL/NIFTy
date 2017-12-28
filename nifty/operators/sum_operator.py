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


class SumOperator(LinearOperator):
    def __init__(self, op1, op2, neg=False):
        super(SumOperator, self).__init__()
        if op1.domain != op2.domain or op1.target != op2.target:
            raise ValueError("domain mismatch")
        self._op1 = op1
        self._op2 = op2
        self._neg = bool(neg)

    @property
    def domain(self):
        return self._op1.domain

    @property
    def target(self):
        return self._op1.target

    @property
    def capability(self):
        return (self._op1.capability & self._op2.capability &
                (self.TIMES | self.ADJOINT_TIMES))

    def apply(self, x, mode):
        self._check_mode(mode)
        res1 = self._op1.apply(x, mode)
        res2 = self._op2.apply(x, mode)
        return res1 - res2 if self._neg else res1 + res2
