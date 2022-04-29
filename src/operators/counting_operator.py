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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras

from .endomorphic_operator import EndomorphicOperator
from .operator import Operator


class CountingOperator(Operator):
    def __init__(self, domain):
        from ..sugar import makeDomain

        self._domain = self._target = makeDomain(domain)
        self._count_apply = 0
        self._count_apply_lin = 0
        self._derivative = _JacCountingOperator(self._domain)

    def apply(self, x):
        from ..sugar import is_linearization

        self._check_input(x)
        if is_linearization(x):
            self._count_apply_lin += 1
            x = x.new(x.val, self._derivative)
        else:
            self._count_apply += 1
        return x

    @property
    def count_apply(self):
        return self._count_apply

    @property
    def count_apply_lin(self):
        return self._count_apply_lin

    @property
    def count_jac(self):
        return self._derivative._count_times

    @property
    def count_jac_adj(self):
        return self._derivative._count_adjoint_times

    def __repr__(self):
        return f"CountingOperator({self._domain.__repr__()})"

    def report(self):
        s = [f"* apply: \t\t{self.count_apply:>7}",
             f"* apply Linearization: \t{self.count_apply_lin:>7}",
             f"* Jacobian: \t\t{self.count_jac:>7}",
             f"* Adjoint Jacobian: \t{self.count_jac_adj:>7}"]
        return "\n".join(s)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        from .simplify_for_const import InsertionOperator
        return None, self @ InsertionOperator(self.domain, c_inp)


class _JacCountingOperator(EndomorphicOperator):
    def __init__(self, domain):
        from ..sugar import makeDomain

        self._domain = makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._count_times = 0
        self._count_adjoint_times = 0

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            self._count_times += 1
        else:
            self._count_adjoint_times += 1
        return x
