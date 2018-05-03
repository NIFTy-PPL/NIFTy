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
from ..field import Field
from ..domain_tuple import DomainTuple


class ScalarDistributor(LinearOperator):
    def __init__(self, weight):
        super(ScalarDistributor, self).__init__()

        if not isinstance(weight, Field):
            raise TypeError("Field object required")

        self._weight = weight
        self._domain = DomainTuple.make(())

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return self._weight*x.to_global_data()
        else:
            return Field.from_global_data(self.domain, self._weight.vdot(x))

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._weight.domain

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
