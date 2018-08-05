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

from __future__ import absolute_import, division, print_function

from ..compat import *
from ..operators.operator import Operator
from ..operators.sandwich_operator import SandwichOperator
from ..sugar import makeOp
from ..linearization import Linearization


class BernoulliEnergy(Operator):
    def __init__(self, p, d):
        super(BernoulliEnergy, self).__init__()
        self._p = p
        self._d = d

    @property
    def domain(self):
        return self._p.domain

    @property
    def target(self):
        return DomainTuple.scalar_domain()

    def apply(self, x):
        x = self._p(x)
        v = x.log().vdot(-self._d) - (1.-x).log().vdot(1.-self._d)
        if not isinstance(x, Linearization):
            return v
        met = makeOp(1./(x.val*(1.-x.val)))
        met = SandwichOperator.make(x.jac, met)
        return v.add_metric(met)
