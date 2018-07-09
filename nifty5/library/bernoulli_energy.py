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
from numpy import inf, isnan
from ..minimization.energy import Energy
from ..operators.sandwich_operator import SandwichOperator
from ..sugar import log, makeOp


class BernoulliEnergy(Energy):
    def __init__(self, p, d):
        """
        p: Model object


        """
        super(BernoulliEnergy, self).__init__(p.position)
        self._p = p
        self._d = d

        p_val = self._p.value
        self._value = -self._d.vdot(log(p_val)) - (1.-d).vdot(log(1.-p_val))
        if isnan(self._value):
            self._value = inf
        metric = makeOp(1. / (p_val * (1.-p_val)))
        self._gradient = self._p.jacobian.adjoint_times(metric(p_val-d))

        self._metric = SandwichOperator.make(self._p.jacobian, metric)

    def at(self, position):
        return self.__class__(self._p.at(position), self._d)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    def metric(self):
        return self._metric
