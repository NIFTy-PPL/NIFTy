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
from ..operators.operator import EnergyOperator
from ..operators.sandwich_operator import SandwichOperator
from ..domain_tuple import DomainTuple
from ..linearization import Linearization


class GaussianEnergy(EnergyOperator):
    def __init__(self, mean=None, covariance=None, domain=None):
        super(GaussianEnergy, self).__init__()
        self._domain = None
        if mean is not None:
            self._checkEquivalence(mean.domain)
        if covariance is not None:
            self._checkEquivalence(covariance.domain)
        if domain is not None:
            self._checkEquivalence(domain)
        if self._domain is None:
            raise ValueError("no domain given")
        self._mean = mean
        self._icov = None if covariance is None else covariance.inverse

    def _checkEquivalence(self, newdom):
        if self._domain is None:
            self._domain = newdom
        else:
            if self._domain is not newdom:
                raise ValueError("domain mismatch")

    @property
    def domain(self):
        return self._domain

    def apply(self, x):
        residual = x if self._mean is None else x-self._mean
        icovres = residual if self._icov is None else self._icov(residual)
        res = .5*residual.vdot(icovres)
        if not isinstance(x, Linearization):
            return res
        metric = SandwichOperator.make(x.jac, self._icov)
        return res.add_metric(metric)
