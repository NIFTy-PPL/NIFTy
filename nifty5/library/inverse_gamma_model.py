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

import numpy as np
from scipy.stats import invgamma, norm

from ..compat import *
from ..operators.operator import Operator
from ..linearization import Linearization
from ..field import Field
from ..sugar import makeOp


class InverseGammaModel(Operator):
    def __init__(self, domain, alpha, q):
        self._domain = domain
        self._alpha = alpha
        self._q = q

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._domain

    def apply(self, x):
        lin = isinstance(x, Linearization)
        val = x.val.local_data if lin else x.local_data
        # MR FIXME?!
        points = np.clip(val, None, 8.2)
        points = self.IG(points)
        points = Field.from_local_data(self._domain, points)
        if not lin:
            return points
        inner = norm.pdf(val)
        outer_inv = invgamma.pdf(invgamma.ppf(norm.cdf(val),
                                              self._alpha,
                                              scale=self._q),
                                 self._alpha, scale=self._q)
        # FIXME
        outer_inv = np.clip(outer_inv, 1e-20, None)
        outer = 1/outer_inv
        jac = makeOp(Field.from_local_data(self._domain, inner*outer))
        jac = jac(x.jac)
        return Linearization(points, jac)

    def IG(self, field):
        return invgamma.ppf(norm.cdf(field), self._alpha, scale=self._q)

# MR FIXME: Do we need this?
#     def inverseIG(self, u):
#         return Field.from_local_data(
#             u.domain, norm.ppf(invgamma.cdf(u.local_data, self._alpha,
#                                             scale=self._q)))
