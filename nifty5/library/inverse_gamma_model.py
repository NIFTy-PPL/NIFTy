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
from ..domain_tuple import DomainTuple
from ..field import Field
from ..linearization import Linearization
from ..operators.operator import Operator
from ..sugar import makeOp


class InverseGammaModel(Operator):
    def __init__(self, domain, alpha, q, delta):
        """Model which transforms a Gaussian into an inverse gamma distribution.

        The pdf of the inverse gamma distribution is defined as follows:

        .. math::
            \frac {\beta ^{\alpha }}{\Gamma (\alpha )}}x^{-\alpha -1}\exp \left(-{\frac {\beta }{x}}\right)

        That means that for large x the pdf falls off like x^(-alpha -1).
        The mean of the pdf is at q / (alpha - 1) if alpha > 1.
        The mode is q / (alpha + 1).

        This transformation is implemented as a linear interpolation which
        maps a Gaussian onto a inverse gamma distribution.

        Parameters
        ----------
        domain : Domain, tuple of Domain or DomainTuple
            The domain on which the field shall be defined. This is at the same
            time the domain and the target of the operator.
        alpha : float
            The alpha-parameter of the inverse-gamma distribution.
        q : float
            The q-parameter of the inverse-gamma distribution.
        delta : float
            distance between sampling points for linear interpolation.
        """
        self._domain = self._target = DomainTuple.make(domain)
        self._alpha, self._q, self._delta = alpha, q, delta

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, Linearization)
        val = x.val.local_data if lin else x.local_data

        val = np.clip(val, None, 8.2)
        # Precompute
        x0 = val.min()
        dx = self._delta
        xs = np.arange(x0, val.max()+2*dx, dx)
        table = np.log(invgamma.ppf(norm.cdf(xs), self._alpha, scale=self._q))

        # Operator
        fi = np.array(np.floor((val - x0)/dx), dtype=np.int)
        w = (val - xs[fi])/dx
        res = np.exp((1 - w)*table[fi] + w*table[fi + 1])

        points = Field.from_local_data(self._domain, res)
        if not lin:
            return points

        # Derivative of linear interpolation
        inner_der = (table[fi + 1] - table[fi])/dx
        der = inner_der*res

        jac = makeOp(Field.from_local_data(self._domain, der))
        jac = jac(x.jac)
        return x.new(points, jac)

    @staticmethod
    def IG(field, alpha, q):
        foo = invgamma.ppf(norm.cdf(field.local_data), alpha, scale=q)
        return Field.from_local_data(field.domain, foo)

    @staticmethod
    def inverseIG(u, alpha, q):
        res = norm.ppf(invgamma.cdf(u.local_data, alpha, scale=q))
        return Field.from_local_data(u.domain, res)

    @property
    def alpha(self):
        return self._alpha

    @property
    def q(self):
        return self._q
