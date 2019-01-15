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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from scipy.stats import invgamma, norm

from ..domain_tuple import DomainTuple
from ..field import Field
from ..linearization import Linearization
from ..operators.operator import Operator
from ..sugar import makeOp


class InverseGammaOperator(Operator):
    """Operator which transforms a Gaussian into an inverse gamma distribution.

    The pdf of the inverse gamma distribution is defined as follows:

    .. math::
        \frac {\beta ^{\alpha }}{\Gamma (\alpha )}}x^{-\alpha -1}\exp \left(-{\frac {\beta }{x}}\right)

    That means that for large x the pdf falls off like x^(-alpha -1).
    The mean of the pdf is at q / (alpha - 1) if alpha > 1.
    The mode is q / (alpha + 1).

    This transformation is implemented as a linear interpolation which maps a
    Gaussian onto a inverse gamma distribution.

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
    def __init__(self, domain, alpha, q, delta=0.001):
        self._domain = self._target = DomainTuple.make(domain)
        self._alpha, self._q, self._delta = float(alpha), float(q), float(delta)
        self._xmin, self._xmax = -8.2, 8.2
        # Precompute
        xs = np.arange(self._xmin, self._xmax+2*delta, delta)
        self._table = np.log(invgamma.ppf(norm.cdf(xs), self._alpha,
                                          scale=self._q))
        self._deriv = (self._table[1:]-self._table[:-1]) / delta

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, Linearization)
        val = x.val.local_data if lin else x.local_data

        val = (np.clip(val, self._xmin, self._xmax) - self._xmin) / self._delta

        # Operator
        fi = np.floor(val).astype(int)
        w = val - fi
        res = np.exp((1 - w)*self._table[fi] + w*self._table[fi + 1])

        points = Field.from_local_data(self._domain, res)
        if not lin:
            return points

        # Derivative of linear interpolation
        der = self._deriv[fi]*res

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
