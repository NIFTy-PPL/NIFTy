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


class _InterpolationOperator(Operator):
    def __init__(self, domain, func, xmin, xmax, delta, table_func=None, inverse_table_func=None):
        self._domain = self._target = DomainTuple.make(domain)
        self._xmin, self._xmax = float(xmin), float(xmax)
        self._d = float(delta)
        self._xs = np.arange(xmin, xmax+2*self._d, self._d)
        if table_func is not None:
            foo = func(np.random.randn(10))
            np.testing.assert_allclose(foo, inverse_table_func(table_func(foo)))
        self._table = table_func(func(self._xs))
        self._deriv = (self._table[1:]-self._table[:-1]) / self._d
        self._inv_table_func = inverse_table_func

    def apply(self, x, difforder):
        self._check_input(x)
        val = (np.clip(x.val, self._xmin, self._xmax) - self._xmin) / self._d
        fi = np.floor(val).astype(int)
        w = val - fi
        res = self._inv_table_func((1-w)*self._table[fi] + w*self._table[fi+1])
        resfld = Field(self._domain, res)
        if difforder == self.VALUE_ONLY:
            return resfld
        jac = makeOp(Field(self._domain, self._deriv[fi]*res))
        return Linearization(resfld, jac)


def InverseGammaOperator(domain, alpha, q, delta=0.001):
    """Transforms a Gaussian into an inverse gamma distribution.

    The pdf of the inverse gamma distribution is defined as follows:

    .. math::
        \\frac{q^\\alpha}{\\Gamma(\\alpha)}x^{-\\alpha -1}
        \\exp \\left(-\\frac{q}{x}\\right)

    That means that for large x the pdf falls off like :math:`x^(-\\alpha -1)`.
    The mean of the pdf is at :math:`q / (\\alpha - 1)` if :math:`\\alpha > 1`.
    The mode is :math:`q / (\\alpha + 1)`.

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
    op = _InterpolationOperator(domain,
                                lambda x: invgamma.ppf(norm.cdf(x), float(alpha)),
                                -8.2, 8.2, delta, np.log, np.exp)
    if np.isscalar(q):
        return op.scale(q)
    return makeOp(q) @ op
