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

from .. import Adder
from ..domain_tuple import DomainTuple
from ..field import Field
from ..linearization import Linearization
from ..operators.operator import Operator
from ..sugar import makeOp


class _InterpolationOperator(Operator):
    """
    Calculates a function pointwise on a field by interpolation.
    Can be supplied with a table_func to increase the interpolation accuracy,
    Best results are achieved when table_func(func) is roughly linear.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    func : function
        The function which is applied on the field.
    xmin : float
        The smallest value for which func will be evaluated.
    xmax : float
        The largest value for which func will be evaluated.
    delta : float
        Distance between sampling points for linear interpolation.
    table_func : {'None', 'exp', 'log', 'power'}

    exponent : float
        This is only used by the 'power' table_func.
    """
    def __init__(self, domain, func, xmin, xmax, delta, table_func=None, exponent=None):
        self._domain = self._target = DomainTuple.make(domain)
        self._xmin, self._xmax = float(xmin), float(xmax)
        self._d = float(delta)
        self._xs = np.arange(xmin, xmax+2*self._d, self._d)
        self._table = func(self._xs)
        self._transform = table_func is not None
        self._args = []
        if exponent is not None and table_func != 'power':
            raise Exception("exponent is only used when table_func is 'power'.")
        if table_func is None:
            pass
        elif table_func == 'exp':
            self._table = np.exp(self._table)
            self._inv_table_func = 'log'
        elif table_func == 'log':
            self._table = np.log(self._table)
            self._inv_table_func = 'exp'
        elif table_func == 'power':
            if not np.isscalar(exponent):
                return NotImplemented
            self._table = np.power(self._table, exponent)
            self._inv_table_func = '__pow__'
            self._args = [np.power(float(exponent), -1)]
        else:
            return NotImplemented
        self._deriv = (self._table[1:] - self._table[:-1]) / self._d

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, Linearization)
        xval = x.val.val if lin else x.val
        val = (np.clip(xval, self._xmin, self._xmax) - self._xmin) / self._d
        fi = np.floor(val).astype(int)
        w = val - fi
        res = (1-w)*self._table[fi] + w*self._table[fi+1]
        resfld = Field(self._domain, res)
        if not lin:
            if self._transform:
                resfld = getattr(resfld, self._inv_table_func)(*self._args)
            return resfld
        lin = Linearization.make_var(resfld)
        if self._transform:
            lin = getattr(lin, self._inv_table_func)(*self._args)
        jac = makeOp(Field(self._domain, self._deriv[fi]))
        return x.new(lin.val, lin.jac @ jac)


def InverseGammaOperator(domain, alpha, q, delta=0.001):
    """Transforms a Gaussian with unit covariance and zero mean into an
    inverse gamma distribution.

    The pdf of the inverse gamma distribution is defined as follows:

    .. math::
        \\frac{q^\\alpha}{\\Gamma(\\alpha)}x^{-\\alpha -1}
        \\exp \\left(-\\frac{q}{x}\\right)

    That means that for large x the pdf falls off like :math:`x^(-\\alpha -1)`.
    The mean of the pdf is at :math:`q / (\\alpha - 1)` if :math:`\\alpha > 1`.
    The mode is :math:`q / (\\alpha + 1)`.

    This transformation is implemented as a linear interpolation which maps a
    Gaussian onto an inverse gamma distribution.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    alpha : float
        The alpha-parameter of the inverse-gamma distribution.
    q : float or Field
        The q-parameter of the inverse-gamma distribution.
    delta : float
        Distance between sampling points for linear interpolation.
    """
    func = lambda x: invgamma.ppf(norm.cdf(x), float(alpha))
    op = _InterpolationOperator(domain, func, -8.2, 8.2, delta, 'log')
    if np.isscalar(q):
        return op.scale(q)
    return makeOp(q) @ op


def UniformOperator(domain, loc=0, scale=1, delta=1e-3):
    """
    Transforms a Gaussian with unit covariance and zero mean into a uniform
    distribution. The uniform distribution's support is ``[loc, loc + scale]``.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    loc: float or Field

    scale: float or Field

    delta : float
        Distance between sampling points for linear interpolation.
    """
    op = _InterpolationOperator(domain, lambda x: norm.cdf(x), -8.2, 8.2, delta)
    loc = Adder(loc, domain=domain)
    if np.isscalar(scale):
        return loc @ op.scale(scale)
    return loc @ makeOp(scale) @ op
