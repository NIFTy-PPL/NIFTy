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
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..linearization import Linearization
from ..operators.operator import Operator
from ..sugar import makeOp


def _f_on_np(f, arr):
    fld = Field.from_raw(UnstructuredDomain(arr.shape), arr)
    return f(fld).val


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
        The function which is applied on the field. Assumed to act on numpy
        arrays.
    xmin : float
        The smallest value for which func will be evaluated.
    xmax : float
        The largest value for which func will be evaluated.
    delta : float
        Distance between sampling points for linear interpolation.
    table_func : function
        Non-linear function applied to table in order to transform the table
        to a more linear space. Assumed to act on `Linearization`s, optional.
    inv_table_func : function
        Inverse of table_func, optional.
    """
    def __init__(self, domain, func, xmin, xmax, delta, table_func=None, inv_table_func=None):
        self._domain = self._target = DomainTuple.make(domain)
        self._xmin, self._xmax = float(xmin), float(xmax)
        self._d = float(delta)
        self._xs = np.arange(xmin, xmax+2*self._d, self._d)
        self._table = func(self._xs)
        if table_func is not None:
            if inv_table_func is None:
                raise ValueError
            a = func(np.random.randn(10))
            a1 = _f_on_np(lambda x: inv_table_func(table_func(x)), a)
            np.testing.assert_allclose(a, a1)
            self._table = _f_on_np(table_func, self._table)
        self._inv_table_func = inv_table_func
        self._deriv = (self._table[1:] - self._table[:-1]) / self._d

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, Linearization)
        xval = x.val.val if lin else x.val
        val = (np.clip(xval, self._xmin, self._xmax) - self._xmin) / self._d
        fi = np.floor(val).astype(int)
        w = val - fi
        res = (1-w)*self._table[fi] + w*self._table[fi+1]
        res = Field(self._domain, res)
        if lin:
            res = x.new(res, makeOp(Field(self._domain, self._deriv[fi])))
        if self._inv_table_func is not None:
            res = self._inv_table_func(res)
        return res


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
    op = _InterpolationOperator(domain, lambda x: invgamma.ppf(norm.cdf(x), float(alpha)),
                                -8.2, 8.2, delta, lambda x: x.log(), lambda x: x.exp())
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
