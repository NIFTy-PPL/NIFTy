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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import invgamma, laplace, norm

from .. import random
from ..domain_tuple import DomainTuple
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..operators.adder import Adder
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
        self._xs = np.arange(xmin, xmax, self._d)
        self._table = func(self._xs)
        if table_func is not None:
            if inv_table_func is None:
                raise ValueError
# MR FIXME: not sure whether we should have this in production code
            a = func(random.current_rng().random(10))
            a1 = _f_on_np(lambda x: inv_table_func(table_func(x)), a)
            np.testing.assert_allclose(a, a1)
            self._table = _f_on_np(table_func, self._table)
        self._interpolator = CubicSpline(self._xs, self._table)
        self._deriv = self._interpolator.derivative()
        self._inv_table_func = inv_table_func

    def apply(self, x):
        self._check_input(x)
        lin = x.jac is not None
        xval = x.val.val if lin else x.val
        res = self._interpolator(xval)
        res = Field(self._domain, res)
        if lin:
            res = x.new(res, makeOp(Field(self._domain, self._deriv(xval))))
        if self._inv_table_func is not None:
            res = self._inv_table_func(res)
        return res


def InverseGammaOperator(domain, alpha, q, delta=1e-2):
    """Transform a standard normal into an inverse gamma distribution.

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
    op = _InterpolationOperator(domain, lambda x: invgamma.ppf(norm._cdf(x), float(alpha)),
                                -8.2, 8.2, delta, lambda x: x.ptw("log"), lambda x: x.ptw("exp"))
    if np.isscalar(q):
        return op.scale(q)
    return makeOp(q) @ op


def LogInverseGammaOperator(domain, alpha, q, delta=1e-2):
    """Transform a standard normal into the log of an inverse gamma distribution.

    See also
    --------
    :class:`InverseGammaOperator`
    """
    op = _InterpolationOperator(domain, lambda x: np.log(invgamma.ppf(norm._cdf(x), float(alpha))),
                                -8.2, 8.2, delta)
    q = np.log(q) if np.isscalar(q) else q.log()
    return Adder(q, domain=op.target) @ op


class UniformOperator(Operator):
    """Transform a standard normal into a uniform distribution.

    The uniform distribution's support is ``[loc, loc + scale]``.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    loc: float
    scale: float

    """
    def __init__(self, domain, loc=0, scale=1):
        self._target = self._domain = DomainTuple.make(domain)
        self._loc = float(loc)
        self._scale = float(scale)

    def apply(self, x):
        self._check_input(x)
        lin = x.jac is not None
        xval = x.val.val if lin else x.val
        res = Field(self._target, self._scale*norm._cdf(xval) + self._loc)
        if not lin:
            return res
        jac = makeOp(Field(self._domain, norm._pdf(xval)*self._scale))
        return x.new(res, jac)

    def inverse(self, field):
        res = norm._ppf((field.val - self._loc) / self._scale)
        return Field(field.domain, res)


class LaplaceOperator(Operator):
    """Transform a standard normal to a Laplace distribution.

    Parameters
    -----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    loc : float

    scale : float
    """
    def __init__(self, domain, loc=0, scale=1):
        self._target = self._domain = DomainTuple.make(domain)
        self._loc = float(loc)
        self._scale = float(scale)

    def apply(self, x):
        self._check_input(x)
        lin = x.jac is not None
        xval = x.val.val if lin else x.val
        res = Field(self._target, laplace.ppf(norm._cdf(xval), self._loc, self._scale))
        if not lin:
            return res
        y = norm._cdf(xval)
        y = self._scale * np.where(y > 0.5, 1/(1-y), 1/y)
        jac = makeOp(Field(self.domain, y*norm._pdf(xval)))
        return x.new(res, jac)

    def inverse(self, x):
        res = laplace.cdf(x.val, self._loc, self._scale)
        res = norm._ppf(res)
        return Field(x.domain, res)
