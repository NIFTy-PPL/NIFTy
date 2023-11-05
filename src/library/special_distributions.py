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
# Copyright(C) 2013-2022 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import gamma, invgamma, laplace, norm

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

        try:
            from jax import numpy as jnp

            def jax_expr(x):
                res = jnp.interp(x, self._xs, self._table)
                if inv_table_func is not None:
                    ve = (
                        "can not translate arbitrary inverse"
                        f" table function {inv_table_func!r}"
                    )
                    raise ValueError(ve)
                return res

            self._jax_expr = jax_expr
        except ImportError:
            self._jax_expr = None

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


class InverseGammaOperator(Operator):
    """Transform a standard normal into an inverse gamma distribution.

    The pdf of the inverse gamma distribution is defined as follows:

    .. math::
        \\frac{q^\\alpha}{\\Gamma(\\alpha)}x^{-\\alpha -1}
        \\exp \\left(-\\frac{q}{x}\\right)

    That means that for large x the pdf falls off like :math:`x^{(-\\alpha -1)}`.
    The mean of the pdf is at :math:`q / (\\alpha - 1)` if :math:`\\alpha > 1`.
    The mode is :math:`q / (\\alpha + 1)`.

    The operator can be initialized by setting either alpha and q or mode and mean.
    In accordance to the statements above the mean must be greater
    than the mode. Otherwise would get alpha < 0 and so no mean would be defined.

    This transformation is implemented as a linear interpolation which maps a
    Gaussian onto an inverse gamma distribution.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    alpha : float
        The alpha-parameter of the inverse-gamma distribution.
    q : float or :class:`nifty8.field.Field`
        The q-parameter of the inverse-gamma distribution.
    mode: float
        The mode of the inverse-gamma distribution.
    mean: float
        The mean of the inverse-gamma distribution.
    delta : float
        Distance between sampling points for linear interpolation.
    """
    def __init__(self, domain, alpha=None, q=None, delta=1e-2, mode=None, mean=None):
        self._domain = self._target = DomainTuple.make(domain)

        if alpha is not None and q is not None:
            self._alpha = float(alpha)
            self._q = q if isinstance(q, Field) else float(q)
            self._mode = self._q / (self._alpha + 1)
            if self._alpha > 1:
                self._mean = self._q / (self._alpha - 1)
        elif mean is not None and mode is not None:
            if mean < mode:
                raise ValueError('Mean should be greater than mode, otherwise alpha < 0')
            self._mean = float(mean)
            self._mode = float(mode)
            self._alpha = 2 / (self._mean / self._mode - 1) + 1
            self._q = self._mode * (self._alpha + 1)
        else:
            raise ValueError("Either one pair of arguments (mode, mean or alpha, q) must be given.")

        self._delta = float(delta)
        op = _InterpolationOperator(self._domain, lambda x: invgamma.ppf(norm._cdf(x), float(self._alpha)),
                                    -8.2, 8.2, self._delta, lambda x: x.ptw("log"), lambda x: x.ptw("exp"))
        if np.isscalar(self._q):
            op = op.scale(self._q)
        else:
            op = makeOp(self._q) @ op
        self._op = op

        try:
            from ..re.num.stats_distributions import invgamma_prior

            q_val = self._q.val if isinstance(self._q, Field) else self._q
            self._jax_expr = invgamma_prior(float(self._alpha), q_val)
        except ImportError:
            self._jax_expr = None

    def apply(self, x):
        return self._op(x)

    @property
    def alpha(self):
        """float : The value of the alpha-parameter of the inverse-gamma distribution"""
        return self._alpha

    @property
    def q(self):
        """float : The value of q-parameters of the inverse-gamma distribution"""
        return self._q

    @property
    def mode(self):
        """float : The value of the mode of the inverse-gamma distribution"""
        return self._mode

    @property
    def mean(self):
        """float : The value of the mean of the inverse-gamma distribution. Only existing for alpha > 1."""
        if self._alpha <= 1:
            raise ValueError('mean only existing for alpha > 1')
        return self._mean

    @property
    def var(self):
        """float : The value of the variance of the inverse-gamma distribution. Only existing for alpha > 2."""
        if self._alpha <= 2:
            raise ValueError('variance only existing for alpha > 2')
        return self._q**2 / ((self._alpha - 1)**2 * (self._alpha - 2))


class GammaOperator(Operator):
    """Transform a standard normal into a gamma distribution.

    The pdf of the gamma distribution is defined as follows:

    .. math::
        \\frac{1}{\\Gamma(k)\\theta^k} x^{k-1}
        \\exp \\left( -\\frac{x}{\\theta} \\right)

    This transformation is implemented as a linear interpolation which maps a
    Gaussian onto a gamma distribution.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    alpha : float
        The shape parameter of the gamma distribution.
    beta : float or :class:`nifty8.field.Field`
        The rate parameter of the gamma distribution.
    theta : float or :class:`nifty8.field.Field`
        The scale parameter of the gamma distribution.
    mean : float
        Mean of the gamma distribution.
    var : float
        Variance of the gamma distribution.
    delta : float
        Distance between sampling points for linear interpolation.
    """
    def __init__(self, domain, alpha=None, beta=None, theta=None, mean=None, var=None, delta=1e-2):
        self._domain = self._target = DomainTuple.make(domain)

        if mean is not None and var is not None:
            theta = var / mean
            alpha = mean / theta

        if beta is None and theta is None:
            raise ValueError("Either beta or theta need to be defined")
        if beta is not None:
            theta = 1 / beta
        self._alpha = float(alpha)
        self._theta = theta if isinstance(theta, Field) else float(theta)

        self._delta = float(delta)
        op = _InterpolationOperator(self._domain, lambda x: gamma.ppf(norm._cdf(x), self._alpha),
                                    -8.2, 8.2, self._delta)
        if np.isscalar(self._theta):
            op = op.scale(self._theta)
        else:
            op = makeOp(self._theta) @ op
        self._op = op

    def apply(self, x):
        return self._op(x)

    @property
    def alpha(self):
        """float : The value of the shape-parameter of the gamma distribution"""
        return self._alpha

    @property
    def beta(self):
        """float : The value of rate parameter of the gamma distribution"""
        return 1 / self._theta

    @property
    def theta(self):
        """float : The value of scale parameter of the gamma distribution"""
        return self._theta

    @property
    def mode(self):
        """float : The value of the mode of the gamma distribution. Exists for alpha >= 1 or k >= 1."""
        if self._alpha < 1:
            raise ValueError('mode only existing for alpha >= 1 or k >= 1')
        return (self._alpha - 1) * self._theta

    @property
    def mean(self):
        """float : The value of the mean of the gamma distribution."""
        return self._alpha * self._theta

    @property
    def var(self):
        """float : The value of the variance of the gamma distribution."""
        return self._alpha * self._theta**2


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
