from typing import Callable, Optional

from jax import numpy as np


def laplace_prior(alpha):
    """
    Takes random normal samples and outputs samples distributed according to

    .. math::
        P(x|a) = exp(-|x|/a)/a/2

    """
    from jax.scipy.stats import norm
    res = lambda x: (x<0)*(norm.logcdf(x) + np.log(2))\
                - (x>0)*(norm.logcdf(-x) + np.log(2))
    return lambda x: res(x) * alpha


def normal_prior(mean, std) -> Callable:
    """Match standard normally distributed random variables to non-standard
    variables.
    """
    def standard_to_normal(xi):
        return mean + std * xi

    return standard_to_normal


def lognormal_moments(mean, std):
    """Compute the cumulants a log-normal process would need to comply with the
    provided mean and standard-deviation `std`
    """
    if np.any(mean <= 0.):
        raise ValueError(f"`mean` must be greater zero; got {mean!r}")
    if np.any(std <= 0.):
        raise ValueError(f"`std` must be greater zero; got {std!r}")

    logstd = np.sqrt(np.log1p((std / mean)**2))
    logmean = np.log(mean) - 0.5 * logstd**2
    return logmean, logstd


def lognormal_prior(mean, std) -> Callable:
    """Moment-match standard normally distributed random variables to log-space

    Takes random normal samples and outputs samples distributed according to

    .. math::
        P(xi|mu,sigma) \\propto exp(mu + sigma * xi)

    such that the mean and standard deviation of the distribution matches the
    specified values.
    """
    standard_to_normal = normal_prior(*lognormal_moments(mean, std))

    def standard_to_lognormal(xi):
        return np.exp(standard_to_normal(xi))

    return standard_to_lognormal


def interpolator(
    func: Callable,
    xmin: float,
    xmax: float,
    *,
    step: Optional[float] = None,
    num: Optional[int] = None,
    table_func: Optional[Callable] = None,
    inv_table_func: Optional[Callable] = None
):  # Adapted from NIFTy
    """
    Evaluate a function point-wise by interpolation.  Can be supplied with a
    table_func to increase the interpolation accuracy, Best results are
    achieved when `lambda x: table_func(func(x))` is roughly linear.

    Parameters
    ----------
    func : function
        Function to interpolate.
    xmin : float
        The smallest value for which `func` will be evaluated.
    xmax : float
        The largest value for which `func` will be evaluated.
    step : float
        Distance between sampling points for linear interpolation. Either of
        `step` or `num` must be specified.
    num : int
        The number of interpolation points. Either of `step` of `num` must be
        specified.
    table_func : function
        Non-linear function applied to the tabulated function in order to
        transform the table to a more linear space.
    inv_table_func : function
        Inverse of `table_func`.
    """
    # from scipy.interpolate import CubicSpline

    if step is not None and num is not None:
        ve = "either but not both of `step` and `num` must be specified"
        raise ValueError(ve)
    if step is not None:
        xs = np.arange(xmin, xmax + step, step)
    elif num is not None:
        xs = np.linspace(xmin, xmax, num)
    else:
        ve = "either of `step` or `num` must be specified"
        raise ValueError(ve)

    ys = func(xs)
    if table_func is not None:
        if inv_table_func is None:
            raise ValueError("no `inv_table_func` specified")
        ys = table_func(ys)

    # interpolator = CubicSpline(xs, ys)
    # deriv = interpolator.derivative()

    def interp(x):
        # res = interpolator(x)
        res = np.interp(x, xs, ys)
        if inv_table_func is not None:
            res = inv_table_func(res)
        return res

    return interp


def invgamma_prior(a, scale, loc=0., step=1e-2) -> Callable:
    """Transform a standard normal into an inverse gamma distribution.

    The pdf of the inverse gamma distribution is defined as follows using
    :math:`q` to denote the scale:

    .. math::
        \\frac{q^\\a}{\\Gamma(\\a)}x^{-\\a -1}
        \\exp \\left(-\\frac{q}{x}\\right)

    That means that for large x the pdf falls off like :math:`x^(-\\a -1)`.
    The mean of the pdf is at :math:`q / (\\a - 1)` if :math:`\\a > 1`.
    The mode is :math:`q / (\\a + 1)`.

    This transformation is implemented as a linear interpolation which maps a
    Gaussian onto an inverse gamma distribution.

    Parameters
    ----------
    a : float
        The shape-parameter of the inverse-gamma distribution.
    scale : float
        The scale-parameter of the inverse-gamma distribution.
    loc : float
        An option shift of the whole distribution.
    step : float
        Distance between sampling points for linear interpolation.
    """
    from scipy.stats import invgamma, norm

    xmin, xmax = -8.2, 8.2  # (1. - norm.cdf(8.2)) * 2 < 1e-15
    standard_to_invgamma = interpolator(
        lambda x: invgamma.ppf(norm._cdf(x), a=a, loc=loc, scale=scale),
        xmin,
        xmax,
        step=step,
        table_func=np.log,
        inv_table_func=np.exp
    )
    return standard_to_invgamma
