from typing import Callable

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
    """Compute the cumulants a log-normal process would have to comply with the
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
