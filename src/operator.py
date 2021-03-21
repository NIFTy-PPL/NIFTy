from jax import numpy as np
from jax import jvp, vjp


class Likelihood():
    def __init__(
        self,
        energy,
        metric,
        draw_metric_sample=None,
    ):
        self._hamiltonian = energy
        self._metric = metric
        self._draw_metric_sample = draw_metric_sample

    def __call__(self, primals):
        return self._hamiltonian(primals)

    def jit(self):
        from jax import jit
        return Likelihood(jit(self._hamiltonian), jit(self._metric),
                jit(self._draw_metric_sample))

    def __matmul__(self, f):
        nham = lambda x: self.energy(f(x))
        def met(primals, tangents):
            y, t = jvp(f, (primals,), (tangents,))
            r = self.metric(y, t)
            _, bwd = vjp(f, primals)
            res = bwd(r)
            return res[0]

        def draw_sample(primals, key):
            y, bwd = vjp(f, primals)
            samp, nkey = self.draw_sample(y, key)
            return bwd(samp)[0], nkey

        return Likelihood(nham, met, draw_sample)

    def __add__(self, ham):
        if not isinstance(ham, Likelihood):
            te = (
                "object which to add to this instance is of invalid type"
                f" {type(ham)!r}"
            )
            raise TypeError(te)

        def draw_metric_sample(primals, key, **kwargs):
            # Ensure that samples are not inverted in any of the recursive calls
            assert "from_inverse" not in kwargs
            # Ensure there is no prior for the CG algorithm in recursive calls
            # as the prior is sensible only for the top-level likelihood
            assert "x0" not in kwargs

            key, subkeys = random.split(key, 2)
            smpl_self, _ = self.draw_sample(primals, key=subkeys[0], **kwargs)
            smpl_other, _ = ham.draw_sample(primals, key=subkeys[1], **kwargs)

            return smpl_self + smpl_other, key

        return Likelihood(
            energy=lambda p: self(p) + ham(p),
            metric=lambda p, t: self.metric(p, t) + ham.metric(p, t),
            draw_metric_sample=draw_metric_sample
        )

    def energy(self, primals):
        return self._hamiltonian(primals)

    def metric(self, primals, tangents):
        return self._metric(primals, tangents)

    def draw_sample(
        self,
        primals,
        key,
        from_inverse = False,
        x0 = None,
        maxiter = None,
        **kwargs
    ):
        if not self._draw_metric_sample:
            nie = "`draw_sample` is not implemented"
            raise NotImplementedError(nie)

        if from_inverse:
            nie = "Cannot draw from the inverse of this operator"
            raise NotImplementedError(nie)
        else:
            return self._draw_metric_sample(primals, key=key, **kwargs)


def laplace_prior(alpha):
    """
    Takes random normal samples and outputs samples distributed according to
    P(x|a) = exp(-|x|/a)/a/2
    """
    from jax.scipy.stats import norm
    res = lambda x: (x<0)*(norm.logcdf(x) + np.log(2))\
                - (x>0)*(norm.logcdf(-x) + np.log(2))
    return lambda x: res(x)*alpha


def normal_prior(mean, std):
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


def lognormal_prior(mean, std):
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


def interpolate(xmin=-7., xmax=7., N=14000):
    """
    Replaces a local nonlinearity such as np.exp with a linear interpolation.

    Interpolating functions speeds up code and increases numerical stability in
    some cases, but at a cost of precision and range.

    Parameters
    ----------

    xmin: float
    minimal interpolation value. Default: -7.

    xmax: float
    maximal interpolation value. Default: 7.

    N: int
    How many points are used for the interpolation. Default: 14000
    """
    def decorator(f):
        from functools import wraps

        x = np.linspace(xmin, xmax, N)
        y = f(x)

        @wraps(f)
        def wrapper(t):
            return np.interp(t, x, y)
        return wrapper
    return decorator
