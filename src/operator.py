from jax import numpy as np
from jax import jvp, vjp
from jax.tree_util import Partial, tree_leaves, all_leaves

from .optimize import cg
from .sugar import is1d, random_like_shapewdtype


class ShapeWithDtype():
    def __init__(self, shape, dtype=None):
        if not is1d(shape):
            ve = f"invalid shape; got {shape!r}"
            return ValueError(ve)

        self._shape = shape
        self._dtype = np.float64 if dtype is None else dtype

    @classmethod
    def from_leave(cls, element):
        # Usage: `tree_map(ShapeWithDtype.from_leave, tree)`
        import numpy as onp

        if not all_leaves((element, )):
            ve = "tree is not flat and still contains leaves"
            raise ValueError(ve)
        return cls(np.shape(element), onp.common_type(element))

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        nm = self.__class__.__name__
        return f"{nm}(shape={self.shape}, dtype={self.dtype})"


class Likelihood():
    def __init__(
        self,
        energy,
        metric=None,
        left_sqrt_metric=None,
        lsm_tangents_shape=None
    ):
        self._hamiltonian = energy
        self._metric = metric
        self._left_sqrt_metric = left_sqrt_metric

        if lsm_tangents_shape is not None:
            if is1d(lsm_tangents_shape):
                lsm_tangents_shape = ShapeWithDtype(lsm_tangents_shape)
            else:
                leaves = tree_leaves(lsm_tangents_shape)
                if not all(isinstance(e, ShapeWithDtype) for e in leaves):
                    te = "objects of invalid type in tangent shapes"
                    raise TypeError(te)
        self._lsm_tan_shp = lsm_tangents_shape

    def __call__(self, primals):
        return self._hamiltonian(primals)

    def energy(self, primals):
        return self._hamiltonian(primals)

    def metric(self, primals, tangents):
        if self._metric is None:
            nie = "`metric` is not implemented"
            raise NotImplementedError(nie)
        return self._metric(primals, tangents)

    def left_sqrt_metric(self, primals, tangents):
        if self._left_sqrt_metric is None:
            nie = "`left_sqrt_metric` is not implemented"
            raise NotImplementedError(nie)
        return self._left_sqrt_metric(primals, tangents)

    def inv_metric(self, primals, tangents, cg=cg, **cg_kwargs):
        res, _ = cg(Partial(self.metric, primals), tangents, **cg_kwargs)
        return res

    def draw_sample(self, primals, key, from_inverse=False, cg=cg, **cg_kwargs):
        if self._lsm_tan_shp is None:
            nie = "Cannot draw sample without knowing the shape of the data"
            raise NotImplementedError(nie)

        if from_inverse:
            nie = "Cannot draw from the inverse of this operator"
            raise NotImplementedError(nie)
        else:
            white_sample = random_like_shapewdtype(self._lsm_tan_shp, key=key)
            return self.left_sqrt_metric(primals, white_sample)

    @property
    def left_sqrt_metric_tangents_shape(self):
        return self._lsm_tan_shp

    def new(self, energy, metric, left_sqrt_metric):
        return Likelihood(
            energy,
            metric=metric,
            left_sqrt_metric=left_sqrt_metric,
            lsm_tangents_shape=self._lsm_tan_shp
        )

    def jit(self):
        from jax import jit

        if self._left_sqrt_metric is not None:
            j_lsm = jit(self._left_sqrt_metric)
        else:
            j_lsm = None
        j_m = jit(self._metric) if self._metric is not None else None
        return self.new(
            jit(self._hamiltonian), metric=j_m, left_sqrt_metric=j_lsm
        )

    def __matmul__(self, f):
        def energy_at_f(primals):
            return self.energy(f(primals))

        def metric_at_f(primals, tangents):
            y, t = jvp(f, (primals, ), (tangents, ))
            r = self.metric(y, t)
            _, bwd = vjp(f, primals)
            res = bwd(r)
            return res[0]

        def left_sqrt_metric_at_f(primals, tangents):
            y, bwd = vjp(f, primals)
            left_at_fp = self.left_sqrt_metric(y, tangents)
            return bwd(left_at_fp)[0]

        return self.new(
            energy_at_f,
            metric=metric_at_f,
            left_sqrt_metric=left_sqrt_metric_at_f
        )

    def __add__(self, other):
        if not isinstance(other, Likelihood):
            te = (
                "object which to add to this instance is of invalid type"
                f" {type(other)!r}"
            )
            raise TypeError(te)

        def joined_hamiltonian(p):
            return self.energy(p) + other.energy(p)

        def joined_metric(p, t):
            return self.metric(p, t) + other.metric(p, t)

        joined_tangents_shape = {
            "lh_left": self._lsm_tan_shp,
            "lh_right": other._lsm_tan_shp
        }

        def joined_left_sqrt_metric(p, t):
            return self.left_sqrt_metric(
                p, t["lh_left"]
            ) + other.left_sqrt_metric(p, t["lh_right"])

        return Likelihood(
            joined_hamiltonian,
            metric=joined_metric,
            left_sqrt_metric=joined_left_sqrt_metric,
            lsm_tangents_shape=joined_tangents_shape
        )


def laplace_prior(alpha):
    """
    Takes random normal samples and outputs samples distributed according to
    P(x|a) = exp(-|x|/a)/a/2
    """
    from jax.scipy.stats import norm
    res = lambda x: (x<0)*(norm.logcdf(x) + np.log(2))\
                - (x>0)*(norm.logcdf(-x) + np.log(2))
    return lambda x: res(x) * alpha


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
