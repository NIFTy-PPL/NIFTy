from typing import Callable, Optional

import sys
from jax import numpy as np
from jax.tree_util import tree_map

from .likelihood import Likelihood, ShapeWithDtype


def _shape_w_fixed_dtype(dtype):
    def shp_w_dtp(e):
        return ShapeWithDtype(np.shape(e), dtype)

    return shp_w_dtp


def Gaussian(
    data,
    noise_cov_inv: Optional[Callable] = None,
    noise_std_inv: Optional[Callable] = None
):
    """Gaussian likelihood of the data

    Parameters
    ----------
    data : tree-like structure of np.ndarray and float
        Data with additive noise following a Gaussian distribution.
    noise_cov_inv : callable acting on type of data
        Function applying the inverse noise covariance of the Gaussian.
    noise_std_inv : callable acting on type of data
        Function applying the square root of the inverse noise covariance.

    Notes
    -----
    If `noise_std_inv` is `None` it is inferred by assuming a diagonal noise
    covariance, i.e. by applying it to a vector of ones and taking the square
    root. If both `noise_cov_inv` and `noise_std_inv` are `None`, a unit
    covariance is assumed.
    """
    if not noise_cov_inv and not noise_std_inv:

        def noise_cov_inv(tangents):
            return tangents

        def noise_std_inv(tangents):
            return tangents
    elif not noise_cov_inv:
        wm = (
            "assuming a diagonal covariance matrix"
            ";\nsetting `noise_cov_inv` to"
            " `noise_std_inv(np.ones_like(data))**2`"
        )
        print(wm, file=sys.stderr)
        noise_std_inv_sq = noise_std_inv(tree_map(np.ones_like, data))**2

        def noise_cov_inv(tangents):
            return noise_std_inv_sq * tangents
    elif not noise_std_inv:
        wm = (
            "assuming a diagonal covariance matrix"
            ";\nsetting `noise_std_inv` to"
            " `noise_cov_inv(np.ones_like(data))**0.5`"
        )
        print(wm, file=sys.stderr)
        noise_cov_inv_sqrt = np.sqrt(
            noise_cov_inv(tree_map(np.ones_like, data))
        )

        def noise_std_inv(tangents):
            return noise_cov_inv_sqrt * tangents

    def hamiltonian(primals):
        p_res = primals - data
        return 0.5 * np.sum(p_res * noise_cov_inv(p_res))

    def metric(primals, tangents):
        return noise_cov_inv(tangents)

    def left_sqrt_metric(primals, tangents):
        return noise_std_inv(tangents)

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, data)

    return Likelihood(
        hamiltonian,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


def Poissonian(data, sampling_dtype=float):
    """Computes the negative log-likelihood, i.e. the Hamiltonians of an
    expected count field constrained by Poissonian count data.

    Represents up to an f-independent term :math:`log(d!)`:

    .. math ::
        E(f) = -\\log \\text{Poisson}(d|f) = \\sum f - d^\\dagger \\log(f),

    where f is a field in data space of the expectation values for the counts.

    Parameters
    ----------
    data : ndarray of uint
        Data field with counts. Needs to have integer dtype and all values need
        to be non-negative.
    sampling_dtype : dtype, optional
        Data-type for sampling.
    """
    import numpy as onp

    if isinstance(data, (np.ndarray, onp.ndarray)):
        dtp = data.dtype
    else:
        dtp = onp.common_type(data)
    if not np.issubdtype(dtp, np.integer):
        raise TypeError("`data` of invalid type")
    if np.any(data < 0):
        raise ValueError("`data` may not be negative")

    def hamiltonian(primals):
        return np.sum(primals) - np.vdot(np.log(primals), data)

    def metric(primals, tangents):
        return tangents / primals

    def left_sqrt_metric(primals, tangents):
        return tangents / np.sqrt(primals)

    lsm_tangents_shape = tree_map(_shape_w_fixed_dtype(sampling_dtype), data)

    return Likelihood(
        hamiltonian,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


def VariableCovarianceGaussian(data):
    """Gaussian likelihood of the data with a variable covariance

    Parameters
    ----------
    data : tree-like structure of np.ndarray and float
        Data with additive noise following a Gaussian distribution.

    Notes
    -----
    The likelihood acts on a tuple of (mean, std_inv).
    """
    from .sugar import sum_of_squares

    def hamiltonian(primals):
        """
        primals : pair of (mean, std_inv)
        """
        res = (primals[0] - data) * primals[1]
        return 0.5 * sum_of_squares(res) - np.sum(np.log(primals[1]))

    def metric(primals, tangents):
        """
        primals, tangent : pair of (mean, std_inv)
        """
        prim_std_inv_sq = primals[1]**2
        res = (
            prim_std_inv_sq * tangents[0] ,
            2 * tangents[1] / prim_std_inv_sq
        )
        return type(primals)(res)

    def left_sqrt_metric(primals, tangents):
        """
        primals, tangent : pair of (mean, std_inv)
        """
        res = (
            primals[1] * tangents[0], np.sqrt(2) * tangents[1] / primals[1]
        )
        return type(primals)(res)

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, (data, data))

    return Likelihood(
        hamiltonian,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


def VariableCovarianceStudentT(data, dof):
    """Student's t likelihood of the data with a variable covariance

    Parameters
    ----------
    data : tree-like structure of np.ndarray and float
        Data with additive noise following a Gaussian distribution.
    dof : tree-like structure of np.ndarray and float
        Degree-of-freedom parameter of Student's t distribution.

    Notes
    -----
    The likelihood acts on a tuple of (mean, std).
    """
    def standard_t(nwr, dof):
        return np.sum(np.log1p(nwr**2 / dof) * (dof + 1)) / 2

    def hamiltonian(primals):
        """
        primals : pair of (mean, std)
        """
        t = standard_t((data - primals[0]) / primals[1], dof)
        t += np.sum(np.log(primals[1]))
        return t

    def metric(primals, tangent):
        """
        primals, tangent : pair of (mean, std)
        """
        return (
            tangent[0] * (dof + 1) / (dof + 3) / primals[1]**2,
            tangent[1] * 2 * dof / (dof + 3) / primals[1]**2
        )

    def left_sqrt_metric(primals, tangents):
        """
        primals, tangents : pair of (mean, std)
        """
        cov = (
            (dof + 1) / (dof + 3) / primals[1]**2,
            2 * dof / (dof + 3) / primals[1]**2
        )
        res = (np.sqrt(cov[0]) * tangents[0], np.sqrt(cov[1]) * tangents[1])
        return res

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, (data, data))

    return Likelihood(
        hamiltonian,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


def Categorical(data, axis=-1, sampling_dtype=float):
    """Categorical likelihood of the data, equivalent to cross entropy

    Parameters
    ----------
    data : sequence of int
        An array stating which of the categories is the realized in the data.
        Must agree with the input shape except for the shape[axis]
    axis : int
        Axis over which the categories are formed
    sampling_dtype : dtype, optional
        Data-type for sampling.
    """
    def hamiltonian(primals):
        from jax.nn import log_softmax
        logits = log_softmax(primals, axis=axis)
        return -np.sum(np.take_along_axis(logits, data, axis))

    def metric(primals, tangents):
        from jax.nn import softmax

        preds = softmax(primals, axis=axis)
        norm_term = np.sum(preds * tangents, axis=axis, keepdims=True)
        return preds * tangents - preds * norm_term

    def left_sqrt_metric(primals, tangents):
        from jax.nn import softmax

        sqrtp = np.sqrt(softmax(primals, axis=axis))
        norm_term = np.sum(sqrtp * tangents, axis=axis, keepdims=True)
        return sqrtp * (tangents - sqrtp * norm_term)

    lsm_tangents_shape = tree_map(_shape_w_fixed_dtype(sampling_dtype), data)

    return Likelihood(
        hamiltonian,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )
