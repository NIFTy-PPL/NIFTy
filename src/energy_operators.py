from typing import Optional

from jax import numpy as np
from jax.tree_util import tree_map

from .likelihood import Likelihood, ShapeWithDtype


def Gaussian(
    data,
    noise_cov_inv: Optional[callable] = None,
    noise_std_inv: Optional[callable] = None
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
    elif not noise_std_inv:
        wm = (
            "assuming a diagonal covariance matrix"
            ";\nsetting `noise_std_inv` to"
            " `noise_cov_inv(np.ones_like(data))**0.5`"
        )
        import sys
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


def Categorical(data, axis=-1):
    """Categorical likelihood of the data, equivalent to cross entropy

    Parameters
    ----------
    data: sequence of int
        An array stating which of the categories is the realized in the data.
        Must agree with the input shape except for the shape[axis]
    axis: int
        Axis over which the categories are formed
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

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, data)

    return Likelihood(
        hamiltonian,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )
