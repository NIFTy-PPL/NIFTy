# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from typing import Callable, Optional, Tuple

from jax import numpy as jnp
from jax.tree_util import tree_map

from .tree_math import ShapeWithDtype
from .likelihood import Likelihood
from .logger import logger
from .tree_math import vdot, tree_reduce, Vector


def standard_t(nwr, dof):
    return jnp.sum(jnp.log1p((nwr.conj() * nwr).real / dof) * (dof + 1)) / 2


def _shape_w_fixed_dtype(dtype):
    def shp_w_dtp(e):
        return ShapeWithDtype(jnp.shape(e), dtype)

    return shp_w_dtp


def _get_cov_inv_and_std_inv(
    cov_inv: Optional[Callable],
    std_inv: Optional[Callable],
    primals=None
) -> Tuple[Callable, Callable]:
    if not cov_inv and not std_inv:

        def cov_inv(tangents):
            return tangents

        def std_inv(tangents):
            return tangents

    elif not cov_inv:
        wm = (
            "assuming a diagonal covariance matrix"
            ";\nsetting `cov_inv` to `std_inv(jnp.ones_like(data))**2`"
        )
        logger.warning(wm)
        noise_std_inv_sq = std_inv(
            tree_map(jnp.real, tree_map(jnp.ones_like, primals))
        )**2

        def cov_inv(tangents):
            return noise_std_inv_sq * tangents

    elif not std_inv:
        wm = (
            "assuming a diagonal covariance matrix"
            ";\nsetting `std_inv` to `cov_inv(jnp.ones_like(data))**0.5`"
        )
        logger.warning(wm)
        noise_cov_inv_sqrt = tree_map(
            jnp.sqrt,
            cov_inv(tree_map(jnp.real, tree_map(jnp.ones_like, primals)))
        )

        def std_inv(tangents):
            return noise_cov_inv_sqrt * tangents

    if not (callable(cov_inv) and callable(std_inv)):
        raise ValueError("received un-callable input")
    return cov_inv, std_inv


def Gaussian(
    data,
    noise_cov_inv: Optional[Callable] = None,
    noise_std_inv: Optional[Callable] = None
):
    """Gaussian likelihood of the data

    Parameters
    ----------
    data : tree-like structure of jnp.ndarray and float
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
    noise_cov_inv, noise_std_inv = _get_cov_inv_and_std_inv(
        noise_cov_inv, noise_std_inv, data
    )

    def normalized_residual(primals):
        return noise_std_inv(data - primals)

    def hamiltonian(primals):
        p_res = data - primals
        return 0.5 * vdot(p_res, noise_cov_inv(p_res)).real

    def metric(primals, tangents):
        return noise_cov_inv(tangents)

    def left_sqrt_metric(primals, tangents):
        return noise_std_inv(tangents)

    def transformation(primals):
        return noise_std_inv(primals)

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, data)

    return Likelihood(
        hamiltonian,
        normalized_residual=normalized_residual,
        transformation=transformation,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


def StudentT(
    data,
    dof,
    noise_cov_inv: Optional[Callable] = None,
    noise_std_inv: Optional[Callable] = None
):
    """Student's t likelihood of the data

    Parameters
    ----------
    data : tree-like structure of jnp.ndarray and float
        Data with additive noise following a Gaussian distribution.
    dof : tree-like structure of jnp.ndarray and float
        Degree-of-freedom parameter of Student's t distribution.
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
    noise_cov_inv, noise_std_inv = _get_cov_inv_and_std_inv(
        noise_cov_inv, noise_std_inv, data
    )

    def hamiltonian(primals):
        """
        primals : mean
        """
        return standard_t(noise_std_inv(data - primals), dof)

    def metric(primals, tangents):
        """
        primals, tangent : mean
        """
        return noise_cov_inv((dof + 1) / (dof + 3) * tangents)

    def left_sqrt_metric(primals, tangents):
        """
        primals, tangents : mean
        """
        return noise_std_inv(jnp.sqrt((dof + 1) / (dof + 3)) * tangents)

    def normalized_residual(primals):
        """
        primals : mean
        """
        return left_sqrt_metric(None, data - primals)

    def transformation(primals):
        """
        primals : mean
        """
        return noise_std_inv(jnp.sqrt((dof + 1) / (dof + 3)) * primals)

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, data)

    return Likelihood(
        hamiltonian,
        normalized_residual=normalized_residual,
        transformation=transformation,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


def Poissonian(data, sampling_dtype=float):
    """Computes the negative log-likelihood, i.e. the Hamiltonians of an
    expected count Vector constrained by Poissonian count data.

    Represents up to an f-independent term :math:`log(d!)`:

    .. math ::
        E(f) = -\\log \\text{Poisson}(d|f) = \\sum f - d^\\dagger \\log(f),

    where f is a Vector in data space of the expectation values for the counts.

    Parameters
    ----------
    data : jnp.ndarray or tree-like structure of jnp.ndarray and float
        Data Vector with counts. Needs to have integer dtype and all values need
        to be non-negative.
    sampling_dtype : dtype, optional
        Data-type for sampling.
    """
    from .tree_math import result_type

    dtp = result_type(data)
    if not jnp.issubdtype(dtp, jnp.integer):
        raise TypeError("`data` of invalid type")
    if isinstance(data, jnp.ndarray):
        if jnp.any(data < 0):
            raise ValueError("`data` may not be negative")
    elif isinstance(data, Vector):
        smaller_zero = lambda x: jnp.any(x < 0)
        if any(value is False for value in tree_map(smaller_zero, data).tree.values()):
            raise ValueError("`data` may not be negative")
    else:
        raise NotImplementedError('Data needs to be a jnp.array or have a tree-like structure of jnp.arrays')

    def hamiltonian(primals):
        if isinstance(primals, jnp.ndarray) and isinstance(data, jnp.ndarray):
            ham = jnp.sum(primals) - vdot(jnp.log(primals), data)
        elif isinstance(primals, Vector) and isinstance(data, Vector):
            children_lklhd = tree_map(jnp.sum, primals) - vdot(tree_map(jnp.log, primals), data)
            ham = tree_reduce(jnp.add, children_lklhd)
        else:
            raise NotImplementedError('Data and primals need to have the same type and need to be either an'
                                      'array or have a tree-like structure.')
        return ham

    def metric(primals, tangents):
        return tangents / primals

    def left_sqrt_metric(primals, tangents):
        if isinstance(primals, jnp.ndarray) and isinstance(tangents, jnp.ndarray):
            lsm = tangents / jnp.sqrt(primals)
        elif isinstance(primals, Vector) and isinstance(tangents, Vector):
            lsm = tangents / tree_map(jnp.sqrt, primals)
        else:
            raise NotImplementedError('Tangents and primals need to have the same type and need to be either an'
                                      'array or have a tree-like structure.')
        return lsm

    def normalized_residual(primals):
        return left_sqrt_metric(primals, data - primals)

    def transformation(primals):
        func = lambda x: jnp.sqrt(x) * 2.
        if isinstance(primals, jnp.ndarray):
            trans = func(primals)
        elif isinstance(primals, Vector):
            trans = tree_map(func, primals)
        else:
            raise NotImplementedError('Primals need to be either an array or have a tree-like structure.')
        return trans

    lsm_tangents_shape = tree_map(_shape_w_fixed_dtype(sampling_dtype), data)

    return Likelihood(
        hamiltonian,
        normalized_residual=normalized_residual,
        transformation=transformation,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


def VariableCovarianceGaussian(data, iscomplex=False):
    """Gaussian likelihood of the data with a variable covariance

    Parameters
    ----------
    data : tree-like structure of jnp.ndarray and float
        Data with additive noise following a Gaussian distribution.
    iscomplex: Boolean
        Whether the parameters are complex-valued.

    Notes
    -----
    The likelihood acts on a tuple of (mean, std_inv).
    """

    # TODO: make configurable whether `std_inv` or `std` is passed

    def normalized_residual(primals):
        return (data - primals[0]) * primals[1]

    def hamiltonian(primals):
        """
        primals : pair of (mean, std_inv)
        """
        res = (data - primals[0]) * primals[1]
        fct = 2 if iscomplex else 1
        return 0.5 * vdot(res, res).real - fct * jnp.sum(jnp.log(primals[1]))

    def metric(primals, tangents):
        """
        primals, tangent : pair of (mean, std_inv)
        """
        fct = 4 if iscomplex else 2
        prim_std_inv_sq = primals[1]**2
        res = (
            prim_std_inv_sq * tangents[0], fct * tangents[1] / prim_std_inv_sq
        )
        return type(primals)(res)

    def left_sqrt_metric(primals, tangents):
        """
        primals, tangent : pair of (mean, std_inv)
        """
        fct = 2 if iscomplex else jnp.sqrt(2)
        res = (primals[1] * tangents[0], fct * tangents[1] / primals[1])
        return type(primals)(res)

    def transformation(primals):
        """
        pirmals : pair of (mean, std_inv)

        Notes
        -----
        A global transformation to Euclidean space does not exist. A local
        approximation invoking the residual is used instead.
        """
        # TODO: test by drawing synthetic data that actually follows the
        # noise-cov and then average over it
        fct = 2 if iscomplex else 1
        res = (
            primals[1] * (primals[0] - data),
            fct * tree_map(jnp.log, primals[1])
        )
        return type(primals)(res)

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, (data, data.real))

    return Likelihood(
        hamiltonian,
        normalized_residual=normalized_residual,
        transformation=transformation,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


def VariableCovarianceStudentT(data, dof):
    """Student's t likelihood of the data with a variable covariance

    Parameters
    ----------
    data : tree-like structure of jnp.ndarray and float
        Data with additive noise following a Gaussian distribution.
    dof : tree-like structure of jnp.ndarray and float
        Degree-of-freedom parameter of Student's t distribution.

    Notes
    -----
    The likelihood acts on a tuple of (mean, std).
    """

    # TODO: make configurable whether `std_inv` or `std` is passed
    def hamiltonian(primals):
        """
        primals : pair of (mean, std)
        """
        t = standard_t((data - primals[0]) / primals[1], dof)
        t += jnp.sum(jnp.log(primals[1]))
        return t

    def metric(primals, tangent):
        """
        primals, tangent : pair of (mean, std)
        """
        return (
            tangent[0] * (dof + 1) / (dof + 3) / primals[1]**2,
            tangent[1] * 2 * dof / (dof + 3) / primals[1]**2
        )

    def normalized_residual(primals):
        return (data - primals[0]) / primals[1] * ((dof + 1) / (dof + 3))**0.5

    def left_sqrt_metric(primals, tangents):
        """
        primals, tangents : pair of (mean, std)
        """
        cov = (
            (dof + 1) / (dof + 3) / primals[1]**2,
            2 * dof / (dof + 3) / primals[1]**2
        )
        res = (jnp.sqrt(cov[0]) * tangents[0], jnp.sqrt(cov[1]) * tangents[1])
        return res

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, (data, data))

    return Likelihood(
        hamiltonian,
        normalized_residual=normalized_residual,
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
        return -jnp.sum(jnp.take_along_axis(logits, data, axis))

    def metric(primals, tangents):
        from jax.nn import softmax

        preds = softmax(primals, axis=axis)
        norm_term = jnp.sum(preds * tangents, axis=axis, keepdims=True)
        return preds * tangents - preds * norm_term

    def left_sqrt_metric(primals, tangents):
        from jax.nn import softmax

        sqrtp = jnp.sqrt(softmax(primals, axis=axis))
        norm_term = jnp.sum(sqrtp * tangents, axis=axis, keepdims=True)
        return sqrtp * (tangents - sqrtp * norm_term)

    lsm_tangents_shape = tree_map(_shape_w_fixed_dtype(sampling_dtype), data)

    return Likelihood(
        hamiltonian,
        normalized_residual = None,
        left_sqrt_metric=left_sqrt_metric,
        metric=metric,
        lsm_tangents_shape=lsm_tangents_shape
    )
