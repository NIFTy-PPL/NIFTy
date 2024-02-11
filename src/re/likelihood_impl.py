# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import dataclasses
import operator
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

from jax import numpy as jnp
from jax.tree_util import Partial, tree_map

from .likelihood import Likelihood
from .logger import logger
from .model import LazyModel
from .tree_math import ShapeWithDtype, result_type, sum, vdot


def _standard_t(nwr, dof):
    res = (nwr.conj() * nwr).real / dof
    return sum(tree_map(jnp.log1p, res) * (dof + 1)) / 2


def _shape_w_fixed_dtype(dtype):
    def shp_w_dtp(e):
        return ShapeWithDtype(jnp.shape(e), dtype)

    return shp_w_dtp


@Partial
def _identity(x):
    return x


def _get_cov_inv_and_std_inv(
    cov_inv: Optional[Callable],
    std_inv: Optional[Callable],
    primals=None
) -> Tuple[Union[Partial, LazyModel], Union[Partial, LazyModel]]:
    if cov_inv is None and std_inv is None:
        _cov_inv, _std_inv = _identity, _identity

    if not callable(cov_inv) and cov_inv is not None:
        msg = "assuming the specified inverse covariance is diagonal"
        logger.warning(msg)
        _cov_inv = Partial(operator.mul, cov_inv)
    elif cov_inv is None:
        wm = (
            "assuming a diagonal covariance matrix"
            ";\nsetting `cov_inv` to `std_inv(ones_like(data))**2`"
        )
        logger.warning(wm)
        # Note, `_std_inv` is not properly initialized yet
        si = std_inv if std_inv is not None else _std_inv
        noise_std_inv_sq = si(
            tree_map(jnp.real, tree_map(jnp.ones_like, primals))
        )**2
        _cov_inv = Partial(operator.mul, noise_std_inv_sq)
    else:
        _cov_inv = cov_inv if isinstance(cov_inv,
                                         (Partial,
                                          LazyModel)) else Partial(cov_inv)

    if not callable(std_inv) and std_inv is not None:
        msg = "assuming the specified sqrt of the inverse covariance is diagonal"
        logger.warning(msg)
        _std_inv = Partial(operator.mul, std_inv)
    elif std_inv is None:
        wm = (
            "assuming a diagonal covariance matrix"
            ";\nsetting `std_inv` to `cov_inv(ones_like(data))**0.5`"
        )
        logger.warning(wm)
        noise_cov_inv_sqrt = tree_map(
            jnp.sqrt,
            _cov_inv(tree_map(jnp.real, tree_map(jnp.ones_like, primals)))
        )
        _std_inv = Partial(operator.mul, noise_cov_inv_sqrt)
    else:
        _std_inv = std_inv if isinstance(std_inv,
                                         (Partial,
                                          LazyModel)) else Partial(std_inv)

    assert all(
        isinstance(c, (Partial, LazyModel)) for c in (_cov_inv, _std_inv)
    )
    return _cov_inv, _std_inv


class Gaussian(Likelihood):
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

    See :class:`Likelihood` for details on the properties.
    """
    data: Any = dataclasses.field(metadata=dict(static=False))
    noise_cov_inv: Callable = dataclasses.field(metadata=dict(static=False))
    noise_std_inv: Callable = dataclasses.field(metadata=dict(static=False))

    def __init__(
        self,
        data,
        noise_cov_inv: Optional[Callable] = None,
        noise_std_inv: Optional[Callable] = None
    ):
        self.data = data
        noise_cov_inv, noise_std_inv = _get_cov_inv_and_std_inv(
            noise_cov_inv, noise_std_inv, data
        )
        self.noise_cov_inv = noise_cov_inv
        self.noise_std_inv = noise_std_inv
        shp = tree_map(ShapeWithDtype.from_leave, data)
        super().__init__(domain=shp, lsm_tangents_shape=shp)

    def energy(self, primals):
        p_res = self.data - primals
        return 0.5 * vdot(p_res, self.noise_cov_inv(p_res)).real

    def normalized_residual(self, primals):
        return self.noise_std_inv(self.data - primals)

    def metric(self, primals, tangents):
        return self.noise_cov_inv(tangents)

    def left_sqrt_metric(self, primals, tangents):
        return self.noise_std_inv(tangents)

    def transformation(self, primals):
        return self.noise_std_inv(primals)


class StudentT(Likelihood):
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

    See :class:`Likelihood` for details on the properties.
    """
    data: Any = dataclasses.field(metadata=dict(static=False))
    dof: Any = dataclasses.field(metadata=dict(static=False))
    noise_cov_inv: Callable = dataclasses.field(metadata=dict(static=False))
    noise_std_inv: Callable = dataclasses.field(metadata=dict(static=False))

    def __init__(
        self,
        data,
        dof,
        noise_cov_inv: Optional[Callable] = None,
        noise_std_inv: Optional[Callable] = None,
    ):
        self.data = data
        self.dof = dof
        noise_cov_inv, noise_std_inv = _get_cov_inv_and_std_inv(
            noise_cov_inv, noise_std_inv, data
        )
        self.noise_cov_inv = noise_cov_inv
        self.noise_std_inv = noise_std_inv
        shp = tree_map(ShapeWithDtype.from_leave, data)
        super().__init__(domain=shp, lsm_tangents_shape=shp)

    def energy(self, primals):
        return _standard_t(self.noise_std_inv(self.data - primals), self.dof)

    def metric(self, primals, tangents):
        return self.noise_cov_inv((self.dof + 1) / (self.dof + 3) * tangents)

    def left_sqrt_metric(self, primals, tangents):
        return self.noise_std_inv(
            ((self.dof + 1) / (self.dof + 3))**0.5 * tangents
        )

    def normalized_residual(self, primals):
        return self.left_sqrt_metric(None, self.data - primals)

    def transformation(self, primals):
        return self.noise_std_inv(
            ((self.dof + 1) / (self.dof + 3))**0.5 * primals
        )


class Poissonian(Likelihood):
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


    See :class:`Likelihood` for details on the properties.
    """
    data: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, data, sampling_dtype=float):
        dtp = result_type(data)
        if not jnp.issubdtype(dtp, jnp.integer):
            raise TypeError("`data` of invalid type")
        if sum(tree_map(lambda x: jnp.any(x < 0), data)):
            raise ValueError("`data` must not be negative")
        self.data = data
        shp = tree_map(_shape_w_fixed_dtype(sampling_dtype), data)
        super().__init__(domain=shp, lsm_tangents_shape=shp)

    def energy(self, primals):
        return sum(primals) - vdot(tree_map(jnp.log, primals), self.data)

    def metric(self, primals, tangents):
        return tangents / primals

    def left_sqrt_metric(self, primals, tangents):
        return tangents / primals**0.5

    def normalized_residual(self, primals):
        return self.left_sqrt_metric(primals, self.data - primals)

    def transformation(self, primals):
        return 2. * primals**0.5


class VariableCovarianceGaussian(Likelihood):
    """Gaussian likelihood of the data with a variable covariance

    Parameters
    ----------
    data : tree-like structure of jnp.ndarray and float
        Data with additive noise following a Gaussian distribution.
    iscomplex: Boolean, optional
        Whether the parameters are complex-valued.

    Notes
    -----
    **The likelihood acts on a tuple of (mean, std_inv)**.

    See :class:`Likelihood` for details on the properties.
    """
    data: Any = dataclasses.field(metadata=dict(static=False))
    iscomplex: bool = False

    def __init__(self, data, iscomplex=False):
        # TODO: make configurable whether `std_inv` or `std` is passed
        self.data = data
        self.iscomplex = iscomplex
        shp = tree_map(ShapeWithDtype.from_leave, (data, data.real))
        super().__init__(domain=shp, lsm_tangents_shape=shp)

    def energy(self, primals):
        res = (self.data - primals[0]) * primals[1]
        fct = 1 + self.iscomplex
        return 0.5 * vdot(res,
                          res).real - fct * sum(tree_map(jnp.log, primals[1]))

    def metric(self, primals, tangents):
        fct = 2 * (1 + self.iscomplex)
        prim_std_inv_sq = primals[1]**2
        res = (
            prim_std_inv_sq * tangents[0], fct * tangents[1] / prim_std_inv_sq
        )
        return type(primals)(res)

    def left_sqrt_metric(self, primals, tangents):
        fct = jnp.sqrt(2)**(1 + self.iscomplex)
        res = (primals[1] * tangents[0], fct * tangents[1] / primals[1])
        return type(primals)(res)

    def transformation(self, primals):
        """
        Notes
        -----
        A global transformation to Euclidean space does not exist. A local
        approximation invoking the residual is used instead.
        """
        # TODO: test by drawing synthetic data that actually follows the
        # noise-cov and then average over it
        fct = 1 + self.iscomplex
        res = (
            primals[1] * (primals[0] - self.data),
            fct * tree_map(jnp.log, primals[1])
        )
        return type(primals)(res)

    def normalized_residual(self, primals):
        return (self.data - primals[0]) * primals[1]


class VariableCovarianceStudentT(Likelihood):
    """Student's t likelihood of the data with a variable covariance

    Parameters
    ----------
    data : tree-like structure of jnp.ndarray and float
        Data with additive noise following a Gaussian distribution.
    dof : tree-like structure of jnp.ndarray and float
        Degree-of-freedom parameter of Student's t distribution.

    Notes
    -----
    **The likelihood acts on a tuple of (mean, std)**.

    See :class:`Likelihood` for details on the properties.
    """
    data: Any = dataclasses.field(metadata=dict(static=False))
    dof: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, data, dof):
        # TODO: make configurable whether `std_inv` or `std` is passed
        self.data = data
        self.dof = dof
        shp = tree_map(ShapeWithDtype.from_leave, (data, data))
        super().__init__(domain=shp, lsm_tangents_shape=shp)

    def energy(self, primals):
        t = _standard_t((self.data - primals[0]) / primals[1], self.dof)
        t += sum(tree_map(jnp.log, primals[1]))
        return t

    def metric(self, primals, tangent):
        res = (
            tangent[0] * (self.dof + 1) / (self.dof + 3) / primals[1]**2,
            tangent[1] * 2 * self.dof / (self.dof + 3) / primals[1]**2
        )
        return type(primals)(res)

    def left_sqrt_metric(self, primals, tangents):
        cov = (
            (self.dof + 1) / (self.dof + 3) / primals[1]**2,
            2 * self.dof / (self.dof + 3) / primals[1]**2
        )
        res = (cov[0]**0.5 * tangents[0], cov[1]**0.5 * tangents[1])
        return type(primals)(res)

    def normalized_residual(self, primals):
        return (self.data - primals[0]
               ) / primals[1] * ((self.dof + 1) / (self.dof + 3))**0.5


class Categorical(Likelihood):
    """Categorical likelihood of the data, equivalent to cross entropy

    Parameters
    ----------
    data : tree-like structure of jnp.ndarray and int
        Which of the categories is the realized in the data. Must agree with the
        input shape except for the shape[axis] of the leafs
    axis : int
        Leaf-axis over which the categories are formed
    sampling_dtype : dtype, optional
        Data-type for sampling.


    See :class:`Likelihood` for details on the properties.
    """
    data: Any = dataclasses.field(metadata=dict(static=False))
    axis: int = -1

    def __init__(self, data, axis=-1, sampling_dtype=float):
        self.data = data
        self.axis = axis
        shp = tree_map(_shape_w_fixed_dtype(sampling_dtype), data)
        super().__init__(domain=shp, lsm_tangents_shape=shp)

    def energy(self, primals):
        from jax.nn import log_softmax

        def eval(p, d):
            logits = log_softmax(p, axis=self.axis)
            return -jnp.sum(jnp.take_along_axis(logits, d, self.axis))

        return sum(tree_map(eval, primals, self.data))

    def metric(self, primals, tangents):
        from jax.nn import softmax

        preds = tree_map(partial(softmax, axis=self.axis), primals)
        norm_term = tree_map(
            partial(jnp.sum, axis=self.axis, keepdims=True), preds * tangents
        )
        return preds * tangents - preds * sum(norm_term)

    def left_sqrt_metric(self, primals, tangents):
        from jax.nn import softmax

        # FIXME: not sure if this is really the square root
        sqrtp = tree_map(partial(softmax, axis=self.axis), primals)**0.5
        norm_term = tree_map(
            partial(jnp.sum, axis=self.axis, keepdims=True), sqrtp * tangents
        )
        norm_term = sum(norm_term)
        return sqrtp * (tangents - sqrtp * norm_term)
