# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from typing import Any, Callable, Optional, TypeVar, Union

from jax import linear_transpose, linearize
from jax import numpy as jnp
from jax import vjp
from jax.tree_util import Partial, tree_leaves

from .tree_math import ShapeWithDtype
from .misc import doc_from, is1d, isiterable, split, sum_of_squares

Q = TypeVar("Q")


class Likelihood():
    """Storage class for keeping track of the energy, the associated
    left-square-root of the metric and the metric.
    """
    def __init__(
        self,
        energy: Callable[..., Union[jnp.ndarray, float]],
        transformation: Optional[Callable[[Q], Any]] = None,
        left_sqrt_metric: Optional[Callable[[Q, Q], Any]] = None,
        metric: Optional[Callable[[Q, Q], Any]] = None,
        lsm_tangents_shape=None
    ):
        """Instantiates a new likelihood.

        Parameters
        ----------
        energy : callable
            Function evaluating the negative log-likelihood.
        transformation : callable, optional
            Function evaluating the geometric transformation of the likelihood.
        left_sqrt_metric : callable, optional
            Function applying the left-square-root of the metric.
        metric : callable, optional
            Function applying the metric.
        lsm_tangents_shape : tree-like structure of ShapeWithDtype, optional
            Structure of the data space.
        """
        # TODO: track forward model and build lsm, metric only when called
        # instead of always partially
        self._hamiltonian = energy
        self._transformation = transformation
        self._left_sqrt_metric = left_sqrt_metric
        self._metric = metric

        if lsm_tangents_shape is not None:
            leaves = tree_leaves(lsm_tangents_shape)
            if not all(
                hasattr(e, "shape") and hasattr(e, "dtype") for e in leaves
            ):
                if is1d(lsm_tangents_shape
                       ) or not isiterable(lsm_tangents_shape):
                    lsm_tangents_shape = ShapeWithDtype(lsm_tangents_shape)
                else:
                    te = "`lsm_tangent_shapes` of invalid type"
                    raise TypeError(te)
        self._lsm_tan_shp = lsm_tangents_shape

    def __call__(self, primals, **primals_kw):
        """Convenience method to access the `energy` method of this instance.
        """
        return self.energy(primals, **primals_kw)

    def energy(self, primals, **primals_kw):
        """Applies the energy to `primals`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the energy.
        **primals_kw : Any
           Additional arguments passed on to the energy.

        Returns
        -------
        energy : float
            Energy at the position `primals`.
        """
        return self._hamiltonian(primals, **primals_kw)

    def metric(self, primals, tangents, **primals_kw):
        """Applies the metric at `primals` to `tangents`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the metric.
        tangents : tree-like structure
            Instance to which to apply the metric.
        **primals_kw : Any
           Additional arguments passed on to the metric.

        Returns
        -------
        naturally_curved : tree-like structure
            Tree-like structure of the same type as primals to which the metric
            has been applied to.
        """
        if self._metric is None:
            from jax import linear_transpose

            lsm_at_p = Partial(self.left_sqrt_metric, primals, **primals_kw)
            rsm_at_p = linear_transpose(
                lsm_at_p, self.left_sqrt_metric_tangents_shape
            )
            res = lsm_at_p(*rsm_at_p(tangents))
            return res
        return self._metric(primals, tangents, **primals_kw)

    def left_sqrt_metric(self, primals, tangents, **primals_kw):
        """Applies the left-square-root of the metric at `primals` to
        `tangents`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the metric.
        tangents : tree-like structure
            Instance to which to apply the metric.
        **primals_kw : Any
           Additional arguments passed on to the LSM.

        Returns
        -------
        metric_sample : tree-like structure
            Tree-like structure of the same type as primals to which the
            left-square-root of the metric has been applied to.
        """
        if self._left_sqrt_metric is None:
            _, bwd = vjp(Partial(self.transformation, **primals_kw), primals)
            res = bwd(tangents)
            return res[0]
        return self._left_sqrt_metric(primals, tangents, **primals_kw)

    def transformation(self, primals, **primals_kw):
        """Applies the coordinate transformation that maps into a coordinate
        system in which the metric of the likelihood is the Euclidean metric.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to transform.
        **primals_kw : Any
           Additional arguments passed on to the transformation.

        Returns
        -------
        transformed_sample : tree-like structure
            Structure of the same type as primals to which the geometric
            transformation has been applied to.
        """
        if self._transformation is None:
            nie = "`transformation` is not implemented"
            raise NotImplementedError(nie)
        return self._transformation(primals, **primals_kw)

    @property
    def left_sqrt_metric_tangents_shape(self):
        """Retrieves the shape of the tangent domain of the
        left-square-root of the metric.
        """
        return self._lsm_tan_shp

    @property
    def lsm_tangents_shape(self):
        # TODO: track domain and infer LSM tan shape from it and LSM
        """Alias for `left_sqrt_metric_tangents_shape`."""
        return self.left_sqrt_metric_tangents_shape

    def new(
        self, energy: Callable, transformation: Optional[Callable],
        left_sqrt_metric: Optional[Callable], metric: Optional[Callable]
    ):
        """Instantiates a new likelihood with the same `lsm_tangents_shape`.

        Parameters
        ----------
        energy : callable
            Function evaluating the negative log-likelihood.
        transformation : callable, optional
            Function evaluating the geometric transformation of the
            log-likelihood.
        left_sqrt_metric : callable, optional
            Function applying the left-square-root of the metric.
        metric : callable, optional
            Function applying the metric.
        """
        return Likelihood(
            energy,
            transformation=transformation,
            left_sqrt_metric=left_sqrt_metric,
            metric=metric,
            lsm_tangents_shape=self._lsm_tan_shp
        )

    def jit(self, **kwargs):
        """Returns a new likelihood with jit-compiled energy, left-square-root
        of metric and metric.
        """
        from jax import jit

        if self._transformation is not None:
            j_trafo = jit(self.transformation, **kwargs)
            j_lsm = jit(self.left_sqrt_metric, **kwargs)
            j_m = jit(self.metric, **kwargs)
        elif self._left_sqrt_metric is not None:
            j_trafo = None
            j_lsm = jit(self.left_sqrt_metric, **kwargs)
            j_m = jit(self.metric, **kwargs)
        elif self._metric is not None:
            j_trafo, j_lsm = None, None
            j_m = jit(self.metric, **kwargs)
        else:
            j_trafo, j_lsm, j_m = None, None, None

        return self.new(
            jit(self._hamiltonian, **kwargs),
            transformation=j_trafo,
            left_sqrt_metric=j_lsm,
            metric=j_m
        )

    def __matmul__(self, f: Callable):
        return self.matmul(f, left_argnames=(), right_argnames=None)

    def matmul(self, f: Callable, left_argnames=(), right_argnames=None):
        """Amend the function `f` to the right of the likelihood.

        Parameters
        ----------
        f : Callable
            Function which to amend to the likelihood.
        left_argnames : tuple or None
            Keys of the keyword arguments of the joined likelihood which
            to pass to the original likelihood. Passing `None` indicates
            the intent to absorb everything not explicitly absorbed by
            the other call.
        right_argnames : tuple or None
            Keys of the keyword arguments of the joined likelihood which
            to pass to the amended function. Passing `None` indicates
            the intent to absorb everything not explicitly absorbed by
            the other call.

        Returns
        -------
        lh : Likelihood
        """
        if (left_argnames is None and right_argnames is None) or \
        (left_argnames is not None and right_argnames is not None):
            ve = "only one of `left_argnames` and `right_argnames` can be (not) `None`"
            raise ValueError(ve)

        def split_kwargs(**kwargs):
            if left_argnames is None:  # right_argnames must be not None
                right_kw, left_kw = split(kwargs, right_argnames)
            else:  # right_argnames must be None
                left_kw, right_kw = split(kwargs, left_argnames)
            return left_kw, right_kw

        def energy_at_f(primals, **primals_kw):
            kw_l, kw_r = split_kwargs(**primals_kw)
            return self.energy(f(primals, **kw_r), **kw_l)

        def transformation_at_f(primals, **primals_kw):
            kw_l, kw_r = split_kwargs(**primals_kw)
            return self.transformation(f(primals, **kw_r), **kw_l)

        def metric_at_f(primals, tangents, **primals_kw):
            kw_l, kw_r = split_kwargs(**primals_kw)
            # Note, judging by a simple benchmark on a large problem,
            # transposing the JVP seems faster than computing the VJP again. On
            # small problems there seems to be no measurable difference.
            y, fwd = linearize(Partial(f, **kw_r), primals)
            bwd = linear_transpose(fwd, primals)
            return bwd(self.metric(y, fwd(tangents), **kw_l))[0]

        def left_sqrt_metric_at_f(primals, tangents, **primals_kw):
            kw_l, kw_r = split_kwargs(**primals_kw)
            y, bwd = vjp(Partial(f, **kw_r), primals)
            left_at_fp = self.left_sqrt_metric(y, tangents, **kw_l)
            return bwd(left_at_fp)[0]

        return self.new(
            energy_at_f,
            transformation=transformation_at_f,
            left_sqrt_metric=left_sqrt_metric_at_f,
            metric=metric_at_f
        )

    def __add__(self, other):
        if not isinstance(other, Likelihood):
            te = (
                "object which to add to this instance is of invalid type"
                f" {type(other)!r}"
            )
            raise TypeError(te)

        def joined_hamiltonian(p, **pkw):
            return self.energy(p, **pkw) + other.energy(p, **pkw)

        def joined_metric(p, t, **pkw):
            return self.metric(p, t, **pkw) + other.metric(p, t, **pkw)

        joined_tangents_shape = {
            "lh_left": self._lsm_tan_shp,
            "lh_right": other._lsm_tan_shp
        }

        def joined_transformation(p, **pkw):
            from warnings import warn

            # FIXME
            warn("adding transformations is untested", UserWarning)
            return {
                "lh_left": self.transformation(p, **pkw),
                "lh_right": other.transformation(p, **pkw)
            }

        def joined_left_sqrt_metric(p, t, **pkw):
            return self.left_sqrt_metric(
                p, t["lh_left"], **pkw
            ) + other.left_sqrt_metric(p, t["lh_right"], **pkw)

        return Likelihood(
            joined_hamiltonian,
            transformation=joined_transformation,
            left_sqrt_metric=joined_left_sqrt_metric,
            metric=joined_metric,
            lsm_tangents_shape=joined_tangents_shape
        )


# TODO: prune/hide/(make simply add unit mat) in favor of just passing around
# likelihood; we exclusively built hierarchical models anyways.
class StandardHamiltonian():
    """Joined object storage composed of a user-defined likelihood and a
    standard normal likelihood as prior.
    """
    def __init__(
        self,
        likelihood: Likelihood,
        _compile_joined: bool = False,
        _compile_kwargs: dict = {}
    ):
        """Instantiates a new standardized Hamiltonian, i.e. a likelihood
        joined with a standard normal prior.

        Parameters
        ----------
        likelihood : Likelihood
            Energy, left-square-root of metric and metric of the likelihood.
        """
        self._lh = likelihood

        def joined_hamiltonian(primals, **primals_kw):
            # Assume the first primals to be the parameters
            return self._lh(primals, **
                            primals_kw) + 0.5 * sum_of_squares(primals)

        def joined_metric(primals, tangents, **primals_kw):
            return self._lh.metric(primals, tangents, **primals_kw) + tangents

        if _compile_joined:
            from jax import jit
            joined_hamiltonian = jit(joined_hamiltonian, **_compile_kwargs)
            joined_metric = jit(joined_metric, **_compile_kwargs)
        self._hamiltonian = joined_hamiltonian
        self._metric = joined_metric

    @doc_from(Likelihood.__call__)
    def __call__(self, primals, **primals_kw):
        return self.energy(primals, **primals_kw)

    @doc_from(Likelihood.energy)
    def energy(self, primals, **primals_kw):
        return self._hamiltonian(primals, **primals_kw)

    @doc_from(Likelihood.metric)
    def metric(self, primals, tangents, **primals_kw):
        return self._metric(primals, tangents, **primals_kw)

    @property
    def likelihood(self):
        return self._lh

    def jit(self, **kwargs):
        return StandardHamiltonian(
            self.likelihood.jit(**kwargs),
            _compile_joined=True,
            _compile_kwargs=kwargs
        )
