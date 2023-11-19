# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from typing import Callable, Optional, TypeVar, Union

import jax
from jax import numpy as jnp
from jax.tree_util import (
    Partial, tree_flatten, tree_leaves, tree_map, tree_structure, tree_unflatten
)

from .misc import doc_from, is1d, isiterable, split
from .model import AbstractModel
from .tree_math import ShapeWithDtype, Vector, conj, vdot

Q = TypeVar("Q")
P = TypeVar("P")


def _functional_conj(func):
    def func_conj(*args, **kwargs):
        # func^*(x) = (func(x^*))^*
        return conj(func(*conj(args), **conj(kwargs)))

    return func_conj


def _parse_point_estimates(point_estimates, primals):
    if isinstance(point_estimates, (tuple, list)):
        if not isinstance(primals, (Vector, dict)):
            te = "tuple-shortcut point-estimate only availble for dict/Vector "
            te += "type primals"
            raise TypeError(te)
        pe = tree_map(lambda x: False, primals)
        pe = pe.tree if isinstance(primals, Vector) else pe
        for k in point_estimates:
            pe[k] = True
        point_estimates = Vector(pe) if isinstance(primals, Vector) else pe
    if tree_structure(primals) != tree_structure(point_estimates):
        print(primals)
        print(point_estimates)
        te = "`primals` and `point_estimates` pytree structre do no match"
        raise TypeError(te)

    primals_liquid, primals_frozen = [], []
    for p, ep in zip(tree_leaves(primals), tree_leaves(point_estimates)):
        if ep:
            primals_frozen.append(p)
        else:
            primals_liquid.append(p)
    primals_liquid = Vector(tuple(primals_liquid))
    primals_frozen = tuple(primals_frozen)
    return point_estimates, primals_liquid, primals_frozen


def _partial_argument(call, insert_axes, flat_fill):
    """For every non-None value in `insert_axes`, amend the value of `flat_fill`
    at the same position to the argument. Both `insert_axes` and `flat_fill` are
    w.r.t. the whole input argument tuple `arg` of `call(*args)`.
    """
    if not flat_fill and not insert_axes:
        return call

    if len(insert_axes) != len(flat_fill):
        ve = "`insert_axes` and `flat_fill` must be of equal length"
        raise ValueError(ve)
    for iae, ffe in zip(insert_axes, flat_fill):
        if iae is not None and ffe is not None:
            if not isinstance(ffe, (tuple, list)):
                te = (
                    f"`flat_fill` must be a tuple of flattened pytrees;"
                    f" got '{flat_fill!r}'"
                )
                raise TypeError(te)
            iae_leaves = tree_leaves(iae)
            if not all(isinstance(e, bool) for e in iae_leaves):
                te = "leaves of `insert_axes` elements must all be boolean"
                raise TypeError(te)
            if sum(iae_leaves) != len(ffe):
                ve = "more inserts in `insert_axes` than elements in `flat_fill`"
                raise ValueError(ve)
        elif iae is not None or ffe is not None:
            ve = (
                "both `insert_axes` and `flat_fill` must be `None` at the same"
                " positions"
            )
            raise ValueError(ve)
    # NOTE, `tree_flatten` replaces `None`s with list of zero length
    insert_axes, in_axes_td = zip(*(tree_flatten(ia) for ia in insert_axes))

    def insert(*x):
        y = []
        assert len(x) == len(insert_axes) == len(flat_fill) == len(in_axes_td)
        for xe, iae, ffe, iatde in zip(x, insert_axes, flat_fill, in_axes_td):
            if ffe is None and not iae:
                y.append(xe)
                continue
            assert iae and ffe is not None
            assert sum(iae) == len(ffe)
            xe, ffe = list(tree_leaves(xe)), list(ffe)
            ye = [xe.pop(0) if not cond else ffe.pop(0) for cond in iae]
            # for cond in iae:
            #     ye.append(xe.pop(0) if not cond else ffe.pop(0))
            y.append(tree_unflatten(iatde, ye))
        return tuple(y)

    def partially_inserted_call(*x):
        return call(*insert(*x))

    return partially_inserted_call


def partial_insert_and_remove(
    call, insert_axes, flat_fill, *, remove_axes=(), unflatten=None
):
    """Return a call in which `flat_fill` is inserted into arguments of `call`
    at `inset_axes` and subsequently removed from its output at `remove_axes`.
    """
    call = _partial_argument(call, insert_axes=insert_axes, flat_fill=flat_fill)

    if not remove_axes:
        return call

    remove_axes = tree_leaves(remove_axes)
    if not all(isinstance(e, bool) for e in remove_axes):
        raise TypeError("leaves of `remove_axes` must all be boolean")

    def remove(x):
        x, y = list(tree_leaves(x)), []
        if tree_structure(x) != tree_structure(remove_axes):
            te = (
                f"`remove_axes` ({tree_structure(remove_axes)!r}) is shaped"
                f" differently than output of `call` ({tree_structure(x)!r})"
            )
            raise TypeError(te)
        for maybe_remove, cond in zip(x, remove_axes):
            if not cond:
                y.append(maybe_remove)
        y = unflatten(tuple(y)) if unflatten is not None else y
        return y

    def partially_removed_call(*x):
        return remove(call(*x))

    return partially_removed_call


class Likelihood(AbstractModel):
    """Storage class for keeping track of the energy, the associated
    left-square-root of the metric and the metric.
    """
    def __init__(
        self,
        energy: Callable[..., Union[jnp.ndarray, float]],
        normalized_residual: Optional[Callable[[Q], P]] = None,
        transformation: Optional[Callable[[Q], P]] = None,
        left_sqrt_metric: Optional[Callable[[Q, Q], P]] = None,
        metric: Optional[Callable[[Q, Q], Q]] = None,
        domain=None,
        lsm_tangents_shape=None
    ):
        """Instantiates a new likelihood.

        Parameters
        ----------
        energy : callable
            Function evaluating the negative log-likelihood.
        transformation : callable, optional
            Function evaluating the geometric transformation of the likelihood.
        normalized_residual : callable, optional
            Function evaluating the data residual normalized by the standard
            deviation of the likelihood.
        left_sqrt_metric : callable, optional
            Function applying the left-square-root of the metric.
        metric : callable, optional
            Function applying the metric.
        domain : tree-like structure of ShapeWithDtype, optional
            Structure of the input/parameter space.
        lsm_tangents_shape : tree-like structure of ShapeWithDtype, optional
            Structure of the data space. Will be inferred from
            `normalized_residual` and `domain` if not set.
        """
        # TODO: track forward model and build lsm, metric, residual only when
        # called instead of always partially
        self._hamiltonian = energy
        self._transformation = transformation
        self._normalized_residual = normalized_residual
        self._left_sqrt_metric = left_sqrt_metric
        self._metric = metric

        self._domain = domain
        # NOTE, `lsm_tangents_shape` is not `normalized_residual` applied to
        # `domain` for e.g. models with a learnable covariance
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

    def normalized_residual(self, primals, **primals_kw):
        """Applies the normalized_residual to `primals`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the energy.
        **primals_kw : Any
           Additional arguments passed on to the energy.

        Returns
        -------
        normalized_residual : tree-like structure
            Structure of the same type as lsm_tangents_shape for which the
            normalized_residual is computed.
        """
        if self._normalized_residual is None:
            nie = "`normalized_residual` is not implemented"
            raise NotImplementedError(nie)
        return self._normalized_residual(primals, **primals_kw)

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
            lsm_at_p = Partial(self.left_sqrt_metric, primals, **primals_kw)
            rsm_at_p = jax.linear_transpose(
                lsm_at_p, self.left_sqrt_metric_tangents_shape
            )
            rsm_at_p = _functional_conj(rsm_at_p)
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
            Must be of shape lsm_tangents_shape.
        **primals_kw : Any
           Additional arguments passed on to the LSM.

        Returns
        -------
        metric_sample : tree-like structure
            Tree-like structure of the same type as primals to which
            the left-square-root of the metric has been applied to.
        """
        if self._left_sqrt_metric is None:
            _, bwd = jax.vjp(
                Partial(self.transformation, **primals_kw), primals
            )
            bwd = _functional_conj(bwd)
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
            Structure of the same type as lsm_tangents_shape to which the
            geometric transformation has been applied to.
        """
        if self._transformation is None:
            nie = "`transformation` is not implemented"
            raise NotImplementedError(nie)
        return self._transformation(primals, **primals_kw)

    @property
    def domain(self):
        return self._domain

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
        self,
        energy: Callable,
        normalized_residual: Optional[Callable],
        transformation: Optional[Callable],
        left_sqrt_metric: Optional[Callable],
        metric: Optional[Callable],
        domain=None,
    ):
        """Instantiates a new likelihood with the same `lsm_tangents_shape`.

        Parameters
        ----------
        energy : callable
            Function evaluating the negative log-likelihood.
        normalized_residual : callable, optional
            Function evaluating the data residual of the likelihood
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
            normalized_residual=normalized_residual,
            transformation=transformation,
            left_sqrt_metric=left_sqrt_metric,
            metric=metric,
            lsm_tangents_shape=self._lsm_tan_shp,
            domain=domain if domain is not None else self._domain
        )

    def jit(self, **kwargs):
        """Returns a new likelihood with jit-compiled energy, left-square-root
        of metric and metric.
        """
        from jax import jit

        j_r = (
            jit(self.normalized_residual)
            if self._normalized_residual is not None else None
        )

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
            normalized_residual=j_r,
            transformation=j_trafo,
            left_sqrt_metric=j_lsm,
            metric=j_m
        )

    def __matmul__(self, f: Callable):
        return self.amend(f, left_argnames=(), right_argnames=None)

    def amend(
        self,
        f: Callable,
        *,
        domain=None,
        left_argnames=(),
        right_argnames=None
    ):
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

        def normalized_residual_at_f(primals, **primals_kw):
            kw_l, kw_r = split_kwargs(**primals_kw)
            return self.normalized_residual(f(primals, **kw_r), **kw_l)

        def transformation_at_f(primals, **primals_kw):
            kw_l, kw_r = split_kwargs(**primals_kw)
            return self.transformation(f(primals, **kw_r), **kw_l)

        def metric_at_f(primals, tangents, **primals_kw):
            kw_l, kw_r = split_kwargs(**primals_kw)
            # Note, judging by a simple benchmark on a large problem,
            # transposing the JVP seems faster than computing the VJP again. On
            # small problems there seems to be no measurable difference.
            y, fwd = jax.linearize(Partial(f, **kw_r), primals)
            bwd = jax.linear_transpose(fwd, primals)
            bwd = _functional_conj(bwd)
            return bwd(self.metric(y, fwd(tangents), **kw_l))[0]

        def left_sqrt_metric_at_f(primals, tangents, **primals_kw):
            kw_l, kw_r = split_kwargs(**primals_kw)
            y, bwd = jax.vjp(Partial(f, **kw_r), primals)
            bwd = _functional_conj(bwd)
            left_at_fp = self.left_sqrt_metric(y, tangents, **kw_l)
            return bwd(left_at_fp)[0]

        domain = f.domain if domain is None and isinstance(
            f, AbstractModel
        ) else domain

        return self.new(
            energy_at_f,
            normalized_residual=normalized_residual_at_f,
            transformation=transformation_at_f,
            left_sqrt_metric=left_sqrt_metric_at_f,
            metric=metric_at_f,
            domain=domain,
        )

    def __add__(self, other):
        if not isinstance(other, Likelihood):
            te = (
                "object which to add to this instance is of invalid type"
                f" {type(other)!r}"
            )
            raise TypeError(te)

        lkey = "lh_left"
        rkey = "lh_right"

        joined_tangents_shape = {
            lkey: self._lsm_tan_shp,
            rkey: other._lsm_tan_shp
        }

        def joined_hamiltonian(p, **pkw):
            return self.energy(p, **pkw) + other.energy(p, **pkw)

        def joined_normalized_residual(p, **pkw):
            from warnings import warn

            # FIXME
            warn("adding residuals is untested", UserWarning)
            lres = self.normalized_residual(p, **pkw)
            rres = other.normalized_residual(p, **pkw)
            lvec, rvec = isinstance(lres, Vector), isinstance(rres, Vector)
            res = {
                lkey: lres.tree if lvec else lres,
                rkey: rres.tree if rvec else rres
            }
            if lvec and rvec:
                return Vector(res)
            return res

        def joined_metric(p, t, **pkw):
            return self.metric(p, t, **pkw) + other.metric(p, t, **pkw)

        def joined_transformation(p, **pkw):
            from warnings import warn

            # FIXME
            warn("adding transformations is untested", UserWarning)
            lres = self.transformation(p, **pkw)
            rres = other.transformation(p, **pkw)
            lvec, rvec = isinstance(lres, Vector), isinstance(rres, Vector)
            res = {
                lkey: lres.tree if lvec else lres,
                rkey: rres.tree if rvec else rres
            }
            if lvec and rvec:
                return Vector(res)
            return res

        def joined_left_sqrt_metric(p, t, **pkw):
            return (
                self.left_sqrt_metric(p, t[lkey], **pkw) +
                other.left_sqrt_metric(p, t[rkey], **pkw)
            )

        return Likelihood(
            joined_hamiltonian,
            normalized_residual=joined_normalized_residual,
            transformation=joined_transformation,
            left_sqrt_metric=joined_left_sqrt_metric,
            metric=joined_metric,
            lsm_tangents_shape=joined_tangents_shape
        )

    def partial(self, point_estimates, primals):
        """TODO
        """
        insert_axes, primals_liquid, primals_frozen = _parse_point_estimates(
            point_estimates, primals
        )
        energy = partial_insert_and_remove(
            self.energy,
            insert_axes=(insert_axes, ),
            flat_fill=(primals_frozen, ),
            remove_axes=None
        )
        trafo = partial_insert_and_remove(
            self.transformation,
            insert_axes=(insert_axes, ),
            flat_fill=(primals_frozen, ),
            remove_axes=None
        )
        norm_residual = partial_insert_and_remove(
            self.normalized_residual,
            insert_axes=(insert_axes, ),
            flat_fill=(primals_frozen, ),
            remove_axes=None
        )
        domain = jax.tree_map(ShapeWithDtype.from_leave, primals_liquid)

        lh = Likelihood(
            energy,
            normalized_residual=norm_residual,
            transformation=trafo,
            lsm_tangents_shape=self.lsm_tangents_shape,
            domain=domain,
        )
        return lh, primals_liquid


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
                            primals_kw) + 0.5 * vdot(primals, primals)

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
