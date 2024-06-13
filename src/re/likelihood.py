# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from dataclasses import field
from typing import Any, Callable, TypeVar

import jax
from jax.tree_util import (
    Partial, tree_flatten, tree_leaves, tree_map, tree_structure, tree_unflatten
)

from .misc import is_iterable_of_non_iterables, isiterable
from .model import LazyModel, NoValue
from .tree_math import ShapeWithDtype, Vector, conj, has_arithmetics, zeros_like

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

    This function is best understood by example:

    .. code-block:: python

        def _identity(x):
            return x

        # _identity takes exactly one argument, thus `insert_axes` and `flat_fill`
        # are length one tuples
        _id_part = jpartial(
            _identity,
            insert_axes=(jft.Vector({
                "a": (True, False),
                "b": False
            }), ),
            flat_fill=(("THIS IS input['a'][0]", ), )
        )
        out = _id_part(("THIS IS input['a'][1]", "THIS IS input['b']"))
        assert out == jft.Vector(
            {
                "a": ("THIS IS input['a'][0]", "THIS IS input['a'][1]"),
                "b": "THIS IS input['b']"
            }
        )

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


def _parse_swd(shape):
    leaves = tree_leaves(shape)
    if not all(hasattr(e, "shape") and hasattr(e, "dtype") for e in leaves):
        if is_iterable_of_non_iterables(shape) or not isiterable(shape):
            shape = ShapeWithDtype(shape)
        else:
            te = "`lsm_tangents_shapes` of invalid type"
            raise TypeError(te)
    return shape


class Likelihood(LazyModel):
    """Storage class for keeping track of the energy, the associated
    left-square-root of the metric and the metric.

    Properties
    ----------
    energy : callable
        Function evaluating the negative log-likelihood.
    transformation : callable
        Function evaluating the geometric transformation of the likelihood.
    normalized_residual : callable
        Function evaluating the data residual normalized by the standard
        deviation of the likelihood.
    left_sqrt_metric : callable
        Function applying the left-square-root of the metric.
    metric : callable
        Function applying the metric.
    domain : tree-like structure of ShapeWithDtype
        Structure of the input/parameter space.
    lsm_tangents_shape : tree-like structure of ShapeWithDtype
        Structure of the data space.
    """

    _lsm_tan_shp: Any = None

    def __init__(
        self, *, domain=NoValue, init=NoValue, lsm_tangents_shape=None
    ):
        # NOTE, `lsm_tangents_shape` is not `normalized_residual` applied to
        # `domain` for e.g. models with a learnable covariance
        self._lsm_tan_shp = _parse_swd(lsm_tangents_shape)
        super().__init__(domain=domain, init=init)

    def __call__(self, primals, **primals_kw):
        """Convenience method to access the `energy` method of this instance."""
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
        nie = "`energy` is not implemented"
        raise NotImplementedError(nie)

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
        nie = "`normalized_residual` is not implemented"
        raise NotImplementedError(nie)

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
        lsm_at_p = Partial(self.left_sqrt_metric, primals, **primals_kw)
        return lsm_at_p(self.right_sqrt_metric(primals, tangents, **primals_kw))

    def left_sqrt_metric(self, primals, tangents, **primals_kw):
        """Applies the left-square-root (LSM) of the metric at `primals` to
        `tangents`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the metric.
        tangents : tree-like structure
            Instance to which to apply the metric. Must be of shape
            `lsm_tangents_shape`.
        **primals_kw : Any
           Additional arguments passed on to the LSM.

        Returns
        -------
        metric_sample : tree-like structure
            Tree-like structure of the same type as primals to which
            the left-square-root of the metric has been applied to.
        """
        _, bwd = jax.vjp(Partial(self.transformation, **primals_kw), primals)
        bwd = _functional_conj(bwd)
        res = bwd(tangents)
        return res[0]

    def right_sqrt_metric(self, primals, tangents, **primals_kw):
        """Applies the right-square-root (RSM) of the metric at `primals` to
        `tangents`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the metric.
        tangents : tree-like structure
            Instance to which to apply the metric. Must be of the same shape as
            `primals`.
        **primals_kw : Any
           Additional arguments passed on to the RSM.

        Returns
        -------
        metric_sample : tree-like structure
            Tree-like structure of the same type as
            `left_sqrt_metric_tangents_shape`.
        """
        lsm_at_p = Partial(self.left_sqrt_metric, primals, **primals_kw)
        rsm_at_p = jax.linear_transpose(
            lsm_at_p, self.left_sqrt_metric_tangents_shape
        )
        rsm_at_p = _functional_conj(rsm_at_p)
        return rsm_at_p(tangents)[0]

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
        raise NotImplementedError("`transformation` is not implemented")

    @property
    def left_sqrt_metric_tangents_shape(self):
        """Retrieves the shape of the tangent domain of the
        left-square-root of the metric.
        """
        return self._lsm_tan_shp

    @property
    def lsm_tangents_shape(self):
        """Alias for `left_sqrt_metric_tangents_shape`."""
        return self.left_sqrt_metric_tangents_shape

    @property
    def right_sqrt_metric_tangents_shape(self):
        """Retrieves the shape of the tangent domain of the
        right-square-root of the metric, a.k.a. the domain.
        """
        return self.domain

    @property
    def rsm_tangents_shape(self):
        """Alias for `right_sqrt_metric_tangents_shape`."""
        return self.right_sqrt_metric_tangents_shape

    def amend(
        self, f: Callable, /, *, domain=NoValue, likelihood_argnames=None
    ):
        """Amend a forward model to the likelihood."""
        return LikelihoodWithModel(
            self, f, domain=domain, likelihood_argnames=likelihood_argnames
        )

    def __add__(self, other):
        return LikelihoodSum(self, other)

    def freeze(self, *, primals, point_estimates):
        """Returns a new likelihood with partially inserted `primals` and the
        remaining unfrozen/liquid `primals`.
        """
        if not point_estimates:
            return self, primals
        lp = LikelihoodPartial(
            self, primals=primals, point_estimates=point_estimates
        )
        return lp, lp.splitx(primals)[0]


class LikelihoodPartial(Likelihood):
    """Likelihood with partially inserted `primals`."""

    likelihood: Likelihood = field(metadata=dict(static=False))
    primals_frozen: Any = field(metadata=dict(static=False))

    def __init__(
        self,
        likelihood,
        /,
        *,
        primals,
        point_estimates,
    ):
        self.likelihood = likelihood
        self.point_estimates = point_estimates
        self.insert_axes, pl, self.primals_frozen = _parse_point_estimates(
            self.point_estimates, primals
        )
        self.unflatten = Vector if self.insert_axes else None
        super().__init__(
            domain=tree_map(ShapeWithDtype.from_leave, pl),
            lsm_tangents_shape=self.likelihood.lsm_tangents_shape,
        )

    @property
    def energy(self):
        return partial_insert_and_remove(
            self.likelihood.energy,
            insert_axes=(self.insert_axes, ),
            flat_fill=(self.primals_frozen, ),
            remove_axes=None,
        )

    @property
    def transformation(self):
        return partial_insert_and_remove(
            self.likelihood.transformation,
            insert_axes=(self.insert_axes, ),
            flat_fill=(self.primals_frozen, ),
            remove_axes=None,
        )

    @property
    def left_sqrt_metric(self):
        return partial_insert_and_remove(
            self.likelihood.left_sqrt_metric,
            insert_axes=(self.insert_axes, None),
            flat_fill=(self.primals_frozen, None),
            remove_axes=self.insert_axes,
            unflatten=self.unflatten,
        )

    @property
    def right_sqrt_metric(self):
        return partial_insert_and_remove(
            self.likelihood.right_sqrt_metric,
            insert_axes=(self.insert_axes, self.insert_axes),
            flat_fill=(self.primals_frozen, zeros_like(self.primals_frozen)),
            remove_axes=None,
        )

    @property
    def metric(self):
        return partial_insert_and_remove(
            self.likelihood.metric,
            insert_axes=(self.insert_axes, self.insert_axes),
            flat_fill=(self.primals_frozen, zeros_like(self.primals_frozen)),
            remove_axes=self.insert_axes,
            unflatten=self.unflatten,
        )

    @property
    def normalized_residual(self):
        return partial_insert_and_remove(
            self.likelihood.normalized_residual,
            insert_axes=(self.insert_axes, ),
            flat_fill=(self.primals_frozen, ),
            remove_axes=None,
        )

    def splitx(self, primals):
        """Split the primals into liquid and frozen.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the energy.

        Returns
        -------
        primals_liqud : tree-like structure
            Liquid/dynamic part of the position at which to evaluate the energy.
        primals_frozen : tree-like structure
            Frozen/static part of the position at which to evaluate the energy.
        """
        return _parse_point_estimates(self.point_estimates, primals)[1:]


class LikelihoodWithModel(Likelihood):
    likelihood: Likelihood = field(metadata=dict(static=False))
    forward: Callable = field(metadata=dict(static=False))
    likelihood_argnames: tuple = ()

    def __init__(
        self,
        likelihood: Likelihood,
        f: Callable,
        /,
        *,
        domain=NoValue,
        init=NoValue,
        likelihood_argnames=None,
    ):
        """Amend the function `f` to the right of the likelihood.

        Parameters
        ----------
        f : Callable
            Function which to amend to the likelihood.
        likelihood_argnames : tuple or None
            Keys of the keyword arguments of the joined likelihood which
            to pass to the original likelihood. Passing `None` indicates
            the intent to absorb everything not explicitly absorbed by
            the other call.

        Returns
        -------
        lh : Likelihood
        """
        self.likelihood = likelihood
        if not callable(f):
            te = f"second argument to {self.__class__.__name__} must be callable; got {f!r}"
            raise TypeError(te)
        self.forward = f if isinstance(f, LazyModel) else Partial(f)
        likelihood_argnames = (
        ) if likelihood_argnames is None else likelihood_argnames
        if not isinstance(likelihood_argnames, (tuple, list)):
            te = f"invalid `likelihood_argnames` {self.likelihood_argnames!r}"
            raise TypeError(te)
        self.likelihood_argnames = likelihood_argnames
        domain = f.domain if domain is NoValue and isinstance(
            f, LazyModel
        ) else domain
        init = f.init if init is NoValue and isinstance(f, LazyModel) else init
        super().__init__(
            domain=domain,
            init=init,
            lsm_tangents_shape=self.likelihood.lsm_tangents_shape,
        )

    def _split_kwargs(self, **kwargs):
        left = {k: kwargs.pop(k) for k in self.likelihood_argnames}
        return left, kwargs

    def energy(self, primals, **kwargs):
        kw_l, kw_r = self._split_kwargs(**kwargs)
        return self.likelihood(self.forward(primals, **kw_r), **kw_l)

    def normalized_residual(self, primals, **kwargs):
        kw_l, kw_r = self._split_kwargs(**kwargs)
        return self.likelihood.normalized_residual(
            self.forward(primals, **kw_r), **kw_l
        )

    def transformation(self, primals, **kwargs):
        kw_l, kw_r = self._split_kwargs(**kwargs)
        return self.likelihood.transformation(
            self.forward(primals, **kw_r), **kw_l
        )

    def metric(self, primals, tangents, **kwargs):
        kw_l, kw_r = self._split_kwargs(**kwargs)
        # Note, judging by a simple benchmark on a large problem,
        # transposing the JVP seems faster than computing the VJP again. On
        # small problems there seems to be no measurable difference.
        y, fwd = jax.linearize(Partial(self.forward, **kw_r), primals)
        bwd = jax.linear_transpose(fwd, primals)
        bwd = _functional_conj(bwd)
        return bwd(self.likelihood.metric(y, fwd(tangents), **kw_l))[0]

    def left_sqrt_metric(self, primals, tangents, **kwargs):
        kw_l, kw_r = self._split_kwargs(**kwargs)
        y, bwd = jax.vjp(Partial(self.forward, **kw_r), primals)
        bwd = _functional_conj(bwd)
        left_at_fp = self.likelihood.left_sqrt_metric(y, tangents, **kw_l)
        return bwd(left_at_fp)[0]

    def right_sqrt_metric(self, primals, tangents, **kwargs):
        kw_l, kw_r = self._split_kwargs(**kwargs)
        y, fwd = jax.linearize(Partial(self.forward, **kw_r), primals)
        return self.likelihood.right_sqrt_metric(y, fwd(tangents), **kw_l)

    def amend(
        self,
        f: Callable,
        *,
        domain=NoValue,
        left_argnames=None,
        likelihood_argnames=None,
    ):
        domain = f.domain if domain is NoValue and isinstance(
            f, LazyModel
        ) else domain
        left_argnames = () if left_argnames is None else left_argnames
        likelihood_argnames = (
            self.likelihood_argnames
            if likelihood_argnames is None else likelihood_argnames
        )

        def ff(primals, **kwargs):
            kw_l = {k: kwargs.pop(k) for k in left_argnames}
            kw_r = kwargs
            return self.forward(f(primals, **kw_r), **kw_l)

        return self.__class__(
            self.likelihood,
            ff,
            domain=domain,
            likelihood_argnames=likelihood_argnames,
        )


class LikelihoodSum(Likelihood):
    left_likelihood: Likelihood = field(metadata=dict(static=False))
    right_likelihood: Likelihood = field(metadata=dict(static=False))

    def __init__(
        self,
        left,
        right,
        /,
        domain=NoValue,
        init=NoValue,
        _left_key="lh_left",
        _right_key="lh_right",
    ):
        if not (isinstance(left, Likelihood) and isinstance(right, Likelihood)):
            te = (
                "object which to add to this instance is of invalid type"
                f" {type(right)!r}"
            )
            raise TypeError(te)
        self._lkey, self._rkey = _left_key, _right_key
        joined_tangents_shape = {
            self._lkey: left._lsm_tan_shp,
            self._rkey: right._lsm_tan_shp,
        }
        if isinstance(left._lsm_tan_shp,
                      Vector) or isinstance(right._lsm_tan_shp, Vector):
            joined_tangents_shape = Vector(joined_tangents_shape)

        if (
            domain is NoValue and left.domain is not NoValue and
            right.domain is not NoValue
        ):
            lvec = isinstance(left.domain, Vector)
            rvec = isinstance(right.domain, Vector)
            ldomain = left.domain.tree if lvec else left.domain
            rdomain = right.domain.tree if rvec else right.domain
            domain = ldomain | rdomain
            domain = Vector(domain) if lvec or rvec else domain
            isswd = hasattr(domain, "shape") and hasattr(domain, "dtype")
            if not isswd and not has_arithmetics(domain):
                ve = (
                    "domains of the Likelihood-summands must support core"
                    " arithmetic operations"
                    "\nmaybe you forgot to wrap your inputs to the liklihoods"
                    " in `Vector`s"
                )
                raise ValueError(ve)
        self.left_likelihood = left
        self.right_likelihood = right
        super().__init__(
            domain=domain, init=init, lsm_tangents_shape=joined_tangents_shape
        )

    def energy(self, primals, **kwargs):
        return self.left_likelihood.energy(
            primals, **kwargs
        ) + self.right_likelihood.energy(primals, **kwargs)

    def normalized_residual(self, primals, **kwargs):
        lres = self.left_likelihood.normalized_residual(primals, **kwargs)
        rres = self.right_likelihood.normalized_residual(primals, **kwargs)
        lvec, rvec = isinstance(lres, Vector), isinstance(rres, Vector)
        res = {self._lkey: lres, self._rkey: rres}
        res = Vector(res) if lvec or rvec else res
        return res

    def metric(self, primals, tangents, **kwargs):
        return self.left_likelihood.metric(
            primals, tangents, **kwargs
        ) + self.right_likelihood.metric(primals, tangents, **kwargs)

    def transformation(self, primals, **kwargs):
        lres = self.left_likelihood.transformation(primals, **kwargs)
        rres = self.right_likelihood.transformation(primals, **kwargs)
        lvec, rvec = isinstance(lres, Vector), isinstance(rres, Vector)
        res = {self._lkey: lres, self._rkey: rres}
        res = Vector(res) if lvec or rvec else res
        return res

    def left_sqrt_metric(self, primals, tangents, **kwargs):
        return self.left_likelihood.left_sqrt_metric(
            primals, tangents[self._lkey], **kwargs
        ) + self.right_likelihood.left_sqrt_metric(
            primals, tangents[self._rkey], **kwargs
        )

    def right_sqrt_metric(self, primals, tangents, **kwargs):
        lres = self.left_likelihood.right_sqrt_metric(
            primals, tangents, **kwargs
        )
        rres = self.right_likelihood.right_sqrt_metric(
            primals, tangents, **kwargs
        )
        lvec, rvec = isinstance(lres, Vector), isinstance(rres, Vector)
        res = {self._lkey: lres, self._rkey: rres}
        res = Vector(res) if lvec or rvec else res
        return res
