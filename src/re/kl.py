# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from operator import getitem
from typing import Callable, Optional, Tuple, TypeVar, Union
from warnings import warn

from jax import numpy as jnp
from jax import random
from jax.tree_util import (
    Partial, register_pytree_node_class, tree_flatten, tree_leaves, tree_map,
    tree_structure, tree_unflatten
)

from . import conjugate_gradient
from .likelihood import Likelihood, StandardHamiltonian
from .tree_math import Vector, assert_arithmetics, random_like, stack, zeros_like

P = TypeVar("P")


def sample_likelihood(likelihood: Likelihood, primals, key):
    white_sample = random_like(key, likelihood.left_sqrt_metric_tangents_shape)
    return likelihood.left_sqrt_metric(primals, white_sample)


def _cond_raise(condition, exception):
    from jax.experimental.host_callback import call

    def maybe_raise(condition):
        if condition:
            raise exception

    call(maybe_raise, condition, result_shape=None)


def _likelihood_metric_plus_standard_prior(lh_metric):
    if isinstance(lh_metric, Likelihood):
        lh_metric = lh_metric.metric

    def joined_metric(primals, tangents, **primals_kw):
        return lh_metric(primals, tangents, **primals_kw) + tangents

    return joined_metric


def _sample_linearly(
    likelihood: Likelihood,
    primals,
    key,
    from_inverse: bool,
    cg: Callable = conjugate_gradient.static_cg,
    cg_name: Optional[str] = None,
    cg_kwargs: Optional[dict] = None,
    _raise_nonposdef: bool = False,
):
    assert_arithmetics(primals)

    if isinstance(likelihood, Likelihood):
        lh = likelihood
        ham_metric = _likelihood_metric_plus_standard_prior(lh)
    elif isinstance(likelihood, StandardHamiltonian):
        msg = "passing `StandardHamiltonian` instead of the `Likelihood` is deprecated"
        warn(msg, DeprecationWarning)
        lh = likelihood.likelihood
        ham_metric = likelihood.metric
    else:
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)
    cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    subkey_nll, subkey_prr = random.split(key, 2)
    nll_smpl = sample_likelihood(lh, primals, key=subkey_nll)
    prr_inv_metric_smpl = random_like(key=subkey_prr, primals=primals)
    # One may transform any metric sample to a sample of the inverse
    # metric by simply applying the inverse metric to it
    prr_smpl = prr_inv_metric_smpl
    # Note, we can sample antithetically by swapping the global sign of
    # the metric sample below (which corresponds to mirroring the final
    # sample) and additionally by swapping the relative sign between
    # the prior and the likelihood sample. The first technique is
    # computationally cheap and empirically known to improve stability.
    # The latter technique requires an additional inversion and its
    # impact on stability is still unknown.
    # TODO: investigate the impact of sampling the prior and likelihood
    # antithetically.
    met_smpl = nll_smpl + prr_smpl
    if from_inverse:
        inv_metric_at_p = partial(
            cg, Partial(ham_metric, primals), **{
                "name": cg_name,
                "_raise_nonposdef": _raise_nonposdef,
                **cg_kwargs
            }
        )
        signal_smpl, info = inv_metric_at_p(met_smpl, x0=prr_inv_metric_smpl)
        _cond_raise(
            (info < 0) if info is not None else False,
            ValueError("conjugate gradient failed")
        )
        return signal_smpl, met_smpl
    else:
        return None, met_smpl


def _curve_sample(
    likelihood, primals, met_smpl, inv_met_smpl, *, minimize_method,
    minimize_options
):
    from .likelihood_impl import Gaussian
    from .optimize import minimize

    if isinstance(likelihood, Likelihood):
        lh = likelihood
    elif isinstance(likelihood, StandardHamiltonian):
        lh = likelihood.likelihood
    else:
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)
    x0 = primals + inv_met_smpl
    lh_trafo_at_p = lh.transformation(primals)

    def g(x):
        return x - primals + lh.left_sqrt_metric(
            primals,
            lh.transformation(x) - lh_trafo_at_p
        )

    r2_half = Gaussian(met_smpl) @ g  # (g - met_smpl)**2 / 2

    opt_state = minimize(
        r2_half,
        x0=x0,
        method=minimize_method,
        options=minimize_options | {"hessp": r2_half.metric},
    )

    return opt_state.x, opt_state.status


def _partial_argument(call, insert_axes, flat_fill):
    """For every non-None value in `insert_axes`, amend the value of `flat_fill`
    at the same position to the argument.
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
            ve = "both `insert_axes` and `flat_full` must None at the same positions"
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


def _post_partial_remove(call, remove_axes, unflatten=None):
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


def _partial_insert_and_remove(
    call, insert_axes, flat_fill, *, remove_axes=(), unflatten=None
):
    """Return a call in which `flat_fill` is inserted into arguments of `call`
    at `inset_axes` and subsequently removed from its output at `remove_axes`.
    """
    call = _partial_argument(call, insert_axes=insert_axes, flat_fill=flat_fill)
    return _post_partial_remove(call, remove_axes, unflatten=unflatten)


def _wrap_likelihood(likelihood, point_estimates, primals_frozen) -> Likelihood:
    energy = _partial_insert_and_remove(
        likelihood.energy,
        insert_axes=(point_estimates, ),
        flat_fill=(primals_frozen, ),
        remove_axes=None
    )
    trafo = _partial_insert_and_remove(
        likelihood.transformation,
        insert_axes=(point_estimates, ),
        flat_fill=(primals_frozen, ),
        remove_axes=None
    )
    lsm = _partial_insert_and_remove(
        likelihood.left_sqrt_metric,
        insert_axes=(point_estimates, None),
        flat_fill=(primals_frozen, None),
        remove_axes=point_estimates,
        unflatten=Vector
    )
    metric = _partial_insert_and_remove(
        likelihood.metric,
        insert_axes=(point_estimates, point_estimates),
        flat_fill=(primals_frozen, ) + (zeros_like(primals_frozen), ),
        remove_axes=point_estimates,
        unflatten=Vector
    )
    return Likelihood(
        energy=energy,
        transformation=trafo,
        left_sqrt_metric=lsm,
        metric=metric,
        lsm_tangents_shape=likelihood.lsm_tangents_shape
    )


def _parse_point_estimates(point_estimates, primals):
    if isinstance(point_estimates, (tuple, list)):
        if not isinstance(primals, (Vector, dict)):
            te = "tuple-shortcut point-estimate only availble for dict/Vector type primals"
            raise TypeError(te)
        pe = tree_map(lambda x: False, primals)
        pe = pe.val if isinstance(primals, Vector) else pe
        for k in point_estimates:
            pe[k] = True
        point_estimates = Vector(pe) if isinstance(primals, Vector) else pe
    if tree_structure(primals) != tree_structure(point_estimates):
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


def sample_evi(
    likelihood: Likelihood,
    primals: P,
    key,
    mirror_linear_sample: bool = True,
    *,
    point_estimates: Union[P, Tuple[str]] = (),
    linear_sampling_cg: Callable = conjugate_gradient.static_cg,
    linear_sampling_name: Optional[str] = None,
    linear_sampling_kwargs: Optional[dict] = None,
    non_linear_sampling_method: str = "NewtonCG",
    non_linear_sampling_name: Optional[str] = None,
    non_linear_sampling_kwargs: Optional[dict] = None,
    _raise_notconverged: bool = False,
) -> "Samples":
    r"""Draws a sample at a given expansion point.

    The sample can be linear, i.e. following a standard normal distribution in
    model space, or non linear, i.e. following a standard normal distribution in
    the canonical coordinate system of the Riemannian manifold associated with
    the metric of the approximate posterior distribution. The coordinate
    transformation for the non-linear sample is approximated by an expansion.

    Both linear and non-linear sample start by drawing a sample from the inverse
    metric. To do so, we draw a sample which has the metric as covariance
    structure and apply the inverse metric to it. The sample transformed in this
    way has the inverse metric as covariance. The first part is trivial since we
    can use the left square root of the metric :math:`L` associated with every
    likelihood:

    .. math::

        \tilde{d} \leftarrow \mathcal{G}(0,\mathbb{1}) \\
        t = L \tilde{d}

    with :math:`t` now having a covariance structure of

    .. math::
        <t t^\dagger> = L <\tilde{d} \tilde{d}^\dagger> L^\dagger = M .

    To transform the sample to an inverse sample, we apply the inverse metric.
    We can do so using the conjugate gradient algorithm (CG). The CG algorithm
    yields the solution to :math:`M s = t`, i.e. applies the inverse of
    :math:`M` to :math:`t`:

    .. math::

        M &s =  t \\
        &s = M^{-1} t = cg(M, t) .

    The linear sample is :math:`s`. The non-linear sample uses :math:`s` as a
    starting value and curves it in a non-linear way as to better resemble the
    posterior locally. See the below reference literature for more details on
    the non-linear sampling.

    Parameters
    ----------
    likelihood:
        Likelihood with assumed standard prior from which to draw samples.
    primals : tree-like structure
        Position at which to draw samples.
    key : tuple, list or jnp.ndarray of uint32 of length two
        Random key with which to generate random variables in data domain.
    mirror_samples : bool, optional
        Whether the mirrored version of the drawn samples are also used. If
        true, the number of used samples doubles. Mirroring samples stabilizes
        the KL estimate as extreme sample variation is counterbalanced.
    point_estimates : tree-like structure or tuple of str
        Pytree of same structure as `primals` but with boolean leaves indicating
        whether to sample the value in `primals` or use it as a point estimate.
        As a convenience method, for Field- and dict-like `primals`, a tuple of
        strings is also valid. From these the boolean indicator pytree is
        automatically constructed.
    linear_sampling_cg : callable
        Implementation of the conjugate gradient algorithm and used to apply the
        inverse of the metric.
    linear_sampling_kwargs : dict
        Additional keyword arguments passed on to `cg`.
    non_linear_sampling_method : str
        Method to use for the minimization.
    non_linear_sampling_kwargs : dict
        Additional keyword arguments passed on to the minimzer of the non-linear
        potential.
    non_linear_sampling_name : str, optional
        Name of the non-linear optimizer.
    non_linear_sampling_kwargs : dict, optional
        Options for the non-linear optimizer.

    Returns
    -------
    sample : tree-like structure
        Sample of which the covariance is the inverse metric.

    See also
    --------
    `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
    Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
    `<https://doi.org/10.3390/e23070853>`_

    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
    """
    if point_estimates:
        point_estimates, primals_liquid, primals_frozen = _parse_point_estimates(
            point_estimates, primals
        )
        likelihood = _wrap_likelihood(
            likelihood, point_estimates, primals_frozen
        )
    else:
        primals_liquid = primals
        primals_frozen = None

    inv_met_smpl, met_smpl = _sample_linearly(
        likelihood,
        primals_liquid,
        key=key,
        from_inverse=True,
        cg=linear_sampling_cg,
        cg_name=linear_sampling_name,
        cg_kwargs=linear_sampling_kwargs,
        _raise_nonposdef=_raise_notconverged,
    )

    nls_kwargs = non_linear_sampling_kwargs
    nls_kwargs = {} if nls_kwargs is None else nls_kwargs.copy()
    nls_kwargs.setdefault("name", non_linear_sampling_name)
    if "hessp" in nls_kwargs:
        ve = "setting the hessian for an unknown function is invalid"
        raise ValueError(ve)
    curve_sample = partial(
        _curve_sample,
        likelihood,
        primals_liquid,
        minimize_method=non_linear_sampling_method,
        minimize_options=nls_kwargs,
    )

    if nls_kwargs.get("maxiter", 0) == 0:
        smpls = [inv_met_smpl]
        if mirror_linear_sample:
            smpls = [inv_met_smpl, -inv_met_smpl]
    else:
        smpl1, smpl1_status = curve_sample(met_smpl, inv_met_smpl)
        _cond_raise(
            _raise_notconverged &
            ((smpl1_status < 0) if smpl1_status is not None else False),
            ValueError("S: failed to invert map")
        )
        smpls = [smpl1 - primals_liquid]
        if mirror_linear_sample:
            smpl2, smpl2_status = curve_sample(-met_smpl, -inv_met_smpl)
            _cond_raise(
                _raise_notconverged &
                ((smpl2_status < 0) if smpl2_status is not None else False),
                ValueError("S: failed to invert map")
            )
            smpls = [smpl1 - primals_liquid, smpl2 - primals_liquid]

    if point_estimates:
        assert primals_frozen is not None
        insert = _partial_argument(
            lambda *x: x[0],
            insert_axes=(point_estimates, ),
            flat_fill=(
                tree_map(
                    lambda x: jnp.zeros((1, ) * jnp.ndim(x)), primals_frozen
                ),
            )
        )
        for i in range(len(smpls)):
            smpls[i] = insert(smpls[i])
    return Samples(pos=primals, samples=stack(smpls))


@register_pytree_node_class
class Samples():
    """Storage class for samples (relative to some expansion point) that is
    fully compatible with JAX transformations like vmap, pmap, etc.

    This class is used to store samples for the Variational Inference schemes
    MGVI and geoVI where samples are defined relative to some expansion point
    (a.k.a. latent mean or offset).

    See also
    --------
    `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
    Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
    `<https://doi.org/10.3390/e23070853>`_

    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
    """
    def __init__(self, *, pos: P = None, samples: P):
        self._pos, self._samples = pos, samples
        self._n_samples = None

    @property
    def pos(self):
        return self._pos

    @property
    def samples(self):
        smpls = self._samples
        if self.pos is not None:
            smpls = tree_map(lambda p, s: p[jnp.newaxis] + s, self.pos, smpls)
        return smpls

    def __len__(self):
        return jnp.shape(tree_leaves(self._samples)[0])[0]

    def __getitem__(self, index):
        def get(b):
            return getitem(b, index)

        if self.pos is None:
            return tree_map(get, self._samples)
        return tree_map(lambda p, s: p + get(s), self.pos, self._samples)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.samples == other.samples

    def at(self, pos, old_pos=None):
        """Update the offset (usually the latent mean) of all samples and
        optionally subtracts `old_pos` from all samples before.
        """
        if self.pos is not None and old_pos is None:
            smpls = self._samples
        elif old_pos is not None:
            smpls = self.samples
            smpls = tree_map(lambda p, s: s - p[jnp.newaxis], old_pos, smpls)
        else:
            raise ValueError("invalid combination of `pos` and `old_pos`")
        return Samples(pos=pos, samples=smpls)

    def squeeze(self):
        """Convenience method to merge the two leading axis of stacked samples
        (e.g. from batching).
        """
        smpls = tree_map(
            lambda s: s.reshape((-1, ) + s.shape[2:]), self._samples
        )
        return Samples(pos=self.pos, samples=smpls)

    def tree_flatten(self):
        # Include mean in samples when passing to JAX (for e.g. vmap, pmap, ...)
        # return ((self.samples, ), (self.pos, ))  # confuses JAX
        return ((self.pos, self._samples), ())

    @classmethod
    def tree_unflatten(cls, aux, children):
        # pos, = aux
        pos, smpls, = children
        # if pos is not None:  # confuses JAX
        #     smpls = tree_map(lambda p, s: s - p[jnp.newaxis], pos, smpls)
        return cls(pos=pos, samples=smpls)
