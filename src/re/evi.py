# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Gordian Edenhofer, Philipp Frank

from functools import partial
from operator import getitem
from typing import Callable, Optional, Tuple, TypeVar, Union

import jax
from jax import numpy as jnp
from jax import random
from jax.tree_util import (
    Partial, register_pytree_node_class, tree_leaves, tree_map
)

from . import conjugate_gradient, optimize
from .likelihood import (
    Likelihood, _parse_point_estimates, partial_insert_and_remove
)
from .tree_math import (
    Vector, assert_arithmetics, dot, hide_strings, random_like, stack, vdot
)

P = TypeVar("P")


def _no_jit(x, **kwargs):
    return x


def _parse_jit(jit):
    if callable(jit):
        return jit
    if isinstance(jit, bool):
        return jax.jit if jit else _no_jit
    raise TypeError(f"expected `jit` to be callable or bolean; got {jit!r}")


def _hcb_maybe_raise(condition_exception):
    condition, exception = condition_exception
    if condition:
        raise exception()


def _cond_raise(condition, exception):
    from jax.debug import callback

    # Register as few host-callbacks as possible by implicitly hashing the
    # exception type and the strings within
    callback(
        _hcb_maybe_raise, (
            condition,
            Partial(exception.__class__, *hide_strings(exception.args))
        )
    )


def _process_point_estimate(x, primals, point_estimates, insert):
    if not point_estimates:
        return x

    point_estimates, _, p_frozen = _parse_point_estimates(
        point_estimates, primals
    )
    assert p_frozen is not None
    fill = tree_map(lambda x: jnp.zeros((1, ) * jnp.ndim(x)), p_frozen)
    in_out = partial_insert_and_remove(
        lambda *x: x[0],
        insert_axes=(point_estimates, ) if insert else None,
        flat_fill=(fill, ) if insert else None,
        remove_axes=None if insert else (point_estimates, ),
        unflatten=None if insert else Vector
    )
    return in_out(x)


def sample_likelihood(likelihood: Likelihood, primals, key):
    white_sample = random_like(key, likelihood.left_sqrt_metric_tangents_shape)
    return likelihood.left_sqrt_metric(primals, white_sample)


def draw_linear_residual(
    likelihood: Likelihood,
    pos: P,
    key,
    *,
    from_inverse: bool = True,
    point_estimates: Union[P, Tuple[str]] = (),
    cg: Callable = conjugate_gradient.static_cg,
    cg_name: Optional[str] = None,
    cg_kwargs: Optional[dict] = None,
    _raise_nonposdef: bool = False,
) -> tuple[P, int]:
    assert_arithmetics(pos)

    if not isinstance(likelihood, Likelihood):
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)
    if point_estimates:
        lh, p_liquid = likelihood.freeze(
            point_estimates=point_estimates, primals=pos
        )
    else:
        lh = likelihood
        p_liquid = pos

    def ham_metric(primals, tangents, **primals_kw):
        return lh.metric(primals, tangents, **primals_kw) + tangents

    cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    subkey_nll, subkey_prr = random.split(key, 2)
    nll_smpl = sample_likelihood(lh, p_liquid, key=subkey_nll)
    prr_inv_metric_smpl = random_like(key=subkey_prr, primals=p_liquid)
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
    smpl = nll_smpl + prr_smpl
    info = 0
    if from_inverse:
        inv_metric_at_p = partial(
            cg, Partial(ham_metric, p_liquid), **{
                "name": cg_name,
                "_raise_nonposdef": _raise_nonposdef,
                **cg_kwargs
            }
        )
        smpl, info = inv_metric_at_p(smpl, x0=prr_inv_metric_smpl)
        _cond_raise(
            (info < 0) if info is not None else False,
            ValueError("conjugate gradient failed")
        )
    smpl = _process_point_estimate(smpl, pos, point_estimates, insert=True)
    return smpl, info


def _nonlinearly_update_residual_functions(
    likelihood, jit: Union[Callable, bool] = False
):
    def _draw_linear_non_inverse(primals, key, *, point_estimates):
        # `draw_linear_residual` already handles `point_estimates` no need to
        # partially insert anything here
        return draw_linear_residual(
            likelihood,
            primals,
            key,
            point_estimates=point_estimates,
            from_inverse=False
        )

    def _residual_vg(e, lh_trafo_at_p, ms_at_p, x, *, point_estimates):
        lh, e_liquid = likelihood.freeze(
            point_estimates=point_estimates, primals=e
        )

        # t = likelihood.transformation(x) - lh_trafo_at_p
        t = tree_map(jnp.subtract, lh.transformation(x), lh_trafo_at_p)
        g = x - e_liquid + lh.left_sqrt_metric(e_liquid, t)
        r = ms_at_p - g
        res = 0.5 * dot(r, r)

        ngrad = tree_map(jnp.conj, r)
        ngrad += lh.left_sqrt_metric(x, lh.right_sqrt_metric(e_liquid, ngrad))
        return (res, -ngrad)

    def _metric(e, primals, tangents, *, point_estimates):
        lh, e_liquid = likelihood.freeze(
            point_estimates=point_estimates, primals=e
        )
        lsm = lh.left_sqrt_metric
        rsm = lh.right_sqrt_metric
        tm = lsm(e_liquid, rsm(primals, tangents)) + tangents
        return lsm(primals, rsm(e_liquid, tm)) + tm

    def _sampnorm(e, natgrad, *, point_estimates):
        lh, e_liquid = likelihood.freeze(
            point_estimates=point_estimates, primals=e
        )
        fpp = lh.right_sqrt_metric(e_liquid, natgrad)
        return jnp.sqrt(vdot(natgrad, natgrad) + vdot(fpp, fpp))

    jit = _parse_jit(jit)
    jit = partial(jit, static_argnames=("point_estimates", ))
    draw_linear_non_inverse = jit(_draw_linear_non_inverse)
    rag = jit(_residual_vg)
    metric = jit(_metric)
    sampnorm = jit(_sampnorm)
    return draw_linear_non_inverse, rag, metric, sampnorm


def nonlinearly_update_residual(
    likelihood=None,
    pos: P = None,
    residual_sample=None,
    metric_sample_key=None,
    metric_sample_sign=1.0,
    *,
    point_estimates=(),
    minimize: Callable[..., optimize.OptimizeResults] = optimize._newton_cg,
    minimize_kwargs={},
    jit: Union[Callable, bool] = False,
    _nonlinear_update_funcs=None,
    _raise_notconverged=False,
) -> tuple[P, optimize.OptimizeResults]:
    assert_arithmetics(pos)
    assert_arithmetics(residual_sample)

    if _nonlinear_update_funcs is None:
        _nonlinear_update_funcs = _nonlinearly_update_residual_functions(
            likelihood, jit=jit
        )
    draw_lni, rag, metric, sampnorm = _nonlinear_update_funcs

    sample = pos + residual_sample
    del residual_sample
    sample = _process_point_estimate(
        sample, pos, point_estimates, insert=False)
    metric_sample, _ = draw_lni(
        pos, metric_sample_key, point_estimates=point_estimates
    )
    metric_sample *= metric_sample_sign
    metric_sample = _process_point_estimate(
        metric_sample, pos, point_estimates, insert=False
    )
    # HACK for skipping the nonlinear update steps and not calling trafo
    skip = isinstance(minimize_kwargs.get("maxiter", None),
                      int) and minimize_kwargs["maxiter"] == 0
    if not skip:
        trafo_at_p = likelihood.transformation(pos)
        options = {
            "fun_and_grad":
                partial(
                    rag,
                    pos,
                    trafo_at_p,
                    metric_sample,
                    point_estimates=point_estimates
                ),
            "hessp":
                partial(metric, pos, point_estimates=point_estimates),
            "custom_gradnorm":
                partial(sampnorm, pos, point_estimates=point_estimates),
        }
        opt_state = minimize(None, x0=sample, **(minimize_kwargs | options))
    else:
        opt_state = optimize.OptimizeResults(sample, True, 0, None, None)
    if _raise_notconverged and (opt_state.status < 0):
        ValueError("S: failed to invert map")
    # Subtract position in the reduced space (i.e. space w/o point-estimates) to
    # not pollute the point-estimated parameters with the mean
    sample = opt_state.x - _process_point_estimate(
        pos, pos, point_estimates, insert=False
    )
    # Remove x from state to avoid copy of the samples
    opt_state = opt_state._replace(x=None, jac=None)
    sample = _process_point_estimate(sample, pos, point_estimates, insert=True)
    return sample, opt_state


def draw_residual(
    likelihood: Likelihood,
    pos: P,
    key,
    *,
    point_estimates: Union[P, Tuple[str]] = (),
    cg: Callable = conjugate_gradient.static_cg,
    cg_name: Optional[str] = None,
    cg_kwargs: Optional[dict] = None,
    minimize: Callable[..., optimize.OptimizeResults] = optimize._newton_cg,
    minimize_kwargs={},
    _nonlinear_update_funcs=None,
    _raise_nonposdef: bool = False,
    _raise_notconverged: bool = False,
) -> tuple[P, optimize.OptimizeResults]:
    residual_sample, _ = draw_linear_residual(
        likelihood,
        pos,
        key,
        point_estimates=point_estimates,
        cg=cg,
        cg_name=cg_name,
        cg_kwargs=cg_kwargs,
        _raise_nonposdef=_raise_nonposdef,
    )
    curve = partial(
        nonlinearly_update_residual,
        likelihood,
        pos,
        metric_sample_key=key,
        point_estimates=point_estimates,
        minimize=minimize,
        minimize_kwargs=minimize_kwargs,
        jit=False,
        _raise_notconverged=_raise_notconverged,
        _nonlinear_update_funcs=_nonlinear_update_funcs,
    )
    return stack(
        (
            curve(residual_sample, metric_sample_sign=1.0),
            curve(-residual_sample, metric_sample_sign=-1.0)
        )
    )


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

    def __init__(self, *, pos: P = None, samples: P, keys=None):
        self._pos, self._samples, self._keys = pos, samples, keys

    @property
    def pos(self):
        return self._pos

    @property
    def samples(self):
        if self._samples is None:
            raise ValueError(f"{self.__class__.__name__} has no samples")

        smpls = self._samples
        if self.pos is not None:
            smpls = tree_map(lambda p, s: p[jnp.newaxis] + s, self.pos, smpls)
        return smpls

    @property
    def keys(self):
        return self._keys

    def __len__(self):
        if self._samples is None:
            return 0
        return jnp.shape(tree_leaves(self._samples)[0])[0]

    def __getitem__(self, index):
        if self._samples is None:
            raise ValueError(f"{self.__class__.__name__} has no samples")

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
        return Samples(pos=pos, samples=smpls, keys=self.keys)

    def squeeze(self):
        """Convenience method to merge the two leading axis of stacked samples
        (e.g. from batching).
        """
        smpls = tree_map(
            lambda s: s.reshape((-1, ) + s.shape[2:]), self._samples
        )
        return Samples(pos=self.pos, samples=smpls, keys=self.keys)

    def tree_flatten(self):
        # Include mean in samples when passing to JAX (for e.g. vmap, pmap, ...)
        # return ((self.samples, ), (self.pos, ))  # confuses JAX
        return ((self.pos, self._samples, self.keys), ())

    @classmethod
    def tree_unflatten(cls, aux, children):
        # pos, = aux
        pos, smpls, keys = children
        # if pos is not None:  # confuses JAX
        #     smpls = tree_map(lambda p, s: s - p[jnp.newaxis], pos, smpls)
        return cls(pos=pos, samples=smpls, keys=keys)
