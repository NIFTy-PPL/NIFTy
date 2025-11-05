# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Gordian Edenhofer, Philipp Frank, Jakob Roth, Vincent Eberle

from functools import partial
from operator import getitem
from typing import Callable, Optional, Tuple, TypeVar, Union

import jax
from jax import numpy as jnp
from jax import random
from jax.tree_util import Partial, register_pytree_node_class, tree_leaves, tree_map

from . import conjugate_gradient, optimize
from .likelihood import (
    Likelihood,
    LikelihoodWithModel,
    _functional_conj,
    _parse_point_estimates,
    partial_insert_and_remove,
)
from .misc import conditional_raise
from .tree_math import (
    Vector,
    assert_arithmetics,
    get_map,
    random_like,
    stack,
    vdot,
    conj,
    zeros_like,
)

P = TypeVar("P")


def _no_jit(x, **kwargs):
    return x


def _parse_jit(jit):
    if callable(jit):
        return jit
    if isinstance(jit, bool):
        return jax.jit if jit else _no_jit
    raise TypeError(f"expected `jit` to be callable or boolean; got {jit!r}")


def _is_no_jit(jit):
    return jit == _no_jit


@jax.jit
def concatenate_zip(*arrays):
    return tree_map(
        lambda *x: jnp.stack(x, axis=1).reshape((-1,) + x[0].shape[1:]), *arrays
    )


def _process_point_estimate(x, primals, point_estimates, insert):
    if not point_estimates:
        return x

    point_estimates, _, p_frozen = _parse_point_estimates(point_estimates, primals)
    assert p_frozen is not None
    fill = tree_map(lambda x: jnp.zeros((1,) * jnp.ndim(x)), p_frozen)
    in_out = partial_insert_and_remove(
        lambda *x: x[0],
        insert_axes=(point_estimates,) if insert else None,
        flat_fill=(fill,) if insert else None,
        remove_axes=None if insert else (point_estimates,),
        unflatten=None if insert else Vector,
    )
    return in_out(x)


def sample_likelihood(likelihood: Likelihood, point_estimates, primals, key):
    lh, p_liquid = likelihood.freeze(point_estimates=point_estimates, primals=primals)
    white_sample = random_like(key, lh.left_sqrt_metric_tangents_shape)
    return lh.left_sqrt_metric(p_liquid, white_sample)


def _ham_metric(likelihood, point_estimates, primals, tangents, **primals_kw):
    lh, p_liquid = likelihood.freeze(point_estimates=point_estimates, primals=primals)
    return lh.metric(p_liquid, tangents, **primals_kw) + tangents


def draw_linear_residual(
    likelihood: Likelihood,
    pos: P,
    key,
    *,
    from_inverse: bool = True,
    point_estimates: Union[P, Tuple[str]] = (),
    cg: Callable = conjugate_gradient.cg,
    cg_name: Optional[str] = None,
    cg_kwargs: Optional[dict] = None,
    jit_metric=False,
    _raise_nonposdef: bool = False,
) -> tuple[P, int]:
    assert_arithmetics(pos)

    if not isinstance(likelihood, Likelihood):
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)
    p_liquid = pos
    if point_estimates:
        _, p_liquid = likelihood.freeze(point_estimates=point_estimates, primals=pos)

    jit = _parse_jit(jit_metric)
    ham_metric = partial(
        jit(_ham_metric, static_argnames="point_estimates"), likelihood, point_estimates
    )

    cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    subkey_nll, subkey_prr = random.split(key, 2)
    nll_smpl = sample_likelihood(likelihood, point_estimates, pos, key=subkey_nll)
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
            cg,
            Partial(ham_metric, pos),
            **{"name": cg_name, "_raise_nonposdef": _raise_nonposdef, **cg_kwargs},
        )
        smpl, info = inv_metric_at_p(smpl, x0=prr_inv_metric_smpl)
        conditional_raise(
            (info < 0) if info is not None else False,
            ValueError("conjugate gradient failed"),
        )
    smpl = _process_point_estimate(smpl, pos, point_estimates, insert=True)
    return smpl, info


def _nonlinear_residual_vg(likelihood, point_estimates, e, lh_trafo_at_p, ms_at_p, x):
    lh, e_liquid = likelihood.freeze(point_estimates=point_estimates, primals=e)

    # t = likelihood.transformation(x) - lh_trafo_at_p
    t = tree_map(jnp.subtract, lh.transformation(x), lh_trafo_at_p)
    g = x - e_liquid + lh.left_sqrt_metric(e_liquid, t)
    r = ms_at_p - g
    res = 0.5 * vdot(r, r)

    r = conj(r)
    ngrad = r + lh.left_sqrt_metric(x, lh.right_sqrt_metric(e_liquid, r))
    return (res, -ngrad)


def _nonlinear_residual_metric(likelihood, point_estimates, e, primals, tangents):
    lh, e_liquid = likelihood.freeze(point_estimates=point_estimates, primals=e)
    lsm = lh.left_sqrt_metric
    rsm = lh.right_sqrt_metric
    tm = lsm(e_liquid, rsm(primals, tangents)) + tangents
    return lsm(primals, rsm(e_liquid, tm)) + tm


def _nonlinear_residual_sampnorm(likelihood, point_estimates, e, natgrad):
    lh, e_liquid = likelihood.freeze(point_estimates=point_estimates, primals=e)
    fpp = lh.right_sqrt_metric(e_liquid, natgrad)
    return jnp.sqrt(vdot(natgrad, natgrad) + vdot(fpp, fpp))


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
    jit_residual_funcs=False,
    _raise_notconverged=False,
) -> tuple[P, optimize.OptimizeResults]:
    assert_arithmetics(pos)
    assert_arithmetics(residual_sample)

    draw_lni = partial(
        draw_linear_residual,
        likelihood,
        from_inverse=False,
        point_estimates=point_estimates,
    )

    jit = _parse_jit(jit_residual_funcs)
    residual_vg = partial(
        jit(_nonlinear_residual_vg, static_argnames="point_estimates"),
        likelihood,
        point_estimates,
    )
    metric = partial(
        jit(_nonlinear_residual_metric, static_argnames="point_estimates"),
        likelihood,
        point_estimates,
    )
    sampnorm = partial(
        jit(_nonlinear_residual_sampnorm, static_argnames="point_estimates"),
        likelihood,
        point_estimates,
    )

    sample = pos + residual_sample
    del residual_sample
    sample = _process_point_estimate(sample, pos, point_estimates, insert=False)
    metric_sample, _ = draw_lni(pos, metric_sample_key)
    metric_sample *= metric_sample_sign
    metric_sample = _process_point_estimate(
        metric_sample, pos, point_estimates, insert=False
    )
    # HACK for skipping the nonlinear update steps and not calling trafo
    skip = (
        isinstance(minimize_kwargs.get("maxiter", None), int)
        and minimize_kwargs["maxiter"] == 0
    )
    if not skip:
        lh_f, e_liquid = likelihood.freeze(point_estimates=point_estimates, primals=pos)
        trafo_at_p = lh_f.transformation(e_liquid)
        options = {
            "fun_and_grad": partial(residual_vg, pos, trafo_at_p, metric_sample),
            "hessp": partial(metric, pos),
            "custom_gradnorm": partial(sampnorm, pos),
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
        _raise_notconverged=_raise_notconverged,
    )
    return stack(
        (
            curve(residual_sample, metric_sample_sign=1.0),
            curve(-residual_sample, metric_sample_sign=-1.0),
        )
    )


@register_pytree_node_class
class Samples:
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
        smpls = tree_map(lambda s: s.reshape((-1,) + s.shape[2:]), self._samples)
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


def wiener_filter_posterior(
    likelihood: LikelihoodWithModel,
    position: Optional[P] = None,
    *,
    key,
    n_samples: int = 0,
    residual_map="lmap",
    draw_linear_kwargs: Optional[dict] = None,
    jit=True,
    model_is_linear: Optional[bool] = True,
    signal_space: Optional[bool] = True,
    noise_covariance: Optional[callable] = None,
) -> Tuple[Samples, Tuple]:
    """Computes Wiener filter solution for a standardized model. For non-linear
    models, the Wiener filter solution is computed for a linearized model.

    Parameters
    ----------
    likelihood : :class:`~nifty.re.likelihood.LikelihoodWithModel`
        Likelihood to be used for the Wiener filter.
    position : tree-like
        Position around which to linearize (if the model is non-linear).
    key : jax random number generation key
    n_samples : int
        Number of samples to draw.
    residual_map : callable
        Map function used for the residual sample drawing.
    draw_linear_kwargs : dict
        Optional parameters for the conjugate gradient used to compute the
        posterior mean and to draw samples.
    jit : bool or callable, default=True
        Whether to JIT-compile the Wiener filter covariance.
    model_is_linear : bool
        Whether the model is linear. If the model is non-linear, you must
        specify a position around which to linearize it. For non-linear models,
        consider applying variational inference via
        :func:`~nifty.re.optimize_kl.optimize_kl` instead.
    signal_space: bool
        Wheter the Wiener filter should be solved in signal or data space.
        The result should be equal, up to numerical precision, thus depends
        on the condition number of the forward model.
    noise_covariance: callable
        Noise covariance of the data. This callable is only needed for the
        Wiener filter in data space.
    """
    if not isinstance(likelihood, LikelihoodWithModel):
        msg = f"likelihood must be of LikelihoodWithModel type; got {likelihood}"
        return TypeError(msg)
    if not model_is_linear and position is None:
        msg = "For nonlinear models a position to linearize must be specified."
        raise ValueError(msg)

    residual_map = get_map(residual_map)
    jit = _parse_jit(jit)
    position = zeros_like(likelihood.domain) if position is None else position

    data = likelihood.likelihood.data

    if model_is_linear:
        forward_lin = likelihood.forward
    else:
        _, forward_lin = jax.linearize(likelihood.forward, position)
        data = data - likelihood.forward(position) + forward_lin(position)

    cg = draw_linear_kwargs.get("cg", conjugate_gradient.cg)
    forward_lin_T = jax.linear_transpose(forward_lin, likelihood.domain)
    forward_lin_T = _functional_conj(forward_lin_T)

    if signal_space:
        n_inv = Partial(likelihood.likelihood.metric, likelihood.forward(position))
        (j,) = forward_lin_T(n_inv(data))

        def post_cov_inv(tangents):
            return forward_lin_T(n_inv(forward_lin(tangents)))[0] + tangents

        post_cov_inv = jit(post_cov_inv)
        post_mean, post_info = cg(
            post_cov_inv,
            j,
            name=draw_linear_kwargs.get("cg_name", None),
            **draw_linear_kwargs.get("cg_kwargs", {}),
        )
        if post_info is not None and post_info < 0:
            raise ValueError("conjugate gradient failed")
    else:
        if noise_covariance is None:
            raise ValueError(
                "To use the Wiener filter in data space, please set the noise_covariance"
            )

        def post_dspace_cov_inv(tangents):
            (R_dagger_d,) = forward_lin_T(tangents)
            RR_dagger_d = forward_lin(R_dagger_d)
            return RR_dagger_d + noise_covariance(tangents)

        post_dspace_cov_inv = jit(post_dspace_cov_inv)
        post_mean_dspace, post_info = cg(
            post_dspace_cov_inv,
            data,
            name=draw_linear_kwargs.get("cg_name", None),
            **draw_linear_kwargs.get("cg_kwargs", {}),
        )
        (post_mean,) = forward_lin_T(post_mean_dspace)
        if post_info is not None and post_info < 0:
            raise ValueError("conjugate gradient failed")

    ks = random.split(key, n_samples)
    draw = Partial(
        draw_linear_residual, likelihood, jit_metric=jit, **draw_linear_kwargs
    )
    draw = residual_map(draw, in_axes=(None, 0))
    smpls, smpls_info = draw(post_mean, ks)

    smpls = Samples(pos=post_mean, samples=concatenate_zip(smpls, -smpls), keys=ks)
    return smpls, (post_info, smpls_info)
