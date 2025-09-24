# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Gordian Edenhofer, Philipp Frank

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
    lh, p_liquid = likelihood, pos
    if point_estimates:
        lh, p_liquid = likelihood.freeze(point_estimates=point_estimates, primals=pos)

    def ham_metric(primals, tangents, **primals_kw):    #geovidoc Alg. 1: return (1 + (A^T A)(primals))∙tangents
        return lh.metric(primals, tangents, **primals_kw) + tangents

    cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    subkey_nll, subkey_prr = random.split(key, 2)
    nll_smpl = sample_likelihood(lh, p_liquid, key=subkey_nll)  #geovidoc Alg. 1:  nll_smpl = A^T η_2
    prr_inv_metric_smpl = random_like(key=subkey_prr, primals=p_liquid)     #geovidoc Alg. 1: prr_inv_metric_smpl = η_1
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
    smpl = nll_smpl + prr_smpl    #geovidoc Alg. 1: smpl = A^T η_2 + η_1
    info = 0
    if from_inverse:
        inv_metric_at_p = partial(
            cg,
            Partial(ham_metric, p_liquid),
            **{"name": cg_name, "_raise_nonposdef": _raise_nonposdef, **cg_kwargs},
        )   #geovidoc Alg. 1: inv_metric_at_p = lambda x: (1 + (A^T A)(p_liquid))⁻¹∙x
        smpl, info = inv_metric_at_p(smpl, x0=prr_inv_metric_smpl)    #geovidoc Alg. 1: smpl = (1 + (A^T A)(p_liquid))⁻¹∙smpl
        conditional_raise(
            (info < 0) if info is not None else False,
            ValueError("conjugate gradient failed"),
        )
    smpl = _process_point_estimate(smpl, pos, point_estimates, insert=True)
    return smpl, info


def nonlinearly_update_residual(    #geovidoc Alg. 1 - implementation
    likelihood=None,
    pos: P = None,
    residual_sample=None,
    metric_sample_key=None,
    metric_sample_sign=1.0,
    *,
    point_estimates=(),
    minimize: Callable[..., optimize.OptimizeResults] = optimize._static_newton_cg,
    minimize_kwargs={},
    _raise_notconverged=False,
    implicit_samples=False
) -> tuple[P, optimize.OptimizeResults]:
    assert_arithmetics(pos)
    assert_arithmetics(residual_sample)

    draw_lni = partial(
        draw_linear_residual,
        likelihood,
        from_inverse=False,
        point_estimates=point_estimates,
    )

    def residual_vg(e, lh_trafo_at_p, ms_at_p, x):    #geovidoc Alg. 1: residual_vg(...) = Energy(ξ̄ , (dx/dξ̄)(ξ̄̄), z, ξ̄)
        lh, e_liquid = likelihood.freeze(point_estimates=point_estimates, primals=e)

        # t = likelihood.transformation(x) - lh_trafo_at_p
        t = tree_map(jnp.subtract, lh.transformation(x), lh_trafo_at_p)     #geovidoc Alg. 1 Line 8: t = x(ξ̄) - x(ξ̄̄)
        g = x - e_liquid + lh.left_sqrt_metric(e_liquid, t)     #geovidoc Alg. 1 Line 8: g = A^T(x(ξ̄) - x(ξ̄̄))
        r = ms_at_p - g     #geovidoc Alg. 1 Line 9: r = z-g̃
        res = 0.5 * vdot(r, r)      #geovidoc Alg. 1 Line 9: r = 0.5 (z-g̃)^T ∙ (z-g̃)

        r = conj(r)
        ngrad = r + lh.left_sqrt_metric(x, lh.right_sqrt_metric(e_liquid, r))
        return (res, -ngrad)


    # def residual_vg_symm(e, ms_at_p, x):

    #     def sampnorm_q2(e, vector):
    #         lh, e_liquid = likelihood.freeze(point_estimates=point_estimates, primals=e)
    #         fpp = lh.right_sqrt_metric(e_liquid, vector)
    #         return vdot(vector, vector) + vdot(fpp, fpp)

    #     lh, e_liquid = likelihood.freeze(point_estimates=point_estimates, primals=e)


    #     lh_trafo_at_p = lh.transformation(e_liquid)
    #     lh_trafo_at_x = lh.transformation(x)

    #     t = tree_map(jnp.subtract, lh_trafo_at_x, lh_trafo_at_p)

    #     # r_1 = ms_at_p - (x - e_liquid + lh.left_sqrt_metric(e_liquid, t))
    #     # r_2 = ms_at_p - (x - e_liquid + lh.left_sqrt_metric(x, t))
    #     # res = 0.25 * (vdot(r_1, r_1) + vdot(r_2, r_2))
    #     # res = 0.25 * (sampnorm_q2(e, r_1) + sampnorm_q2(x, r_2))

    #     r_1 = (x - e_liquid + lh.left_sqrt_metric(e_liquid, t))
    #     r_2 = (x - e_liquid + lh.left_sqrt_metric(x, t))
    #     r = ms_at_p - 0.5*(r_1 + r_2)
    #     res = 0.5 * vdot(r, r)

    #     # r_1 = ms_at_p - (x - e_liquid + lh.left_sqrt_metric(x, t))
    #     # res = 0.5 * vdot(r_1, r_1)
    #     # res = 0.5 * (sampnorm_q2(e, r_1))

    #     return res

    def residual_vg_implicit_mid(e, x):

        lh, e = likelihood.freeze(point_estimates=point_estimates, primals=e)
        
        smpl_at_x,_ = draw_lni((x+e)/2, metric_sample_key)


        lh_trafo_at_p = lh.transformation(e)
        lh_trafo_at_x = lh.transformation(x)

        t = tree_map(jnp.subtract, lh_trafo_at_x, lh_trafo_at_p)

        g_at_x = (x - e + lh.left_sqrt_metric((x+e)/2, t))
        r = smpl_at_x - g_at_x
        res = 0.5 * vdot(r, r)

        # r_1 = ms_at_p - (x - e_liquid + lh.left_sqrt_metric(x, t))
        # res = 0.5 * vdot(r_1, r_1)
        # res = 0.5 * (sampnorm_q2(e, r_1))

        return res
    
    def residual_vg_implicit_non_diag(e, x):

        lh, e = likelihood.freeze(point_estimates=point_estimates, primals=e)
        
        smpl_at_p,_ = draw_lni(e, metric_sample_key)
        smpl_at_x,_ = draw_lni(x, metric_sample_key)


        lh_trafo_at_p = lh.transformation(e)
        lh_trafo_at_x = lh.transformation(x)

        t = tree_map(jnp.subtract, lh_trafo_at_x, lh_trafo_at_p)

        g_at_p = (x - e + lh.left_sqrt_metric(e, t))
        g_at_x = (x - e + lh.left_sqrt_metric(x, t))
        r = 0.5*((smpl_at_p + smpl_at_x) - (g_at_p + g_at_x))
        res = 0.5 * vdot(r, r)

        # r_1 = ms_at_p - (x - e_liquid + lh.left_sqrt_metric(x, t))
        # res = 0.5 * vdot(r_1, r_1)
        # res = 0.5 * (sampnorm_q2(e, r_1))

        return res


    def residual_vg_nr(e, x):

        lh, e = likelihood.freeze(point_estimates=point_estimates, primals=e)

        lh_trafo_at_p = lh.transformation(e)
        lh_trafo_at_x = lh.transformation(x)
        t = tree_map(jnp.subtract, lh_trafo_at_x, lh_trafo_at_p)

        
        def pnt(xi):
            smpl_at_xi,_ = draw_lni(xi, metric_sample_key)
            g_at_xi = (x - e + lh.left_sqrt_metric(xi, t))
            return smpl_at_xi - g_at_xi

        ws = [1/6, 4/6, 1/6]
        xs = [e, (e+x)/2, x]

        ws = [1/2, 1/2]
        xs = [e, x]

        r = sum([wi * pnt(xi) for wi, xi in zip(ws, xs)])
        res = 0.5 * vdot(r, r)

        return res

    def metric(e, primals, tangents):
        lh, e_liquid = likelihood.freeze(point_estimates=point_estimates, primals=e)
        lsm = lh.left_sqrt_metric
        rsm = lh.right_sqrt_metric
        tm = lsm(e_liquid, rsm(primals, tangents)) + tangents
        return lsm(primals, rsm(e_liquid, tm)) + tm

    def sampnorm(e, natgrad):
        lh, e_liquid = likelihood.freeze(point_estimates=point_estimates, primals=e)
        fpp = lh.right_sqrt_metric(e_liquid, natgrad)
        return jnp.sqrt(vdot(natgrad, natgrad) + vdot(fpp, fpp))

    sample = pos + residual_sample
    del residual_sample
    sample = _process_point_estimate(sample, pos, point_estimates, insert=False)
    metric_sample, _ = draw_lni(pos, metric_sample_key) #geovidoc Alg. 1 Line 3 - 5: z = metric_sample
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
        if not implicit_samples or implicit_samples == "standard":
            options = {
                "fun_and_grad": partial(residual_vg, pos, trafo_at_p, metric_sample),
                "hessp": partial(metric, pos),
                "custom_gradnorm": partial(sampnorm, pos),
            }
        elif implicit_samples == "implicit":
            options = {
                "fun": partial(residual_vg_implicit_mid, pos),
                "custom_gradnorm": partial(sampnorm, pos),
            }
        elif implicit_samples == "implicit_non_diag":
                        options = {
                "fun": partial(residual_vg_implicit_non_diag, pos),
                "custom_gradnorm": partial(sampnorm, pos),
            }
        elif isinstance(implicit_samples, int):
                        options = {
                "fun": partial(residual_vg_nr, pos),
                "custom_gradnorm": partial(sampnorm, pos),
            }

        opt_state = minimize(x0=sample, **(minimize_kwargs | options)) #geovidoc Alg. 1 Line 10: opt_state = NewtonCG(Energy, ξ0)
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
    minimize: Callable[..., optimize.OptimizeResults] = optimize._static_newton_cg,
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
    residual_map="smap",
    draw_linear_kwargs: Optional[dict] = None,
    optimize_for_linear: bool = False,
) -> Tuple[Samples, Tuple]:
    """Computes wiener filter solution for a standardized model. For non-linear
    models, the wiener filter solution is computed for a linearized model.

    Parameters
    ----------
    likelihood : :class:`~nifty.re.likelihood.LikelihoodWithModel`
        Likelihood to be used for the wiener filter.
    position : tree-like
        Position around which to linearize (if the model is non-linear). By
        default the model is linearized around 0.
    key : jax random number generation key
    n_samples : int
        Number of samples to draw.
    residual_map: callable
        Map function used for the residual sample drawing.
    draw_linear_kwargs : dict
        Optional parameters passed on to :func:`draw_linear_residual` if these
        should not be the same as for the retrieval of the posterior mean.
    optimize_for_linear: bool
        Whether to optimize computations for linear model.
    """
    if not isinstance(likelihood, LikelihoodWithModel):
        msg = f"likelihood must be of LikelihoodWithModel type; got {likelihood}"
        return TypeError(msg)
    residual_map = get_map(residual_map)

    data = likelihood.likelihood.data
    # Remove any constant offsets from the data/signal that are part of the model
    data = data - likelihood.forward(zeros_like(likelihood.domain))

    position = zeros_like(likelihood.domain) if position is None else position
    if optimize_for_linear:
        forward_T = jax.linear_transpose(likelihood.forward, likelihood.domain)
    else:
        _, forward_T = jax.vjp(likelihood.forward, position)
    forward_T = _functional_conj(forward_T)
    n_inv_d = likelihood.likelihood.metric(likelihood.forward(position), data)
    (j,) = forward_T(n_inv_d)

    def post_cov_inv(tangents, primals):
        return likelihood.metric(primals, tangents) + tangents

    cg = draw_linear_kwargs.get("cg", conjugate_gradient.static_cg)
    post_mean, post_info = cg(
        Partial(post_cov_inv, primals=position),
        j,
        name=draw_linear_kwargs.get("cg_name", None),
        **draw_linear_kwargs.get("cg_kwargs", {}),
    )
    if post_info is not None and post_info < 0:
        raise ValueError("conjugate gradient failed")

    ks = random.split(key, n_samples)
    draw = Partial(draw_linear_residual, likelihood, **draw_linear_kwargs)
    draw = residual_map(draw, in_axes=(None, 0))
    smpls, smpls_info = draw(post_mean, ks)

    smpls = Samples(pos=post_mean, samples=concatenate_zip(smpls, -smpls), keys=ks)
    return smpls, (post_info, smpls_info)
