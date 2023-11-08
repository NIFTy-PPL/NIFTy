# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank, Gordian Edenhofer

import sys
from functools import partial
from operator import getitem
from typing import (Callable, Optional, Tuple, TypeVar, Union, NamedTuple, Any,
                    List)
from warnings import warn
from .logger import logger

from jax import numpy as jnp, Array
from jax import random, vmap, value_and_grad, jvp, vjp, linear_transpose, jit
from jax.tree_util import (
    Partial, register_pytree_node_class, tree_leaves, tree_map, tree_structure
)

from .smap import smap
from .optimize import OptimizeResults, minimize, conjugate_gradient
from .likelihood import (
    Likelihood, StandardHamiltonian, partial_insert_and_remove, _functional_conj
)
from .tree_math import (
    Vector, assert_arithmetics, random_like, stack, dot, vdot, unstack
)


P = TypeVar("P")


def _cond_raise(condition, exception):
    from jax.experimental.host_callback import call

    def maybe_raise(condition):
        if condition:
            raise exception

    call(maybe_raise, condition, result_shape=None)


def _parse_point_estimates(point_estimates, primals):
    if isinstance(point_estimates, (tuple, list)):
        if not isinstance(primals, (Vector, dict)):
            te = "tuple-shortcut point-estimate only availble for dict/Vector type primals"
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


def _partial_func(func, likelihood, point_estimates):
    if point_estimates:
        def partial_func(primals, *args):
            pe, p_liquid, p_frozen = _parse_point_estimates(point_estimates,
                                                            primals)
            return func(likelihood.partial(pe, p_frozen), p_liquid, *args)

        return partial_func
    return partial(func, likelihood)


def _process_point_estimate(x, primals, point_estimates, insert):
    if point_estimates:
        point_estimates, _, p_frozen = _parse_point_estimates(
            point_estimates,
            primals
        )
        assert p_frozen is not None
        fill = tree_map(lambda x: jnp.zeros((1,)*jnp.ndim(x)), p_frozen)
        in_out = partial_insert_and_remove(
            lambda *x: x[0],
            insert_axes=(point_estimates,) if insert else None,
            flat_fill=(fill,) if insert else None,
            remove_axes=None if insert else (point_estimates,),
            unflatten=None if insert else Vector
        )
        return in_out(x)
    return x


def sample_likelihood(likelihood: Likelihood, primals, key):
    white_sample = random_like(key, likelihood.left_sqrt_metric_tangents_shape)
    return likelihood.left_sqrt_metric(primals, white_sample)


def draw_linear_residual(
    likelihood: Likelihood,
    primals,
    key,
    from_inverse: bool,
    point_estimates: Union[P, Tuple[str]] = (),
    cg: Callable = conjugate_gradient.static_cg,
    cg_name: Optional[str] = None,
    cg_kwargs: Optional[dict] = None,
    _raise_nonposdef: bool = False,
):
    assert_arithmetics(primals)

    if not isinstance(likelihood, Likelihood):
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)
    if point_estimates:
        pe, p_liquid, p_frozen = _parse_point_estimates(point_estimates,primals)
        lh = likelihood.partial(pe, p_frozen)
    else:
        lh = likelihood
        p_liquid = primals
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
    return _process_point_estimate(smpl, primals, point_estimates, insert=True)


def linear_residual_sampler(likelihood,
                   point_estimates: Union[P, Tuple[str]] = (),
                   cg: Callable = conjugate_gradient.static_cg,
                   cg_name: Optional[str] = None,
                   cg_kwargs: Optional[dict] = None,
                   samplemap: Callable = smap,
                   do_dit: Callable = jit,
                   _raise_nonposdef: bool = False):
    """Wrapper for `draw_linear_residual` to draw multiple samples at once.

    Returns two functions which take as inputs `primals` and a list of `keys`.
    The first function generates metric samples, the second one inverse metric
    samples.
    Allows to specify how to map over sample generation and how to jit it.
    """
    def draw_linear(primals, keys, from_inverse):
        sampler = partial(draw_linear_residual, likelihood, primals,
                          from_inverse=from_inverse,
                          point_estimates=point_estimates,
                          cg = cg,
                          cg_name = cg_name,
                          cg_kwargs=cg_kwargs,
                          _raise_nonposdef=_raise_nonposdef)
        samples = samplemap(sampler)(keys)
        samples = Samples(
                    pos=primals,
                    samples=tree_map(lambda *x:
                                jnp.concatenate(x), samples, -samples)
        )
        return samples

    return (do_dit(partial(draw_linear, from_inverse=False)),
            do_dit(partial(draw_linear, from_inverse=True)))


def _sample_mean(samples, mean=jnp.mean, axis=0):
    return tree_map(partial(mean, axis=axis), samples)


def kl_vg_and_metric(likelihood,
                     samplemap=vmap,
                     sample_reduce=_sample_mean,
                     do_jit=jit):

    def _ham_vg(primals, primals_samples):
        assert isinstance(primals_samples, Samples)
        ham = StandardHamiltonian(likelihood=likelihood)
        vvg = samplemap(value_and_grad(ham))
        s = vvg(primals_samples.at(primals).samples)
        return sample_reduce(s)

    def _ham_metric(primals, tangents, primals_samples):
        assert isinstance(primals_samples, Samples)
        ham = StandardHamiltonian(likelihood=likelihood)
        vmet = samplemap(ham.metric, in_axes=(0, None))
        s = vmet(primals_samples.at(primals).samples, tangents)
        return sample_reduce(s)

    return do_jit(_ham_vg), do_jit(_ham_metric)


def kl_solver(likelihood,
              samplemap=vmap,
              sample_reduce=_sample_mean,
              do_jit=jit):
    kl_vg, kl_metric = kl_vg_and_metric(likelihood,samplemap=samplemap,
                                        sample_reduce=sample_reduce,
                                        do_jit=do_jit)
    def _minimize_kl(samples,
                     method='newtoncg',
                     method_options={}):
        options = {
            "fun_and_grad": partial(kl_vg, primals_samples=samples),
            "hessp": partial(kl_metric, primals_samples=samples),
        }
        opt_state = minimize(
            None,
            samples.pos,
            method=method,
            options=method_options | options
        )
        return samples.at(opt_state.x), opt_state
    return _minimize_kl


def curve_residual_functions(likelihood, point_estimates=(), do_jit=jit):
    def _trafo(likelihood, p):
        return likelihood.transformation(p)

    def _g(likelihood, p, lh_trafo_at_p, x):
        # t = likelihood.transformation(x) - lh_trafo_at_p
        t = tree_map(jnp.subtract, likelihood.transformation(x), lh_trafo_at_p)
        return x - p + likelihood.left_sqrt_metric(p, t)

    def _residual(likelihood, p, lh_trafo_at_p, ms_at_p, x):
        r = ms_at_p - _g(likelihood, p, lh_trafo_at_p, x)
        return 0.5*dot(r, r)

    def _metric(likelihood, p, lh_trafo_at_p, primals, tangents):
        f = partial(_g, likelihood, p, lh_trafo_at_p)
        _, jj = jvp(f, (primals,), (tangents,))
        return vjp(f, primals)[1](jj)[0]

    def _sampnorm(likelihood, p, natgrad):
        o = partial(likelihood.left_sqrt_metric, p)
        o_transpose = linear_transpose(o, likelihood.lsm_tangents_shape)
        fpp = _functional_conj(o_transpose)(natgrad)
        return jnp.sqrt(vdot(natgrad, natgrad) + vdot(fpp, fpp))

    # Partially insert frozen point estimates
    get_partial = partial(_partial_func, likelihood=likelihood,
                            point_estimates=point_estimates)
    trafo = do_jit(get_partial(_trafo))
    vag = do_jit(value_and_grad(get_partial(_residual), argnums=3))
    metric = do_jit(get_partial(_metric))
    sampnorm = do_jit(get_partial(_sampnorm))
    return trafo, vag, metric, sampnorm


def curve_residual(likelihood=None,
                 point_estimates=(),
                 primals=None,
                 sample=None,
                 metric_sample=None,
                 method='newtoncg',
                 method_options={},
                 do_jit=jit,
                 curve_funcs=None,
                 _raise_notconverged=False):

    if curve_funcs is None:
        trafo, vag, metric, sampnorm = curve_residual_functions(
            likelihood=likelihood,
            point_estimates=point_estimates,
            do_jit=do_jit
        )
    else:
        trafo, vag, metric, sampnorm = curve_funcs

    sample = _process_point_estimate(sample, primals,
                                     point_estimates,
                                     insert=False)
    metric_sample = _process_point_estimate(metric_sample, primals,
                                            point_estimates,
                                            insert=False)
    trafo_at_p = trafo(primals)
    options = {
        "fun_and_grad": partial(vag, primals, trafo_at_p, metric_sample),
        "hessp": partial(metric, primals, trafo_at_p),
        "custom_gradnorm" : partial(sampnorm, primals),
        }
    opt_state = minimize(None, x0=sample, method=method,
                         options=method_options | options)
    if _raise_notconverged & (opt_state.status < 0):
        ValueError("S: failed to invert map")
    newsam = _process_point_estimate(opt_state.x, primals,
                                     point_estimates,
                                     insert=True)
    # Remove x from state to avoid copy of the samples
    return newsam - primals, opt_state._replace(x = None)

def curve_sampler(likelihood,
                  metric_sampler,
                  point_estimates=(),
                  sample_map=None, #TODO
                  do_jit=jit,
                  _raise_notconverged=False):

    curve_funcs = curve_residual_functions(
        likelihood=likelihood,
        point_estimates=point_estimates,
        do_jit=do_jit
    )
    def sampler(samples, keys, method='newtoncg', method_options={}):
        assert isinstance(samples, Samples)
        primals = samples.pos
        residuals = samples._samples
        met_samps = metric_sampler(primals, keys)
        states = []
        for i, (ss, ms) in enumerate(zip(samples, met_samps)):
            rr, state = curve_residual(point_estimates=point_estimates,
                                       primals=primals,
                                       sample=ss,
                                       metric_sample=ms,
                                       method=method,
                                       method_options=method_options,
                                       curve_funcs=curve_funcs,
                                       _raise_notconverged=_raise_notconverged)
            residuals = tree_map(lambda ss, xx: ss.at[i].set(xx), residuals, rr)
            states.append(state)
        return Samples(pos=primals, samples=residuals), states
    return sampler


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


class OptVIState(NamedTuple):
    """Named tuple containing state information."""
    niter: int
    samples: Samples
    keys: Array
    resample: bool
    sampling_method: str
    sampling_states: List[OptimizeResults]
    minimization_state: OptimizeResults


def OptimizeVI(likelihood: Union[Likelihood, None],
               n_iter: int,
               point_estimates: Union[P, Tuple[str]] = (),
               kl_kwargs: dict = {
                    'samplemap': vmap,
                    'sample_reduce': _sample_mean,
                    'do_jit': jit
               },
               linear_sampling_kwargs: dict = {
                    'cg': conjugate_gradient.static_cg,
                    'cg_name': None,
                    'cg_kwargs': None,
                    'samplemap': smap,
                    'do_jit': jit,
               },
               curve_kwargs: dict = {
                    'sample_map': None, #TODO
                    'do_jit': jit,
               },
               _raise_notconverged: bool = False):
    """JaxOpt style minimizer for VI approximation of a Bayesian inference
    problem assuming a standard normal prior distribution.

    Depending on `sampling_method` the VI approximation is performed via
    variants of the `Geometric Variational Inference` and/or
    `Metric Gaussian Variational Inference` algorithms. They produce
    approximate posterior samples that are used for KL estimation internally
    and the final set of samples are the approximation of the posterior.
    The samples can be linear, i.e. following a standard normal distribution
    in model space, or non linear, i.e. following a standard normal
    distribution in the canonical coordinate system of the Riemannian
    manifold associated with the metric of the approximate posterior
    distribution. The coordinate transformation for the non-linear sample is
    approximated by an expansion.

    Both linear and non-linear sample start by drawing a sample from the
    inverse metric. To do so, we draw a sample which has the metric as
    covariance structure and apply the inverse metric to it. The sample
    transformed in this way has the inverse metric as covariance. The first
    part is trivial since we can use the left square root of the metric
    :math:`L` associated with every likelihood:

    .. math::

        \tilde{d} \leftarrow \mathcal{G}(0,\mathbb{1}) \\
        t = L \tilde{d}

    with :math:`t` now having a covariance structure of

    .. math::
        <t t^\dagger> = L <\tilde{d} \tilde{d}^\dagger> L^\dagger = M .

    To transform the sample to an inverse sample, we apply the inverse
    metric. We can do so using the conjugate gradient algorithm (CG). The CG
    algorithm yields the solution to :math:`M s = t`, i.e. applies the
    inverse of :math:`M` to :math:`t`:

    .. math::

        M &s =  t \\
        &s = M^{-1} t = cg(M, t) .

    The linear sample is :math:`s`. The non-linear sample uses :math:`s` as
    a starting value and curves it in a non-linear way as to better resemble
    the posterior locally. See the below reference literature for more
    details on the non-linear sampling.

    Parameters
    ----------
    likelihood : :class:`nifty8.re.likelihood.Likelihood`
        Likelihood to be used for inference.
    n_iter : int
        Number of iterations.
    key : jax random number generataion key
    n_samples : int
        Number of samples used to sample Kullback-Leibler divergence. The
        samples get mirrored, so the actual number of samples used for the
        KL estimate is twice the number of `n_samples`.
    point_estimates : tree-like structure or tuple of str
        Pytree of same structure as likelihood input but with boolean leaves
        indicating whether to sample the value in the input or use it as a
        point estimate. As a convenience method, for dict-like inputs, a
        tuple of strings is also valid. From these the boolean indicator
        pytree is automatically constructed.
    kl_kwargs: dict
        Keyword arguments passed on to `kl_solver`.
    kl_minimizer: str or callable
        Minimization method used for KL minimization.
    kl_minimizer_kwargs : dict
        Keyword arguments for minimizer used for KL minimization.
    sampling_method: str
        Sampling method used for vi approximation. Must be in ('linear',
        'geometric', 'altmetric'). Default is `altmetric`.
    linear_sampling_kwargs: dict
        Keyword arguments passed on to `linear_residual_sampler`. Includes
        the cg gonfig used for linear sampling.
    curve_minimizer: str
        Minimization method used for non-linear sample minimization.
    curve_minimizer_kwargs: dict
        Keyword arguments for minimizer used for sample minimization.
    curve_kwargs: dict
        Keyword arguments passed on to `curve_sampler`.
    _raise_notconverged: bool
        Whether to raise inversion & minimization errors during sampling.
        Default is False.
    _vi_callables: tuple of callable (optional)
        Alternative init to normal init with `likelihood`. If provided,
        includes all functions for sampling and minimization and no new
        functions are build.

    See also
    --------
    `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
    Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
    `<https://doi.org/10.3390/e23070853>`_

    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
    """
    linear_sampling_kwargs.setdefault('_raise_nonposdef',_raise_notconverged)
    curve_kwargs.setdefault('_raise_notconverged',_raise_notconverged)

    # KL funcs
    solver = kl_solver(likelihood, **kl_kwargs)
    # Lin sampling
    draw_metric, draw_linear = linear_residual_sampler(
        likelihood,
        point_estimates,
        **linear_sampling_kwargs
    )
    # Non-lin sampling
    curve = curve_sampler(
        likelihood,
        draw_metric,
        point_estimates,
        **curve_kwargs
    )
    return _OptimizeVI(n_iter, solver, draw_linear, curve)


class _OptimizeVI:
    def __init__(self,
                 n_iter: int,
                 kl_solver: Callable,
                 linear_sampler: Callable,
                 curve_sampler: Callable):
        self._n_iter = n_iter
        self._kl_solver = kl_solver
        self._linear_sampler = linear_sampler
        self._curve_sampler = curve_sampler

    def init_state(self, primals, keys, sampling_method='geometric'):
        if sampling_method not in ['linear', 'geometric', 'altmetric']:
            msg = f"Unknown sampling method: {sampling_method}"
            raise NotImplementedError(msg)
        state = OptVIState(niter=0,
                           samples=None,
                           keys=keys,
                           resample=True,
                           sampling_method=sampling_method,
                           sampling_states=None,
                           minimization_state=None)
        return state

    def update(self, primals, state,
               kl_minimizer: str = 'newtoncg',
               kl_minimizer_kwargs: dict = {},
               curve_minimizer: str = 'newtoncg',
               curve_minimizer_kwargs: dict = {'xtol': 0.01},):
        assert isinstance(state, OptVIState)
        if state.resample or (state.sampling_method in ['linear', 'geometric']):
            samples = self._linear_sampler(primals, state.keys)
        else:
            samples = state.samples.at(primals)

        if state.sampling_method in ['geometric', 'altmetric']:
            if 'absdelta' in curve_minimizer_kwargs.keys():
                msg = 'Geometric sampling uses custom gradientnorm tolerance '
                msg += 'set by `xtol`. Ignoring `absdelta`...'
                logger.warn(msg)
                curve_minimizer_kwargs['absdelta'] = 0.
            samples, sampling_states = self._curve_sampler(
                samples,
                state.keys,
                method=curve_minimizer,
                method_options=curve_minimizer_kwargs
            )
        else:
            sampling_states = None
        samples, opt_state = self._kl_solver(samples,
                                             method=kl_minimizer,
                                             method_options=kl_minimizer_kwargs)
        state = OptVIState(niter=state.niter+1,
                           samples=samples,
                           keys=state.keys,
                           resample=state.resample,
                           sampling_method=state.sampling_method,
                           sampling_states=sampling_states,
                           minimization_state=opt_state)
        return samples.pos, state

    def run(self, primals):
        primals, state = self.init_state(primals)
        for n in range(self._n_iter):
            logger.info(f"OptVI iteration number: {n}")
            primals, state = self.update(primals, state)
        return primals, state