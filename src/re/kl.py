# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank, Gordian Edenhofer

import sys
from functools import partial
from operator import getitem
from typing import (Callable, Optional, Tuple, TypeVar, Union, NamedTuple, Any,
                    List)
from warnings import warn

from jax import numpy as jnp
from jax import random, vmap, value_and_grad, jvp, vjp, linear_transpose, jit
from jax.tree_util import (
    Partial, register_pytree_node_class, tree_leaves, tree_map, tree_structure
)

from .smap import smap
from .optimize import OptimizeResults, minimize, conjugate_gradient
from .likelihood import (
    Likelihood, StandardHamiltonian, _partial_insert_and_remove,
    _partial_argument
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
        in_out = _partial_insert_and_remove(
            lambda *x: x[0],
            insert_axes=(point_estimates,) if insert else None,
            flat_fill=(fill,) if insert else None,
            remove_axes=None if insert else (point_estimates,),
            unflatten=None if insert else Vector
        )
        return in_out(x)
    return x


def _likelihood_metric_plus_standard_prior(lh_metric):
    if isinstance(lh_metric, Likelihood):
        lh_metric = lh_metric.metric

    def joined_metric(primals, tangents, **primals_kw):
        return lh_metric(primals, tangents, **primals_kw) + tangents

    return joined_metric


def sample_likelihood(likelihood: Likelihood, primals, key):
    white_sample = random_like(key, likelihood.left_sqrt_metric_tangents_shape)
    return likelihood.left_sqrt_metric(primals, white_sample)


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


def _ham_vg(likelihood, primals, primals_samples):
    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vvg = vmap(value_and_grad(ham))
    s = vvg(primals_samples.at(primals).samples)
    return tree_map(partial(jnp.mean, axis=0), s)


def _ham_metric(likelihood, primals, tangents, primals_samples):
    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vmet = vmap(ham.metric, in_axes=(0, None))
    s = vmet(primals_samples.at(primals).samples, tangents)
    return tree_map(partial(jnp.mean, axis=0), s)


def _lh_trafo(likelihood, p):
    return likelihood.transformation(p)


def _nl_g(likelihood, p, lh_trafo_at_p, x):
    t = likelihood.transformation(x) - lh_trafo_at_p
    return x - p + likelihood.left_sqrt_metric(p, t)


def _nl_residual(likelihood, p, lh_trafo_at_p, ms_at_p, x):
    r = ms_at_p - _nl_g(likelihood, p, lh_trafo_at_p, x)
    return 0.5*dot(r, r)


def _nl_metric(likelihood, p, lh_trafo_at_p, primals, tangents):
    f = partial(_nl_g, likelihood, p, lh_trafo_at_p)
    _, jj = jvp(f, (primals,), (tangents,))
    return vjp(f, primals)[1](jj)[0]


def _nl_sampnorm(likelihood, p, natgrad):
    o = partial(likelihood.left_sqrt_metric, p)
    fpp = linear_transpose(o, likelihood.lsm_tangents_shape)(natgrad)
    return jnp.sqrt(vdot(natgrad, natgrad) + vdot(fpp, fpp))



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
  sampling_states: List[OptimizeResults]
  minimization_state: OptimizeResults


class OptimizeVI:
    def __init__(self,
                 likelihood: Union[Likelihood, None],
                 n_iter: int,
                 key: random.PRNGKey,
                 n_samples: int,
                 point_estimates: Union[P, Tuple[str]] = (),
                 sampling_method: str = 'altmetric',
                 sampling_minimizer = 'newtoncg',
                 sampling_kwargs: dict = {'xtol':0.01},
                 sampling_cg_kwargs: dict = {'maxiter':50},
                 minimizer: str = 'newtoncg',
                 minimization_kwargs: dict = {},
                 do_jit = jit,
                 _raise_notconverged = False,
                 _lh_funcs: Any = None):
        # TODO reintroduce point-estimates (also possibly different sampling
        # methods for pytree)
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
        sampling_method: str
            Sampling method used for vi approximation. Default is `altmetric`.
        sampling_minimizer: str
            Minimization method used for non-linear sample minimization.
        sampling_kwargs: dict
            Keyword arguments for minimizer used for sample minimization.
        sampling_cg_kwargs: dict
            Keyword arguments for ConjugateGradient used for the linear part of
            sample minimization.
        minimizer: str or callable
            Minimization method used for KL minimization.
        minimization_kwargs : dict
            Keyword arguments for minimizer used for KL minimization.

        See also
        --------
        `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
        Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
        `<https://doi.org/10.3390/e23070853>`_

        `Metric Gaussian Variational Inference`, Jakob Knollmüller,
        Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
        """
        self._n_iter = n_iter
        self._sampling_method = sampling_method
        if self._sampling_method not in ['linear', 'geometric', 'altmetric']:
            msg = f"Unknown sampling method: {self._sampling_method}"
            raise NotImplementedError(msg)
        self._minimizer = minimizer
        self._sampling_minimizer = sampling_minimizer
        self._sampling_kwargs = sampling_kwargs
        # Only use xtol for sampling since a custom gradient norm is used
        self._sampling_kwargs['absdelta'] = 0.
        self._mini_kwargs = minimization_kwargs
        self._keys = random.split(key, n_samples)
        self._likelihood = likelihood
        self._n_samples = n_samples
        self._sampling_cg_kwargs = sampling_cg_kwargs
        self._point_estimates = point_estimates
        self._raise_notconverged = _raise_notconverged

        if _lh_funcs is None:
            if likelihood is None:
                raise ValueError("Neither Likelihood nor funcs provided.")

            def draw_metric(likelihood, p, keys):
                f = partial(_sample_linearly, likelihood,
                            from_inverse=False)
                return vmap(f, in_axes=(None, 0), out_axes=(None, 0))(p, keys)

            def draw_linear(likelihood, p, keys):
                f = partial(_sample_linearly, likelihood,
                            from_inverse=True, cg_kwargs=sampling_cg_kwargs,
                            _raise_nonposdef = self._raise_notconverged)
                return smap(f, in_axes=(None, 0))(p, keys)

            # KL funcs
            self._kl_vg = do_jit(partial(_ham_vg, likelihood))
            self._kl_metric = do_jit(partial(_ham_metric, likelihood))

            # Sampling
            get_partial = partial(_partial_func, likelihood=likelihood,
                                  point_estimates=point_estimates)
            # Lin sampling
            self._draw_metric = do_jit(get_partial(draw_metric))
            self._draw_linear = do_jit(get_partial(draw_linear))
            # Non-lin sampling
            self._lh_trafo = do_jit(get_partial(_lh_trafo))
            self._nl_vag = do_jit(value_and_grad(get_partial(_nl_residual),
                                                 argnums=3))
            self._nl_metric = do_jit(get_partial(_nl_metric))
            self._nl_sampnorm = do_jit(get_partial(_nl_sampnorm))
        else:
            if likelihood is not None:
                msg = "Warning: Likelihood funcs is set, ignoring Likelihood"
                msg += " input"
                print(msg, file=sys.stderr)
            (self._kl_vg,
             self._kl_metric,
             self._draw_linear,
             self._draw_metric,
             self._lh_trafo,
             self._nl_vag,
             self._nl_metric,
             self._nl_sampnorm) = _lh_funcs

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def lh_funcs(self):
        return (self._kl_vg,
                self._kl_metric,
                self._draw_linear,
                self._draw_metric,
                self._lh_trafo,
                self._nl_vag,
                self._nl_metric,
                self._nl_sampnorm)

    def _linear_sampling(self, primals, from_inverse):
        if from_inverse:
            samples, met_smpls = self._draw_linear(primals, self._keys)
            samples = stack(
                tuple(_process_point_estimate(ss,
                                              primals,
                                              self._point_estimates,
                                              insert=True
                                              )
                      for ss in unstack(samples))
            )
            samples = Samples(
                pos=primals,
                samples=tree_map(lambda *x:
                                 jnp.concatenate(x), samples, -samples)
            )
        else:
            _, met_smpls = self._draw_metric(primals, self._keys)
            samples = None
        met_smpls = stack(
            tuple(_process_point_estimate(ss,
                                          primals,
                                          self._point_estimates,
                                          insert=True
                                          )
                    for ss in unstack(met_smpls))
        )
        met_smpls = Samples(
            pos=None,
            samples=tree_map(lambda *x:
                             jnp.concatenate(x), met_smpls, -met_smpls)
        )
        return samples, met_smpls

    def _nonlinear_sampling(self, samples):
        primals = samples.pos
        lh_trafo_at_p = self._lh_trafo(primals)
        metric_samples = self._linear_sampling(primals, False)[1]
        new_smpls = []
        opt_states = []
        for s, ms in zip(samples, metric_samples):
            s = _process_point_estimate(s, primals, self._point_estimates,
                                        insert=False)
            ms = _process_point_estimate(ms, primals, self._point_estimates,
                                         insert=False)
            options = {
                "fun_and_grad":
                    partial(
                        self._nl_vag,
                        primals,
                        lh_trafo_at_p,
                        ms
                    ),
                "hessp":
                    partial(
                        self._nl_metric,
                        primals,
                        lh_trafo_at_p
                    ),
                "custom_gradnorm" : partial(self._nl_sampnorm, primals),
                }
            opt_state = minimize(None, x0=s, method=self._sampling_minimizer,
                                 options=self._sampling_kwargs | options)
            _cond_raise(
                self._raise_notconverged & (opt_state.status < 0),
                ValueError("S: failed to invert map")
            )
            newsam = _process_point_estimate(opt_state.x, primals,
                                             self._point_estimates,
                                             insert=True)
            new_smpls.append(newsam - primals)
            # Remove x from state to avoid copy of the samples
            opt_states.append(opt_state._replace(x = None))
        samples = Samples(pos=primals, samples=stack(new_smpls))
        return samples, opt_states

    def _minimize_kl(self, samples):
        options = {
            "fun_and_grad": partial(self._kl_vg, primals_samples=samples),
            "hessp": partial(self._kl_metric, primals_samples=samples),
        }
        opt_state = minimize(
            None,
            samples.pos,
            method=self._minimizer,
            options=self._mini_kwargs | options
        )
        return samples.at(opt_state.x), opt_state

    def init_state(self, primals):
        if self._sampling_method in ['linear', 'geometric']:
            smpls = self._linear_sampling(primals, False)[1]
        else:
            smpls = self._linear_sampling(primals, True)[0]
        state = OptVIState(niter=0, samples=smpls, sampling_states=None,
                           minimization_state=None)
        return primals, state

    def update(self, primals, state):
        if self._sampling_method in ['linear', 'geometric']:
            samples = self._linear_sampling(primals, True)[0]
        else:
            samples = state.samples.at(primals)

        if self._sampling_method in ['geometric', 'altmetric']:
            samples, sampling_states = self._nonlinear_sampling(samples)
        else:
            sampling_states = None
        samples, opt_state = self._minimize_kl(samples)
        state = OptVIState(niter=state.niter+1, samples=samples,
                           sampling_states=sampling_states,
                           minimization_state=opt_state)
        return samples.pos, state

    def run(self, primals):
        primals, state = self.init_state(primals)
        for i in range(self._n_iter):
            print(f"OptVI iteration number: {i}", file=sys.stderr)
            primals, state = self.update(primals, state)
        return primals, state