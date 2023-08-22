#!/usr/bin/env python3
# Copyright(C) 2013-2023 Philipp Frank
# SPDX-License-Identifier: BSD-2-Clause

import sys
from typing import List, NamedTuple
import jax
import jax.numpy as jnp
from functools import partial

from .optimize import OptimizeResults, minimize
from .tree_math.vector_math import dot, vdot
from .tree_math.forest_math import stack, unstack
from .likelihood import Likelihood, StandardHamiltonian
from .kl import Samples, _sample_linearly
from .smap import smap

@partial(jax.jit, static_argnames=("likelihood", ))
def _ham_vg(likelihood, primals, primals_samples):
    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vvg = jax.vmap(jax.value_and_grad(ham))
    s = vvg(primals_samples.at(primals).samples)
    return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)


@partial(jax.jit, static_argnames=("likelihood", ))
def _ham_metric(likelihood, primals, tangents, primals_samples):
    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vmet = jax.vmap(ham.metric, in_axes=(0, None))
    s = vmet(primals_samples.at(primals).samples, tangents)
    return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)

@partial(jax.jit, static_argnames=("likelihood", ))
def _lh_trafo(likelihood, primals):
    return likelihood.transformation(primals)

@partial(jax.jit, static_argnames=("likelihood", ))
def _lh_lsm(likelihood, primals, tangents):
    return likelihood.left_sqrt_metric(primals, tangents)


class OptVIState(NamedTuple):
  """Named tuple containing state information."""
  niter: int
  samples: Samples
  sampling_states: List[OptimizeResults]
  minimization_state: OptimizeResults

class OptimizeVI:
    def __init__(self, 
                 likelihood: Likelihood,
                 n_iter: int,
                 key: jax.random.PRNGKey,
                 n_samples: int,
                 sampling_method: str = 'altmetric',
                 sampling_minimizer = 'newtoncg',
                 sampling_kwargs: dict = {
                     'cg_kwargs': {},
                 },
                 minimizer: str = 'newtoncg',
                 minimization_kwargs: dict = {}):
        self._n_iter = n_iter
        self._sampling_method = sampling_method
        self._minimizer = minimizer
        self._sampling_minimizer = sampling_minimizer
        self._sampling_kwargs = sampling_kwargs
        # Only use xtol for sampling since a custom gradientnorm is used
        self._sampling_kwargs['absdelta'] = 0.
        self._mini_kwargs = minimization_kwargs
        self._keys = jax.random.split(key, n_samples)

        draw_metric = partial(_sample_linearly, likelihood, from_inverse=False)
        draw_metric = jax.vmap(draw_metric, in_axes=(None, 0), 
                               out_axes=(None, 0))
        draw_linear = partial(_sample_linearly, likelihood, from_inverse=True,
                              cg_kwargs = self._sampling_kwargs['cg_kwargs'])
        draw_linear = smap(draw_linear, in_axes=(None, 0))

        lh_trafo = partial(_lh_trafo, likelihood)
        lh_lsm = partial(_lh_lsm, likelihood)
        def nl_g(x, p, lh_trafo_at_p):
            return x - p + lh_lsm(p, lh_trafo(x) - lh_trafo_at_p)

        def nl_residual(x, p, lh_trafo_at_p, ms_at_p):
            g = nl_g(x, p, lh_trafo_at_p)
            r = ms_at_p - g
            return 0.5*dot(r, r)

        def nl_metric(primals, tangents, p, lh_trafo_at_p):
            f = partial(nl_g, p=p, lh_trafo_at_p=lh_trafo_at_p)
            _, jj = jax.jvp(f, (primals,), (tangents,))
            _, jv = jax.vjp(f, primals)
            r = jv(jj)
            return r[0]

        def nl_sampnorm(natgrad, p):
            v = vdot(natgrad, natgrad)
            tm = lambda x: lh_lsm(p, x)
            o = jax.linear_transpose(tm, likelihood.lsm_tangents_shape)
            fpp = o(natgrad)
            v += vdot(fpp, fpp)
            return jnp.sqrt(v)
        nl_vag = jax.jit(jax.value_and_grad(nl_residual))

        self._kl_vg = partial(_ham_vg, likelihood)
        self._kl_metric = partial(_ham_metric, likelihood)
        self._draw_linear = draw_linear
        self._draw_metric = draw_metric
        self._lh_trafo = lh_trafo
        self._nl_vag = nl_vag
        self._nl_metric = jax.jit(nl_metric)
        self._nl_sampnorm = jax.jit(nl_sampnorm)

    def linear_sampling(self, primals, from_inverse):
        if from_inverse:
            samples, met_smpls = self._draw_linear(primals, self._keys)
            samples = Samples(
                pos=primals, 
                samples=jax.tree_map(lambda *x: 
                                    jnp.concatenate(x), samples, -samples)
            )
        else:
            _, met_smpls = self._draw_metric(primals, self._keys)
            samples = None
        met_smpls = Samples(pos=None,
                            samples=jax.tree_map(
                lambda *x: jnp.concatenate(x), met_smpls, -met_smpls)
        )
        return samples, met_smpls

    def nonlinear_sampling(self, samples):
        primals = samples.pos
        lh_trafo_at_p = self._lh_trafo(primals)
        metric_samples = self.linear_sampling(primals, False)[1]
        new_smpls = []
        opt_states = []
        for s, ms in zip(samples, metric_samples):
            options = {
                "custom_gradnorm" : partial(self._nl_sampnorm, p=primals),
                "fun_and_grad":
                    partial(
                        self._nl_vag,
                        p=primals,
                        lh_trafo_at_p=lh_trafo_at_p,
                        ms_at_p=ms
                    ),
                "hessp":
                    partial(
                        self._nl_metric,
                        p=primals,
                        lh_trafo_at_p=lh_trafo_at_p
                    ),
                }
            opt_state = minimize(None, x0=s, method=self._sampling_minimizer, 
                                 options=self._sampling_kwargs | options)
            new_smpls.append(opt_state.x - primals)
            # Remove x from state to avoid copy of the samples
            opt_states.append(opt_state._replace(x = None))

        samples = Samples(pos=primals, samples=stack(new_smpls))
        return samples, opt_states

    def minimize_kl(self, samples):
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
            smpls = self.linear_sampling(primals, False)[1]
        else:
            smpls = self.linear_sampling(primals, True)[0]
        state = OptVIState(niter=0, samples=smpls, sampling_states=None,
                           minimization_state=None)
        return primals, state

    def update(self, primals, state):
        niter = state.niter
        if self._sampling_method in ['linear', 'geometric']:
            samples = self.linear_sampling(primals, True)[0]
        else:
            samples = state.samples.at(primals)
        if self._sampling_method in ['geometric', 'altmetric']:
            samples, sampling_states = self.nonlinear_sampling(samples)
        else:
            sampling_states = None
        samples, opt_state = self.minimize_kl(samples)
        state = OptVIState(niter=niter+1, samples=samples, 
                           sampling_states=sampling_states,
                           minimization_state=opt_state)
        return samples.pos, state

    def run(self, primals):
        primals, state = self.init_state(primals)
        for i in range(self._n_iter):
            print(f"OptVI iteration number: {i}")
            primals, state = self.update(primals, state)
        return primals, state

def _make_callable(obj):
    if callable(obj) and not isinstance(obj, int):
        return obj
    else:
        return lambda x: obj

