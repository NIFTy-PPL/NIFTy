#!/usr/bin/env python3
# Copyright(C) 2013-2023 Philipp Frank
# SPDX-License-Identifier: BSD-2-Clause

import sys
import jax
import jax.numpy as jnp
from functools import partial

from .optimize import minimize
from .tree_math.vector_math import dot, vdot
from .tree_math.forest_math import stack, unstack
from .likelihood import StandardHamiltonian
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


class OptimizeVI:
    def __init__(self, likelihood, sampling_params, opt_params, N_steps):
        self._sampling_params = sampling_params
        self._opt_params = opt_params
        self.init_with_likelihood(likelihood)
        self.N_steps = int(N_steps)

    def _init_minimize(self, likelihood):
        self._kl_vg = partial(_ham_vg, likelihood)
        self._kl_metric = partial(_ham_metric, likelihood)

    def _init_sampling(self, likelihood, cg_kwargs):
        draw_metric = jax.vmap(draw_metric, in_axes=(None, 0), 
                                        out_axes=(None, 0))
        draw_linear = partial(
                _sample_linearly,
                likelihood,
                from_inverse=True,
                cg_kwargs=cg_kwargs
            )
        self._draw_linear = draw_linear
        self._draw_metric = draw_metric

    def _init_geo(self, likelihood):
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
        self._lh_trafo = lh_trafo
        self._nl_vag = nl_vag
        self._nl_metric = jax.jit(nl_metric)
        self._nl_sampnorm = jax.jit(nl_sampnorm)

    def _linear_sampling(self, primals, keys):
        samples, met_smpls = self._draw_linear(primals, keys)
        samples = Samples(
            pos=primals, 
            samples=jax.tree_map(lambda *x: 
                                    jnp.concatenate(x), samples, -samples)
        )
        met_smpls = Samples(pos=None,
                            samples=jax.tree_map(
                lambda *x: jnp.concatenate(x), met_smpls, -met_smpls)
        )
        return samples, met_smpls

    def _update_samples(self, samples, keys):
        if len(samples) != len(keys):
            raise ValueError("Length mismatch between samples and keys.")
        method = self._sampling_params['method']
        if method not in ['linear', 'geometric', 'altmetric']:
            raise ValueError(f"Unknown sampling method: {method}")

        primals = samples.pos
        if (method == 'linear') or (method == 'geometric'):
            samples, met_smpls = self._linear_sampling(primals, keys)
        elif method == 'altmetric':
            _, met_smpls = self._draw_metric(primals, keys)

        if (method == 'geometric') or (method == 'altmetric'):
            options = self._sampling_params
            lh_trafo_at_p = self._lh_trafo(primals)
            new_smpls = []
            for s, ms in zip(samples, met_smpls):
                options = options.update({
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
                    })
                opt_state = minimize(
                    None, x0=s, method=self._sampling_params['method'], 
                    options=options
                )
                new_smpls += [opt_state.x - primals]
            samples = Samples(pos=primals, samples=stack(new_smpls))
        return samples

    def _minimize_kl(self, samples):
        options = self._opt_params
        options = options.update({
            "fun_and_grad": partial(self._vg, primals_samples=samples),
            "hessp": partial(self._metric, primals_samples=samples),
        })
        opt_state = minimize(
            None,
            samples.pos,
            method=options['method'],
            options=options
        )
        return samples.at(opt_state.x)

    def init_with_likelihood(self, likelihood):
        self._init_sampling(likelihood, self._sampling_params['cg_kwargs'])
        self._init_minimize(likelihood)
        self._init_geo(likelihood)

    def update_params(self, sampling_params = {}, opt_params = {}):
        self._sampling_params.update(sampling_params)
        self._opt_params.update(opt_params)

    def init_state(self, primals, n_samples, key):
        keys = jax.random.split(key, n_samples)
        return primals, self._linear_sampling(primals, keys)[0]

    def update(self, primals, samples, key):
        samples = samples.at(primals)
        keys = jax.random.split(key, len(samples) // 2)
        samples = self._update_samples(samples, keys)
        samples = self._minimize_kl(samples)
        return samples.pos, samples

    def run(self, primals, n_samples, key):
        if self._sampling_params['resample']:
            keys = jax.random.split(key, self.N_steps)
        else:
            keys = (key, ) * self.N_steps
        primals, samples = self.init_state(primals, n_samples, keys[0])
        for i in range(self.N_steps):
            primals, samples = self.update(primals, samples, keys[i])
        return primals, samples