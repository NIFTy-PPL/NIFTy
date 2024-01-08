# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import Array, vmap
from jax.lax import fori_loop
from typing import Union

from .model import Model
from .tree_math import ShapeWithDtype, random_like

def _isscalar(x):
    return jnp.ndim(x) == 0

def discrete_gm_general(xi, init, drift, diffamp):
    if _isscalar(drift):
        drift = drift * jnp.ones((1, 1))
    if _isscalar(diffamp):
        diffamp = diffamp * jnp.ones((1, 1))

    in_ax = (None if len(diffamp.shape)==2 else 0,0)
    res = vmap(jnp.matmul, in_ax, 0)(diffamp, xi)

    def loop(i, a):
        d = drift[i] if len(drift.shape) > 2 else drift
        return a.at[i+1].add(jnp.matmul(d, a[i]))

    res = jnp.concatenate([init[jnp.newaxis,...], res], axis=0)
    return fori_loop(0, res.size, loop, res)

#def continuous_to_discrete(drift, amplitude):



def scalar_gm(xi, init, drift, diffamp):
    if not _isscalar(drift):
        drift = drift[:, jnp.newaxis, jnp.newaxis]
    if not _isscalar(diffamp):
        diffamp = diffamp[:, jnp.newaxis, jnp.newaxis]
    if _isscalar(init):
        init = jnp.array([init,])
    return discrete_gm_general(xi[:, jnp.newaxis], init, drift, diffamp)[:, 0]

def wiener_process(
        x0: float,
        xi: Array,
        sigma: Union[float, Array],
        dt: Union[float, Array]):
    drift = 1.
    amp = jnp.sqrt(dt)*sigma
    return scalar_gm(xi, x0, drift, amp)

def integrated_wiener_process(
        x0: Array,
        xi: Array,
        sigma: Array,
        dt: Array,
        asperity: Union[float, Array] = None):
    asperity = 0. if asperity is None else asperity
    dt = jnp.ones(xi.shape[0])*dt if _isscalar(dt) else dt
    def drift_amp(d, sig, asp):
        drift = jnp.array([[1., d], [0., 1.]])
        amp = jnp.array([[jnp.sqrt(d**2 / 12. + asp), d / 2.],
                         [0., 1.]])
        amp *= sig*jnp.sqrt(d)
        return drift, amp
    axs = (0,
           None if _isscalar(sigma) else 0,
           None if _isscalar(asperity) else 0)
    drift, amp = vmap(drift_amp, axs, (0,0))(dt, sigma, asperity)
    return discrete_gm_general(xi, x0, drift, amp)

def ou_process(
        x0: float,
        xi: Array,
        alpha: Union[float, Array],
        gamma: Union[float, Array],
        dt: Union[float, Array]):
    drift = jnp.exp(-gamma*dt)
    amp = alpha * jnp.sqrt(1. - drift**2)
    return scalar_gm(xi, x0, drift, amp)

def stationary_init_ou(
        xi: Array,
        alpha: Union[float, Array],
        gamma: Union[float, Array],
        dt: Union[float, Array]):
    x0 = (alpha if _isscalar(alpha) else alpha[0]) * xi[0]
    return ou_process(x0, xi[1:], alpha, gamma, dt)


class WienerProcess(Model):
    def __init__(self,
                 x0: Union[float, Model],
                 sigma: Union[float, Array, Model],
                 dt: Union[float, Array],
                 name: str = 'wpxi',
                 N_steps: int = None):
        if _isscalar(dt):
            dt = np.ones(N_steps) * dt
        shape = dt.shape
        domain = {name: ShapeWithDtype(shape, float)}
        init =  partial(random_like, primals=domain)
        for a in [x0, sigma]:
            if isinstance(a, Model):
                domain = domain | a.domain
                init = init | a.init

        def call(x):
            sig = sigma(x) if isinstance(sigma, Model) else sigma
            xinit = x0(x) if isinstance(x0, Model) else x0
            xi = x[name]
            return wiener_process(xinit, xi, sig, dt)

        super().__init__(call=call, domain=domain, init=init)


class OUProcess(Model):
    def __init__(self,
                 alpha: Union[float, Array, Model],
                 gamma: Union[float, Array, Model],
                 dt: Union[float, Array],
                 name: str = 'ouxi',
                 N_steps: int = None):
        if _isscalar(dt):
            dt = np.ones(N_steps) * dt
        domain = {name: ShapeWithDtype((dt.size + 1,), float)}
        #init =  {name: partial(random_like, primals=domain)}
        for a in [alpha, gamma]:
            if isinstance(a, Model):
                domain = domain | a.domain
        #        init = init | a.init
        self.alpha = alpha

        def call(x):
            al = alpha(x) if isinstance(alpha, Model) else alpha
            gam = gamma(x) if isinstance(gamma, Model) else gamma
            xi = x[name]
            return stationary_init_ou(xi, al, gam, dt)

        super().__init__(call=call, domain=domain)#, init=init)