# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.lax import fori_loop
from typing import Callable, Iterable, Optional, Union

from .model import Model
from .tree_math import ShapeWithDtype, random_like


def wiener_process(
        x0: float,
        xi: Array,
        sigma: Union[float, Array],
        dt: Union[float, Array]):
    dts = np.sqrt(dt)
    if isinstance(dts, float):
        dts = np.ones(xi.size)*dts
    else:
        if dts.size != xi.size:
            raise ValueError("Timesteps and excitations are incompatible")
    res = jnp.cumsum(sigma*dts*xi) + x0
    return jnp.concatenate((jnp.array([x0,]), res))


def ou_process(
        x0: float,
        xi: Array,
        alpha: Union[float, Array],
        gamma: Union[float, Array],
        dt: Union[float, Array]):
    if isinstance(dt, float):
        dt = np.ones(xi.size)*dt
    else:
        if dt.size != xi.size:
            raise ValueError("Timesteps and excitations are incompatible")

    bias = jnp.exp(-gamma*dt)
    wgt = alpha * jnp.sqrt(1. - bias**2)

    r = wgt * xi
    res = jnp.concatenate((jnp.array([x0,]), jnp.zeros_like(r)))
    def loop(i, a):
        a = a.at[i+1].set(bias[i] * a[i] + r[i])
        return a
    return fori_loop(0, res.size, loop, res)


def stationary_ou(
        xi: Array,
        alpha: Union[float, Array],
        gamma: Union[float, Array],
        dt: Union[float, Array]):
    x0 = (alpha if isinstance(alpha, float) else alpha[0]) * xi[0]
    xi = xi[1:]
    return ou_process(x0, xi, alpha, gamma, dt)


class WienerProcess(Model):
    def __init__(self,
                 x0: Union[float, Model],
                 sigma: Union[float, Array, Model],
                 dt: Union[float, Array],
                 name: str = 'wpxi',
                 N_steps: int = None):
        if isinstance(dt, float):
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
        if isinstance(dt, float):
            dt = np.ones(N_steps) * dt
        shape = dt.shape
        domain = {name: ShapeWithDtype(shape, float)}
        init =  partial(random_like, primals=domain)
        for a in [alpha, gamma]:
            if isinstance(a, Model):
                domain = domain | a.domain
                init = init | a.init

        def call(x):
            al = alpha(x) if isinstance(alpha, Model) else alpha
            gam = gamma(x) if isinstance(gamma, Model) else gamma
            xi = x[name]
            return stationary_ou(xi, al, gam, dt)

        super().__init__(call=call, domain=domain, init=init)