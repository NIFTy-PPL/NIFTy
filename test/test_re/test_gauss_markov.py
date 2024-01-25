#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pytest

from jax import random, jit
import numpy as np
import nifty8.re as jft

pmp = pytest.mark.parametrize

def _call(process, key):
    key = random.PRNGKey(key)
    rnd = process.init(key)
    process = jit(process)
    res = process(rnd)
    assert np.all(np.isfinite(res))
    process(rnd)

@pmp('x0', [1., (0., 1.5), jft.UniformPrior(-1.,1.,name='x0')])
@pmp('sigma', [0.3, (1.,0.8), jft.UniformPrior(0.1,1.1,name='sigma')])
@pmp('name', ['wiener',])
@pmp('dt, N_steps', [(0.1, 10), (np.arange(10), None), (np.ones(5), 5)])
@pmp('key', [3, 4])
def test_wp(
    x0,
    sigma,
    dt,
    name,
    N_steps,
    key):
    gp = jft.WienerProcess(x0, sigma, dt, name=name, N_steps=N_steps)
    _call(gp, key)


@pmp('x0', [np.array([0.3,-0.4]), (0., 1.5),
            jft.UniformPrior(-1.,1., name='x0', shape=(2,))])
@pmp('sigma', [0.3, (1.,0.8), jft.UniformPrior(0.1,1.1,name='sigma')])
@pmp('name', ['intwiener', ])
@pmp('dt, N_steps', [(0.1, 10), (np.arange(10), None), (np.ones(5), 5)])
@pmp('asperity', [None, 0.2, (1.1,0.5), jft.UniformPrior(0.1, 2.2, name='asp')])
@pmp('key', [3, 4])
def test_iwp(
    x0,
    sigma,
    dt,
    name,
    N_steps,
    asperity,
    key):
    gp = jft.IntegratedWienerProcess(x0, sigma, dt, name=name, N_steps=N_steps,
                                     asperity=asperity)
    _call(gp, key)


@pmp('x0', [None, 1., (0., 1.5), jft.UniformPrior(-1.,1.,name='x0')])
@pmp('sigma', [0.3, (1.,0.8), jft.UniformPrior(0.1,1.1,name='sigma')])
@pmp('name', ['ornsteinup', ])
@pmp('dt, N_steps', [(0.1, 10), (np.arange(10), None), (np.ones(5), 5)])
@pmp('gamma', [0.2, (1.1,0.5), jft.UniformPrior(0.1, 2.2, name='asp')])
@pmp('key', [3, 4])
def test_oup(
    x0,
    sigma,
    dt,
    name,
    N_steps,
    gamma,
    key):
    gp = jft.OrnsteinUhlenbeckProcess(sigma, gamma, dt, name=name, x0=x0,
                                      N_steps=N_steps)
    _call(gp, key)
