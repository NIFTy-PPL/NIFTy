#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
import numpy as np
import pytest
import jax.numpy as jnp
from jax import random, vmap
from numpy.testing import assert_allclose

import nifty8.re as jft

pmp = pytest.mark.parametrize


def _call(process, seed):
    p = process.init(random.PRNGKey(seed))
    process = jax.jit(process)
    assert np.all(np.isfinite(process(p)))


@pmp('x0', [0.1, -2.2])
@pmp('dts', [np.ones((10, )) * 0.2, np.array([0.2, 0.5, 0.1])])
@pmp('gamma', [0.1, 0.5, 0.2])
def test_decay_ornstein_uhlenbeck(x0, dts, gamma):
    ts = np.zeros(dts.size + 1)
    ts[1:] = np.cumsum(dts)
    res = np.exp(-gamma * ts) * x0
    gp = jft.OrnsteinUhlenbeckProcess(1., gamma, dts, x0=x0)
    myres = gp(jft.zeros_like(gp.domain))
    assert_allclose(res, myres)


@pmp('x0', [0.1, -1.2])
@pmp('dts', [np.ones((10, )) * 0.2, np.array([0.2, 0.5, 0.1])])
@pmp('seed', [
    42,
])
def test_const_wiener(x0, dts, seed):
    gp = jft.WienerProcess(x0, 1., dts)
    myres = gp(jft.zeros_like(gp.domain))
    assert_allclose(myres, x0)
    # Compare to cumsum
    p = gp.init(random.PRNGKey(seed))
    res = x0 + np.cumsum(p['wp'] * np.sqrt(dts))
    gpres = gp(p)
    assert_allclose(res, gpres[1:])

    amp = np.sqrt(dts)
    res2 = jft.gauss_markov.scalar_gauss_markov_process(p['wp'], x0, 1., amp)
    assert_allclose(res2, gpres)


@pmp('x0', [np.array([0.1, 0.5]), np.array([-1.2, 0.7])])
@pmp('dts', [np.ones((10, )) * 0.2, np.array([0.2, 0.5, 0.1])])
def test_drift_integrated_wiener(x0, dts):
    ts = np.zeros(dts.size + 1)
    ts[1:] = np.cumsum(dts)
    res = x0[1] * ts + x0[0]
    gp = jft.IntegratedWienerProcess(x0, 1., dts)
    myres = gp(jft.zeros_like(gp.domain))
    # Check if it is a straight line
    assert_allclose(myres[:, 0], res)
    # Check that derivative is const.
    assert_allclose(myres[:, 1], x0[1])


@pmp('dts', [np.ones((10, )) * 0.2, np.array([0.2, 0.5, 0.1])])
def test_iwp_cumsum_vs_fori(dts):
    """Implements the (generalized) Integrated Wiener process (IWP)."""
    x0 = np.array([0.1, 0.5])
    sigma = 1.
    asperity = 0.1
    gp = jft.IntegratedWienerProcess(x0, sigma, dts, asperity=asperity)

    rnd = gp.init(random.PRNGKey(10))
    res = gp(rnd)
    xi = rnd['iwp']

    def drift_amp(d, sig, asp):
        drift = jnp.array([[1., d], [0., 1.]])
        amp = jnp.array([[jnp.sqrt(d**2 / 12. + asp), d / 2.], [0., 1.]])
        amp *= sig * jnp.sqrt(d)
        return drift, amp

    axs = (0, None, None)
    drift, amp = vmap(drift_amp, axs, (0, 0))(dts, sigma, asperity)
    res2 = jft.gauss_markov.discrete_gauss_markov_process(xi, x0, drift, amp)

    assert_allclose(res, res2)


@pmp('x0', [1., (0., 1.5), jft.UniformPrior(-1., 1., name='x0')])
@pmp('sigma', [0.3, (1., 0.8), jft.UniformPrior(0.1, 1.1, name='sigma')])
@pmp('name', [
    'wiener',
])
@pmp('dt, N_steps', [(0.1, 10), (np.arange(10), None), (np.ones(5), 5)])
@pmp('seed', [3, 4])
def test_wiener_eval(x0, sigma, dt, name, N_steps, seed):
    gp = jft.WienerProcess(x0, sigma, dt, name=name, N_steps=N_steps)
    _call(gp, seed)


@pmp(
    'x0', [
        np.array([0.3, -0.4]), (0., 1.5),
        jft.UniformPrior(-1., 1., name='x0', shape=(2, ))
    ]
)
@pmp('sigma', [0.3, (1., 0.8), jft.UniformPrior(0.1, 1.1, name='sigma')])
@pmp('name', [
    'intwiener',
])
@pmp('dt, N_steps', [(0.1, 10), (np.arange(10), None), (np.ones(5), 5)])
@pmp(
    'asperity',
    [None, 0.2,
     (1.1, 0.5), jft.UniformPrior(0.1, 2.2, name='asp')]
)
@pmp('seed', [3, 4])
def test_integrated_wiener_eval(x0, sigma, dt, name, N_steps, asperity, seed):
    gp = jft.IntegratedWienerProcess(
        x0, sigma, dt, name=name, N_steps=N_steps, asperity=asperity
    )
    _call(gp, seed)


@pmp('x0', [None, 1., (0., 1.5), jft.UniformPrior(-1., 1., name='x0')])
@pmp('sigma', [0.3, (1., 0.8), jft.UniformPrior(0.1, 1.1, name='sigma')])
@pmp('name', [
    'ornsteinup',
])
@pmp('dt, N_steps', [(0.1, 10), (np.arange(10), None), (np.ones(5), 5)])
@pmp('gamma', [0.2, (1.1, 0.5), jft.UniformPrior(0.1, 2.2, name='asp')])
@pmp('seed', [3, 4])
def test_ornstein_uhlenbeck_eval(x0, sigma, dt, name, N_steps, gamma, seed):
    gp = jft.OrnsteinUhlenbeckProcess(
        sigma, gamma, dt, name=name, x0=x0, N_steps=N_steps
    )
    _call(gp, seed)
