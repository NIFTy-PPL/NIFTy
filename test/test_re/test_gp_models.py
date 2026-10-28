#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

"""The general Gaussian process models `ICRGP` and `GraphGP` with a fixed or a
model kernel, and their composition in the correlated field."""

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from numpy.testing import assert_allclose

pmp = pytest.mark.parametrize
jax.config.update("jax_enable_x64", True)

RS = jnp.asarray(np.concatenate(([0.0], np.geomspace(0.1, 200.0, 255))))


def _matern(r, scale, length):
    """Matern-3/2 covariance; the length of 20 is resolved by the coarsest
    level of the test grid (ICR's accuracy is set by that)."""
    x = np.sqrt(3.0) * r / length
    return scale**2 * (1.0 + x) * jnp.exp(-x)


def _table(scale=2.0, length=20.0, jitter=1e-3):
    return _matern(RS, scale, length).at[0].mul(1.0 + jitter)


class _Kernel(jft.Model):
    """Matern-3/2 kernel on `RS` with a log-normal scale and length."""

    radial_bins = RS

    def __init__(self, prefix=""):
        self.scale = jft.LogNormalPrior(2.0, 0.5, name=prefix + "scale")
        self.length = jft.LogNormalPrior(20.0, 3.0, name=prefix + "length")
        super().__init__(domain=self.scale.domain | self.length.domain, init=self.scale.init | self.length.init)

    def __call__(self, p):
        return _matern(RS, self.scale(p), self.length(p)).at[0].mul(1.0 + 1e-3)


def _grid(n=40):
    return jft.SimpleOpenGrid(min_shape=(n,), window_size=3, splits=2, distances=1.0, desired_size0=8)


def _points():
    return np.random.default_rng(0).uniform(0.0, 10.0, (200, 2))


def _models():
    out = {"icr": lambda kernel, **kw: jft.ICRGP(_grid(), kernel, **kw)}
    try:
        import graphgp  # noqa: F401

        out["graphgp"] = lambda kernel, **kw: jft.GraphGP(_points(), kernel, n0=32, k=12, cuda=False, **kw)
    except ImportError:
        pass
    return out


def _coordinates(gp):
    if isinstance(gp, jft.ICRGP):
        g = gp.multigrid.at(gp.depth)
        idx = np.mgrid[tuple(slice(0, s) for s in g.shape)].reshape(g.ndim, -1)
        return np.asarray(g.index2coord(idx)).T
    return np.asarray(gp.points).reshape(-1, np.asarray(gp.points).shape[-1])


@pmp("name", list(_models()))
def test_fixed_and_model_kernel(name):
    make = _models()[name]
    cov = _table()
    fixed = make((RS, cov))
    fixed2 = make(cov, radial_bins=RS)
    hier = make(_Kernel("k_"), prefix="h_")
    assert set(fixed.domain) == {"xi"} and set(fixed2.domain) == {"xi"}
    assert set(hier.domain) == {"h_xi", "k_scale", "k_length"}
    assert not fixed.hierarchical and hier.hierarchical
    assert fixed.domain["xi"].shape == fixed.excitation_shape
    p = jft.random_like(jax.random.PRNGKey(0), fixed.domain)
    f = jax.jit(fixed)(p)
    assert f.shape == fixed.shape and np.all(np.isfinite(f))
    assert_allclose(fixed2(p), f, rtol=1e-12, atol=1e-12)
    # the model kernel gives the same field as a fixed table of its values
    q = {"h_xi": p["xi"], "k_scale": jnp.asarray(0.3), "k_length": jnp.asarray(-0.2)}
    fixed_q = make((RS, hier.covariance(q)))
    assert_allclose(hier(q), fixed_q({"xi": p["xi"]}), rtol=1e-12, atol=1e-12)
    assert not np.allclose(hier(q), f)
    # generate is linear and batched over leading axes
    xi = p["xi"]
    assert_allclose(fixed.generate(2.0 * xi, cov), 2.0 * f, rtol=1e-12)
    fb = fixed.generate(jnp.stack([xi, -xi]), cov)
    assert fb.shape == (2,) + fixed.shape
    assert_allclose(fb[1], -f, rtol=1e-12, atol=1e-12)


@pmp("name", list(_models()))
def test_covariance_of_generated_field(name):
    gp = _models()[name]((RS, _table()))
    cov = np.asarray(gp.covariance())
    A = jax.jacfwd(lambda xi: gp.generate(xi, gp.cov_vals).ravel())(jnp.zeros(gp.excitation_shape))
    C = np.asarray(A @ A.T)
    coords = _coordinates(gp)
    r = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    truth = np.interp(r, RS, cov)
    assert np.linalg.norm(C - truth) / np.linalg.norm(truth) < 3e-2
    assert_allclose(np.mean(np.diag(C)), cov[0], rtol=2e-2)


@pmp("name", list(_models()))
def test_jvp_vjp_adjoint_and_pytree(name):
    gp = _models()[name](_Kernel("k_"), prefix="h_")
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(3), 3)
    p = jft.random_like(k0, gp.domain)
    v = jft.random_like(k1, gp.domain)
    w = jax.random.normal(k2, gp.target.shape, dtype=gp.target.dtype)
    _, jv = jax.jvp(gp, (p,), (v,))
    _, vjp = jax.vjp(gp, p)
    (jt,) = vjp(w)
    lhs = jnp.vdot(jv, w)
    rhs = sum(jnp.vdot(v[k], jt[k]) for k in v)
    assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10 * abs(float(lhs)))
    # a pytree: passed as a jit argument, no large constants and a single trace
    consts = jax.make_jaxpr(lambda gp, p: gp(p))(gp, p).consts
    assert sum(int(np.asarray(c).nbytes) for c in consts) < 2**16
    traces = []

    @jax.jit
    def f(gp, p):
        traces.append(1)
        return gp(p)

    assert_allclose(f(gp, p), gp(p), rtol=1e-12, atol=1e-12)
    f(gp, p)
    assert len(traces) == 1


@pmp("name", list(_models()))
def test_correlated_field_composes_models(name):
    """A single-factor correlated field equals the manual composition of its
    spectral kernel and its GP model's `generate` with the zero mode."""
    if name == "icr":
        space = jft.ICRSpace(_grid(), n_bins=256)
    else:
        space = jft.GraphGPSpace(_points(), n0=32, k=12, n_bins=256, cuda=False)
    m = jft.CorrelatedFieldMaker("cf")
    m.set_amplitude_total_offset(offset_mean=0.3, offset_std=(1.0, 0.1))
    m.add_fluctuations(space, fluctuations=(2.0, 0.1), loglogavgslope=(-3.0, 0.3), flexibility=(0.5, 0.1), prefix="s")
    cf = m.finalize()
    assert len(cf.gps) == 1 and len(cf.kernels) == 1
    gp, kernel = cf.gps[0], cf.kernels[0]
    assert isinstance(gp, jft.ICRGP if name == "icr" else jft.GraphGP)
    assert isinstance(kernel, jft.SpectralKernel) and gp.kernel is kernel
    assert set(kernel.domain) <= set(cf.domain) and gp.xi_name not in cf.domain
    p = jft.random_like(jax.random.PRNGKey(5), cf.domain)
    cov = kernel(p)
    assert_allclose(cov, space.covariance(m._fluctuations[0](p)), rtol=1e-12)
    xi = p["cfxi"]
    zm = m.azm(p)
    manual = 0.3 + zm * (xi[0] + gp.generate(xi[1:], cov) / zm)
    assert_allclose(cf(p), manual, rtol=1e-12, atol=1e-12)
    # the same GP model used standalone with the kernel model
    q = {gp.xi_name: xi[1:], **{k: p[k] for k in kernel.domain}}
    assert_allclose(gp(q), gp.generate(xi[1:], cov), rtol=1e-12, atol=1e-12)
    assert_allclose(jft.build_gp(space, (RS, _table())).generate(xi[1:], _table()), jft.build_gp(space, (RS, _table()))(dict(xi=xi[1:])))


def test_icr_excitation_layout_and_table_warning(caplog):
    gp = jft.ICRGP(_grid(), (RS, _table()))
    sizes = [int(np.prod(gp.multigrid.at(l).shape)) for l in range(gp.depth + 1)]
    assert gp.excitation_shape == (sum(sizes),)
    xi = jnp.arange(sum(sizes), dtype=float)
    parts = gp.split_excitations(xi)
    assert [tuple(x.shape) for x in parts] == [tuple(int(s) for s in gp.multigrid.at(l).shape) for l in range(gp.depth + 1)]
    assert_allclose(jnp.concatenate([x.ravel() for x in parts]), xi)
    # a table that ends before the largest window distance is reported
    short = jnp.asarray(np.concatenate(([0.0], np.geomspace(0.1, 5.0, 127))))
    with caplog.at_level("WARNING", logger="nifty.re.logger"):
        jft.ICRGP(_grid(), (short, jnp.exp(-0.5 * (short / 4.0) ** 2)))
    assert any("covariance table ends" in r.getMessage() for r in caplog.records)
