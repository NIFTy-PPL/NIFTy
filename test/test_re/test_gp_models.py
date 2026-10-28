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


def test_icrgp_coordinate_kernel():
    """A covariance function of coordinates gives the same field as its table
    (fine table, so the interpolation error is negligible); a model returning
    such a function is hierarchical."""
    length = 7.0

    def k(x, y):  # (dim, ...) coordinate arrays
        return jnp.exp(-0.5 * jnp.sum((x - y) ** 2, axis=0) / length**2)

    rs = jnp.concatenate((jnp.zeros(1), jnp.geomspace(1e-3, 200.0, 4000)))
    gp_fun = jft.ICRGP(_grid(), k)
    gp_tab = jft.ICRGP(_grid(), (rs, jnp.exp(-0.5 * (rs / length) ** 2)))
    assert gp_fun.coordinate_kernel and not gp_fun.hierarchical and gp_fun.radial_bins is None
    assert set(gp_fun.domain) == {"xi"}
    p = jft.random_like(jax.random.PRNGKey(0), gp_fun.domain)
    assert_allclose(gp_fun(p), gp_tab(p), rtol=1e-3, atol=1e-4)  # the table is interpolated linearly
    jax.jit(jax.grad(lambda p: (gp_fun(p) ** 2).sum()))(p)
    with pytest.raises(TypeError):
        jft.GraphGP(_points(), k, n0=8, k=4, cuda=False) if "graphgp" in _models() else (_ for _ in ()).throw(TypeError())


def test_matern_model_as_kernel():
    """`MaternCovarianceModel(mode="table")` is a kernel model of both GPs; the
    deprecated mode "ICR" (a coordinate kernel) gives the same ICR field."""
    kw = dict(Ndim=1, r_min=0.02, r_max=300.0, variance=(1.0, 0.2), lengthscale=(5.0, 0.5), negloglogslope=(3.0, 0.3), Ninterp=512, prefix="m_")
    with pytest.warns(DeprecationWarning, match="add_fluctuations_matern"):
        table = jft.MaternCovarianceModel(mode="table", **kw)
    assert table.radial_bins[0] == 0.0 and table.radial_bins.shape == (513,)
    for name, make in _models().items():
        gp = make(table)
        assert gp.hierarchical and {"m_variance", "m_lengthscale", "m_negloglogslope"} <= set(gp.domain)
        p = jft.random_like(jax.random.PRNGKey(1), gp.domain)
        f = gp(p)
        assert f.shape == gp.shape and np.all(np.isfinite(f))
        if name == "icr":
            with pytest.warns(DeprecationWarning, match="Mode 'ICR'"):
                icr_mode = jft.MaternCovarianceModel(mode="ICR", **kw)
            gp_fun = jft.ICRGP(_grid(), icr_mode)
            assert gp_fun.coordinate_kernel and gp_fun.hierarchical
            assert_allclose(gp_fun(p), f, rtol=1e-10, atol=1e-10)


def test_icrfield_deprecated():
    with pytest.warns(DeprecationWarning, match="ICRGP"):
        jft.ICRField(_grid(12), lambda x, y: jnp.exp(-0.5 * jnp.sum((x - y) ** 2, axis=0)), compress=dict(rtol=1e-5, atol=1e-10, buffer_size=1000, use_distances=True))


@pytest.mark.skipif(jax.device_count() < 2, reason="needs several devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
def test_icrgp_sharded_matches_unsharded():
    """ICRGP on a one-dimensional mesh (halo exchange per level) equals the
    unsharded model: forward, gradients (excitations and kernel parameters)
    and the adjointness of its derivatives."""
    from jax.sharding import Mesh

    grid = jft.HPLogRGrid(nside=4, r_min_shape=4, r_min=1.0, r_max=10.0, nside0=1)
    rs = jnp.asarray(np.concatenate(([0.0], np.geomspace(0.01, 50.0, 511))))

    class Cov(jft.Model):
        radial_bins = rs

        def __init__(self):
            super().__init__(domain={"scale": jft.ShapeWithDtype(())})

        def __call__(self, p):
            return (p["scale"] ** 2) * jnp.exp(-0.5 * (rs / 2.0) ** 2).at[0].mul(1.001)

    mesh = Mesh(np.array(jax.devices()), ("x",))
    ref = jft.ICRGP(grid, Cov())
    sharded = jft.ICRGP(grid, Cov(), mesh=mesh)
    assert sharded.excitation_shape == ref.excitation_shape and sharded.shape == ref.shape
    assert [t.sharded for t in sharded._shard_levels] == [True, True] and sharded._shard_levels[0].coarse_full
    p = jft.random_like(jax.random.PRNGKey(0), ref.domain)
    p["scale"] = jnp.asarray(1.2)
    ps = sharded.shard(p)
    f_ref, f_sh = ref(p), sharded(ps)
    assert f_sh.sharding.is_equivalent_to(sharded.sharding, f_sh.ndim)
    assert_allclose(f_sh, f_ref, rtol=1e-12, atol=1e-12)
    w = jax.random.normal(jax.random.PRNGKey(3), f_ref.shape)
    g_ref = jax.grad(lambda p: jnp.sum(w * ref(p)))(p)
    g_sh = jax.jit(jax.grad(lambda p: jnp.sum(w * sharded(p))))(ps)
    assert_allclose(g_sh["scale"], g_ref["scale"], rtol=1e-10)
    assert_allclose(sharded.relayout(np.asarray(g_sh["xi"]), inverse=True), g_ref["xi"], rtol=1e-10, atol=1e-12)
    v = jft.random_like(jax.random.PRNGKey(5), sharded.domain)
    _, jv = jax.jvp(sharded, (ps,), (v,))
    _, vjp = jax.vjp(sharded, ps)
    (jt,) = vjp(w)
    lhs, rhs = jnp.vdot(jv, w), sum(jnp.vdot(v[k], jt[k]) for k in v)
    assert_allclose(lhs, rhs, rtol=1e-10)
    # the relayout is a permutation and its own inverse composition
    xi = p["xi"]
    assert_allclose(sharded.relayout(sharded.relayout(xi), inverse=True), xi)
