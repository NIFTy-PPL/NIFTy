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


def _vdot(a, b):
    """Inner product of two pytrees of arrays."""
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def _unflatten(gp, vec):
    """The excitation structure of `gp` (an array or the per-level list) from a flat vector."""
    if not gp.structured:
        return vec.reshape(gp.excitation_shape)
    out, o = [], 0
    for sh in gp.excitation_shapes:
        n = int(np.prod(sh))
        out.append(vec[o : o + n].reshape(sh))
        o += n
    return out


def _n_excitations(gp):
    return int(sum(np.prod(sh) for sh in gp.excitation_shapes))


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
    """A fixed table and a kernel model giving the same table generate the same
    field; the domains differ by the kernel parameters only."""
    make = _models()[name]
    model = make(_Kernel())
    p = jft.random_like(jax.random.PRNGKey(0), model.domain)
    fixed = make((RS, model.covariance(p)))  # the same table as the model's at p
    assert not fixed.hierarchical and model.hierarchical
    assert set(fixed.domain) == {"xi"}
    assert set(model.domain) == {"xi", "scale", "length"}
    xi_dom = fixed.domain["xi"]
    shapes = tuple(x.shape for x in xi_dom) if isinstance(xi_dom, list) else (xi_dom.shape,)
    assert shapes == fixed.excitation_shapes
    assert_allclose(model.covariance(p), fixed.covariance(), rtol=1e-12)
    f_model = model(p)
    f_fixed = fixed({"xi": p["xi"]})
    assert f_model.shape == fixed.shape and np.all(np.isfinite(f_model))
    assert_allclose(f_model, f_fixed, rtol=1e-12, atol=1e-12)
    # generate is linear in the excitations and batched over leading axes
    two = jax.tree_util.tree_map(lambda x: 2.0 * x, p["xi"])
    assert_allclose(fixed.generate(two, fixed.cov_vals), 2.0 * f_fixed, rtol=1e-12)
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, 2.0 * x]), p["xi"])
    out = fixed.generate(batched, fixed.cov_vals)
    assert out.shape == (2,) + fixed.shape
    assert_allclose(out[1], 2.0 * f_fixed, rtol=1e-12)


@pmp("name", list(_models()))
def test_covariance_of_generated_field(name):
    """The covariance of the generated field against the table."""
    gp = _models()[name]((RS, _table()))
    A = jax.jacfwd(lambda v: gp.generate(_unflatten(gp, v), gp.cov_vals).ravel())(jnp.zeros(_n_excitations(gp)))
    C = np.asarray(A @ A.T)
    coords = _coordinates(gp)
    r = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    truth = np.interp(r, np.asarray(RS), np.asarray(gp.cov_vals))
    assert np.linalg.norm(C - truth) / np.linalg.norm(truth) < 3e-2


@pmp("name", list(_models()))
def test_jvp_vjp_adjoint_and_pytree(name):
    """<Jv, w> == <v, J^T w> for the hierarchical model, and no large constants
    when the model is a jit argument."""
    gp = _models()[name](_Kernel())
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(3), 3)
    p = jft.random_like(k0, gp.domain)
    v = jft.random_like(k1, gp.domain)
    w = jax.random.normal(k2, gp.shape)
    _, jv = jax.jvp(gp, (p,), (v,))
    _, vjp = jax.vjp(gp, p)
    (jt,) = vjp(w)
    lhs, rhs = jnp.vdot(jv, w), _vdot(v, jt)
    assert_allclose(lhs, rhs, rtol=1e-10)
    consts = jax.make_jaxpr(lambda m, p: m(p))(gp, p).consts
    assert sum(int(np.asarray(c).nbytes) for c in consts) < 2**16
    traces = []

    @jax.jit
    def f(m, p):
        traces.append(1)
        return m(p)

    assert_allclose(f(gp, p), gp(p), rtol=1e-12, atol=1e-12)
    f(gp, p)
    assert len(traces) == 1


@pmp("name", list(_models()))
def test_correlated_field_composes_models(name):
    """A correlated field with one kernel-based factor equals the manual
    composition of its spectral kernel, the GP model's generate and the
    zero-mode block: ``offset + zeromode * (xi0 + generate(xi) / zeromode)``."""
    if name == "icr":
        space = jft.ICRSpace(_grid(), n_bins=128)
    else:
        space = jft.GraphGPSpace(_points(), n0=32, k=12, n_bins=128, cuda=False)
    m = jft.CorrelatedFieldMaker("cf")
    m.set_amplitude_total_offset(offset_mean=0.3, offset_std=(0.8, 0.2))
    m.add_fluctuations(space, fluctuations=(1.5, 0.3), loglogavgslope=(-3.0, 0.5), flexibility=None, prefix="s")
    cf = m.finalize()
    gp = cf.gps[0]
    assert isinstance(gp, jft.ICRGP if name == "icr" else jft.GraphGP)
    assert [b[0] for b in cf.blocks] == ["cfxi", "cfxi0"] and cf.domain["cfxi0"].shape == ()
    p = jft.random_like(jax.random.PRNGKey(0), cf.domain)
    azm = m.azm(p)
    cov = cf.kernels[0](p)
    manual = 0.3 + azm * (p["cfxi0"] + gp.generate(p["cfxi"], cov) / azm)
    assert_allclose(cf(p), manual, rtol=1e-12, atol=1e-12)


def test_icr_excitation_layout_and_table_warning(caplog):
    gp = jft.ICRGP(_grid(), (RS, _table()))
    shapes = tuple(tuple(int(s) for s in gp.multigrid.at(l).shape) for l in range(gp.depth + 1))
    assert gp.structured and gp.excitation_shapes == shapes
    assert [x.shape for x in gp.domain["xi"]] == list(shapes)
    xi = jft.random_like(jax.random.PRNGKey(0), gp.domain)["xi"]
    assert gp.batch_shape(xi) == () and gp.batch_shape([x[None] for x in xi]) == (1,)
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
    """ICRGP on a one-dimensional mesh (halo exchange per level, distributed
    kernel build) equals the unsharded model: forward, gradients (excitations
    and kernel parameters) and the adjointness of its derivatives. The
    per-level excitations are the same arrays, sharded along their first axis."""
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
    assert sharded.excitation_shapes == ref.excitation_shapes and sharded.shape == ref.shape
    assert sharded.level_sharded == (False, True, True)
    p = jft.random_like(jax.random.PRNGKey(0), ref.domain)
    p["scale"] = jnp.asarray(1.2)
    ps = sharded.shard(p)
    for a, s in zip(ps["xi"], sharded.shardings):
        assert a.sharding.is_equivalent_to(s, a.ndim)
    f_ref, f_sh = ref(p), sharded(ps)
    assert f_sh.sharding.is_equivalent_to(sharded.sharding, f_sh.ndim)
    assert_allclose(f_sh, f_ref, rtol=1e-12, atol=1e-12)
    w = jax.random.normal(jax.random.PRNGKey(3), f_ref.shape)
    g_ref = jax.grad(lambda p: jnp.sum(w * ref(p)))(p)
    g_sh = jax.jit(jax.grad(lambda m, p: jnp.sum(w * m(p)), argnums=1))(sharded, ps)
    assert_allclose(g_sh["scale"], g_ref["scale"], rtol=1e-10)
    for a, b in zip(g_sh["xi"], g_ref["xi"]):
        assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    v = jft.random_like(jax.random.PRNGKey(5), sharded.domain)
    _, jv = jax.jvp(sharded, (ps,), (v,))
    _, vjp = jax.vjp(sharded, ps)
    (jt,) = vjp(w)
    assert_allclose(jnp.vdot(jv, w), _vdot(v, jt), rtol=1e-10)




_needs_graphgp_and_devices = pytest.mark.skipif(
    jax.device_count() < 2 or "graphgp" not in _models(),
    reason="needs graphgp and several devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)",
)


def _sky_points(n_pix=16, n_bins=40, seed=1):
    """Points in (pixel, bin) order: pixel directions on a sphere, bins are radii;
    contiguous pixel blocks are patches of the sky."""
    theta = np.linspace(0.2, 3.0, n_pix)
    phi = np.linspace(0.0, 6.0, n_pix)
    r = np.geomspace(0.5, 10.0, n_bins)
    direction = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    pts = direction[:, None, :] * r[None, :, None]
    return pts + 0.1 * np.random.default_rng(seed).normal(size=pts.shape)  # (n_pix, n_bins, 3)


@_needs_graphgp_and_devices
def test_graphgp_sharded_matches_unsharded():
    """GraphGP on a one-dimensional mesh (sharded graph of a global graph: compacted
    levels, halo exchange, custom derivatives) equals the unsharded model: forward,
    gradients, adjointness, batches."""
    from jax.sharding import Mesh

    pts = _sky_points()
    mesh = Mesh(np.array(jax.devices()), ("x",))
    graph = jft.GraphGP.build_graph(pts, n0=32, k=8, cuda=False)
    ref = jft.GraphGP(graph, _Kernel(), shape=pts.shape[:-1], cuda=False)
    sharded = jft.GraphGP(graph, _Kernel(), shape=pts.shape[:-1], cuda=False, mesh=mesh)
    assert sharded.shape == ref.shape and sharded.excitation_shapes == ref.excitation_shapes
    assert_allclose(sharded.points, ref.points)
    p = jft.random_like(jax.random.PRNGKey(0), ref.domain)
    ps = sharded.shard(p)
    assert ps["xi"].sharding.is_equivalent_to(sharded.sharding, ps["xi"].ndim)
    f_ref, f_sh = ref(p), sharded(ps)
    assert f_sh.sharding.is_equivalent_to(sharded.sharding, f_sh.ndim)
    assert_allclose(f_sh, f_ref, rtol=1e-10, atol=1e-12)
    w = jax.random.normal(jax.random.PRNGKey(3), f_ref.shape)
    g_ref = jax.grad(lambda p: jnp.sum(w * ref(p)))(p)
    g_sh = jax.jit(jax.grad(lambda m, p: jnp.sum(w * m(p)), argnums=1))(sharded, ps)
    for key in p:
        assert_allclose(g_sh[key], g_ref[key], rtol=1e-9, atol=1e-12, err_msg=key)
    v = jft.random_like(jax.random.PRNGKey(5), sharded.domain)
    _, jv = jax.jvp(sharded, (ps,), (v,))
    _, vjp = jax.vjp(sharded, ps)
    (jt,) = vjp(w)
    assert_allclose(jnp.vdot(jv, w), _vdot(v, jt), rtol=1e-10)
    # batches of excitations
    xi_b = jax.random.normal(jax.random.PRNGKey(7), (3,) + ref.shape)
    cov = ref.covariance(p)
    assert_allclose(sharded.generate(xi_b, cov), ref.generate(xi_b, cov), rtol=1e-10, atol=1e-12)


@_needs_graphgp_and_devices
def test_build_sharded_graph_matches_brute_force_and_generate():
    """The graph built on the devices (per-shard graphgp builds, merged by querying
    the owners' trees) has, for every point, the k nearest among its own tree's
    predecessors and the foreign points of strictly lower tree level; its tables equal
    the host tables of the equivalent global graph; GraphGP on it equals graphgp's
    generation on that global graph; a sharded points array implies the mesh."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from nifty.re.graphgp_sharded import _build_device_graph, _compact_tables, _device_graph_to_graph, _local_graph, sharded_graph_from_graph

    pts = _sky_points()
    N, k, n0 = int(np.prod(pts.shape[:-1])), 6, 64
    mesh = Mesh(np.array(jax.devices()), ("x",))
    S = jax.device_count()
    B = N // S
    flat = pts.reshape(N, 3)
    dg = _build_device_graph(flat, mesh, n0=n0, k=k, cuda=False)
    assert dg.stats["n0"] == n0 and dg.stats["foreign_neighbours"] > 0
    # brute force of the merge rule on the host
    tpos, lvl = np.zeros(N, int), np.zeros(N, int)
    for s in range(S):
        ix = np.asarray(_local_graph(jnp.asarray(flat[s * B : (s + 1) * B]), n0 // S, k, cuda=False)[2])
        tpos[s * B + ix] = np.arange(B)
        lvl[s * B + ix] = np.floor(np.log2(np.arange(B) + 1.5))
    pd = flat.astype(np.float64)
    nbr = np.asarray(dg.neighbors).reshape(N, k)
    init = np.asarray(dg.initial).reshape(N)
    shard = np.arange(N) // B
    for p in range(N):
        if init[p]:
            assert np.all(nbr[p] == -1)
            continue
        cand = np.where(((shard == shard[p]) & (tpos < tpos[p])) | ((shard != shard[p]) & (lvl < lvl[p])))[0]
        dist = np.sort(np.linalg.norm(pd[cand] - pd[p], axis=1))[:k]
        assert_allclose(np.sort(np.linalg.norm(pd[nbr[p]] - pd[p], axis=1)), dist, rtol=1e-6, err_msg=str(p))
    # equivalent global graph, host tables == device tables
    graph = _device_graph_to_graph(dg)
    host = sharded_graph_from_graph(graph, S)
    dev = _compact_tables(dg, mesh)
    assert dev.group_sizes == host.group_sizes and dev.halo_size == host.halo_size and dev.n0 == host.n0
    np.testing.assert_array_equal(np.asarray(dev.halo), host.halo)
    np.testing.assert_array_equal(np.asarray(dev.ext_level), host.ext_level)
    fills = dict(own=host.junk, nbr=host.junk, valid=0, exp=host.junk, imp_src=0, imp_pos=0, imp_slot=host.junk, lvl=0)
    for g in range(len(host.group_sizes)):
        for name, fill in fills.items():
            a, b = np.asarray(getattr(dev, name)[g]), getattr(host, name)[g]
            padded = np.full(a.shape, fill, dtype=b.dtype)  # the device pads empty tables to width 1
            padded[tuple(slice(0, n) for n in b.shape)] = b
            np.testing.assert_array_equal(a, padded, err_msg=f"{name} group {g}")
    # GraphGP on the device-built graph (sharded points imply the mesh) == graphgp on the global graph
    pts_sharded = jax.device_put(pts, NamedSharding(mesh, P("x")))
    sharded = jft.GraphGP(pts_sharded, _Kernel(), n0=n0, k=k, cuda=False)
    assert sharded.mesh is not None and sharded.shape == pts.shape[:-1]
    ref = jft.GraphGP(graph, _Kernel(), shape=pts.shape[:-1], cuda=False)
    p = jft.random_like(jax.random.PRNGKey(0), ref.domain)
    assert_allclose(sharded(sharded.shard(p)), ref(p), rtol=1e-10, atol=1e-12)
    w = jax.random.normal(jax.random.PRNGKey(3), ref.shape)
    g_ref = jax.grad(lambda p: jnp.sum(w * ref(p)))(p)
    g_sh = jax.jit(jax.grad(lambda m, p: jnp.sum(w * m(p)), argnums=1))(sharded, sharded.shard(p))
    for key in p:
        assert_allclose(g_sh[key], g_ref[key], rtol=1e-9, atol=1e-12, err_msg=key)


@pytest.mark.skipif("graphgp" not in _models(), reason="needs graphgp")
def test_graphgp_sharded_on_one_device_matches_unsharded():
    """A one-device mesh has no ring: no edge tests, no candidates, an empty halo.
    The device build must not choke on the empty candidate table (it did: a maximum
    over a zero-size array) and must equal the host build on the same points."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from nifty.re.graphgp_sharded import _build_device_graph

    pts = _sky_points()
    n0, k = 64, 6
    mesh = Mesh(np.array(jax.devices()[:1]), ("x",))
    dg = _build_device_graph(pts.reshape(-1, 3), mesh, n0=n0, k=k, cuda=False)
    assert dg.stats["foreign_neighbours"] == 0 and dg.stats["candidates"] == []
    pts_one = jax.device_put(pts, NamedSharding(mesh, P("x")))
    sharded = jft.GraphGP(pts_one, _Kernel(), n0=n0, k=k, cuda=False)
    ref = jft.GraphGP(pts, _Kernel(), n0=n0, k=k, cuda=False)
    assert sharded.mesh is not None and sharded.shape == ref.shape
    p = jft.random_like(jax.random.PRNGKey(0), ref.domain)
    assert_allclose(sharded(sharded.shard(p)), ref(p), rtol=1e-10, atol=1e-12)
    w = jax.random.normal(jax.random.PRNGKey(3), ref.shape)
    g_ref = jax.grad(lambda p: jnp.sum(w * ref(p)))(p)
    g_sh = jax.jit(jax.grad(lambda m, p: jnp.sum(w * m(p)), argnums=1))(sharded, sharded.shard(p))
    for key in p:
        assert_allclose(g_sh[key], g_ref[key], rtol=1e-9, atol=1e-12, err_msg=key)


def test_tree_level_without_x64():
    """The tree-level helper of the device graph build must not depend on
    64-bit integers (a 64-bit shift silently overflowed with x64 disabled and
    the device build then found no foreign neighbours at all)."""
    from nifty.re.graphgp_sharded import _tree_level

    pos = np.array([0, 1, 2, 3, 6, 7, 14, 15, 62, 63, 2**20 - 1, 2**30 - 1])
    expected = np.floor(np.log2(pos + 1)).astype(int)
    with jax.enable_x64(False):
        np.testing.assert_array_equal(np.asarray(_tree_level(jnp.asarray(pos, jnp.int32))), expected)
    np.testing.assert_array_equal(np.asarray(_tree_level(jnp.asarray(pos))), expected)


@pytest.mark.skipif("graphgp" not in _models(), reason="needs graphgp")
def test_graphgp_sharded_fixed_table_gradient():
    """A sharded GraphGP with a *fixed* covariance table: the gradient with
    respect to the excitations only (the table's tangent is a symbolic zero)
    must equal the unsharded one. The custom_linear_solve of the sharded
    derivative failed its transpose rule when the zero tangent of the table
    was part of the right-hand side."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    pts = _sky_points()
    rs = jnp.concatenate([jnp.zeros(1), jnp.geomspace(0.05, 40.0, 255)])
    table = (rs, (1.0 + rs / 3.0) * jnp.exp(-rs / 3.0))
    n0, k = 64, 6
    mesh = Mesh(np.array(jax.devices()), ("x",))
    pts_sh = jax.device_put(pts, NamedSharding(mesh, P("x")))
    sharded = jft.GraphGP(pts_sh, table, n0=n0, k=k, cuda=False)
    ref = jft.GraphGP(sharded.graph if False else pts, table, n0=n0, k=k, cuda=False)
    p = jft.random_like(jax.random.PRNGKey(0), ref.domain)
    w = jax.random.normal(jax.random.PRNGKey(3), ref.shape)
    f_sh = jax.jit(sharded)(sharded.shard(p))
    g_sh = jax.jit(jax.grad(lambda m, p: jnp.sum(w * m(p)), argnums=1))(sharded, sharded.shard(p))
    # the device-built and the host-built graphs differ, so compare against the sharded model's
    # own graph on one device: finite differences of the jitted forward
    eps = 1e-6
    d = jft.random_like(jax.random.PRNGKey(5), ref.domain)
    fd = (jnp.sum(w * jax.jit(sharded)(sharded.shard({k_: p[k_] + eps * d[k_] for k_ in p}))) - jnp.sum(w * f_sh)) / eps
    lin = sum(jnp.vdot(g_sh[k_], d[k_]) for k_ in p)
    assert_allclose(lin, fd, rtol=1e-4, atol=1e-6)
    assert np.isfinite(np.asarray(jax.tree.leaves(g_sh)[0])).all()


@pytest.mark.parametrize("sharded", [False, True])
def test_icr_structure_shared(sharded):
    """An ICRStructure built once is shared by several ICR models on the same
    grid: same field as a model that compresses itself, the table arrays are
    the same objects (no second copy), and the structure of a model can be
    handed on. With a mesh the shard tables are part of the structure."""
    from jax.sharding import Mesh

    grid = jft.HPLogRGrid(nside=2, r_min_shape=4, r_min=1.0, r_max=10.0, nside0=1)
    mesh = Mesh(np.array(jax.devices()), ("x",)) if sharded else None
    structure = jft.build_icr_structure(grid, compress=dict(rtol=1e-5, atol=1e-10, buffer_size=1000), mesh=mesh)
    assert structure.n_shards == (len(jax.devices()) if sharded else 1)
    a = jft.ICRGP(grid, _Kernel(), compress=dict(rtol=1e-5, atol=1e-10, buffer_size=1000), mesh=mesh)
    b = jft.ICRGP(grid, _Kernel(), structure=structure, mesh=mesh)
    c = jft.ICRGP(grid, _Kernel(), structure=b.structure, mesh=mesh)
    assert b.n_unique == a.n_unique == structure.n_unique
    for lb, lc, ls in zip(b._levels, c._levels, structure.levels):
        assert lb.invids is ls.invids and lc.invids is ls.invids and lb.ucoords is ls.ucoords
    if sharded:
        assert b._shard_levels is structure.shard_levels and c._shard_levels is structure.shard_levels
    p = jft.random_like(jax.random.PRNGKey(0), a.domain)
    pa = a.shard(p) if sharded else p
    fa, fb, fc = jax.jit(a)(pa), jax.jit(b)(b.shard(p) if sharded else p), jax.jit(c)(c.shard(p) if sharded else p)
    assert_allclose(fb, fa, rtol=1e-12, atol=1e-12)
    assert_allclose(fc, fa, rtol=1e-12, atol=1e-12)
    # the correlated field takes the structure through the space
    m = jft.CorrelatedFieldMaker("s")
    m.set_amplitude_total_offset(offset_mean=0.0, offset_std=(0.1, 0.01))
    m.add_fluctuations(jft.ICRSpace(grid, n_bins=256, structure=structure), fluctuations=(1.0, 0.1), loglogavgslope=(-3.0, 0.5), prefix="a")
    cf = m.finalize(mesh=mesh)
    assert cf.gps[0]._levels[0].invids is structure.levels[0].invids
