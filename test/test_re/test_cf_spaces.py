#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

"""Correlated fields built from `Space` factors: regular, spherical, ICR and
graphgp spaces, their products and the covariance of the kernel-based ones."""

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from numpy.testing import assert_allclose

pmp = pytest.mark.parametrize
jax.config.update("jax_enable_x64", True)


def _maker(spaces, *, fluctuations=(2.0, 1e-6), slope=(-3.0, 1e-6), offset_std=(1.0, 1e-6)):
    m = jft.CorrelatedFieldMaker("cf")
    m.set_amplitude_total_offset(offset_mean=0.0, offset_std=offset_std)
    for i, space in enumerate(spaces):
        m.add_fluctuations(
            space, fluctuations=fluctuations, loglogavgslope=slope, flexibility=None, prefix=f"s{i}"
        )
    return m


def _open_grid_1d(n=40):
    return jft.SimpleOpenGrid(min_shape=(n,), window_size=3, splits=2, distances=1.0, desired_size0=8)


def _open_grid_2d(n=6):
    return jft.SimpleOpenGrid(min_shape=(n, n), window_size=3, splits=2, distances=1.0, desired_size0=16)


def _hp_grid():
    return jft.HPLogRGrid(nside=2, r_min_shape=4, r_min=1.0, r_max=10.0, nside0=1)


def _fluctuation_covariance(cf, p, space):
    """Covariance of the fluctuations of a single-factor field: the outer
    product of its Jacobian with respect to the main excitation block, with
    the zero-mode block at zero."""
    gp = cf.gps[0]

    def unflatten(vec):
        if not gp.structured:
            return vec.reshape(gp.excitation_shape)
        out, o = [], 0
        for sh in gp.excitation_shapes:
            n = int(np.prod(sh))
            out.append(vec[o : o + n].reshape(sh))
            o += n
        return out

    def fluct(vec):
        q = dict(p)
        q["cfxi"] = unflatten(vec)
        q["cfxi0"] = jnp.zeros(())
        return cf(q).ravel()

    A = jax.jacfwd(fluct)(jnp.zeros(space.n_points))
    return np.asarray(A @ A.T)


def _vdot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def _coordinates(space):
    if isinstance(space, jft.ICRSpace):
        g = space.multigrid.at(space.depth)
        idx = np.mgrid[tuple(slice(0, s) for s in g.shape)].reshape(g.ndim, -1)
        return np.asarray(g.index2coord(idx)).T
    return np.asarray(space.points).reshape(-1, space.dim)


@pmp(
    "legacy, spaces",
    [
        ([((32,), 0.1, "fourier")], [jft.RegularSpace((32,), 0.1)]),
        ([((8, 12), (0.5, 0.25), "fourier")], [jft.RegularSpace((8, 12), (0.5, 0.25))]),
        ([((4,), None, "spherical")], [jft.SphericalSpace(4)]),
        (
            [((6,), 1.0, "fourier"), ((2,), None, "spherical")],
            [jft.RegularSpace((6,), 1.0), jft.SphericalSpace(2)],
        ),
        (
            [((5, 4), 0.2, "fourier"), ((7,), 0.3, "fourier")],
            [jft.RegularSpace((5, 4), 0.2), jft.RegularSpace((7,), 0.3)],
        ),
    ],
)
def test_spaces_match_legacy_arguments(legacy, spaces):
    if any(ht == "spherical" for _, _, ht in legacy):
        pytest.importorskip("jaxbind")
    kw = dict(fluctuations=(1.0, 0.3), loglogavgslope=(-3.0, 0.5), flexibility=(0.8, 0.2), asperity=(0.3, 0.1))
    m_legacy = jft.CorrelatedFieldMaker("cf")
    m_legacy.set_amplitude_total_offset(offset_mean=0.3, offset_std=(0.5, 0.1))
    m_spaces = jft.CorrelatedFieldMaker("cf")
    m_spaces.set_amplitude_total_offset(offset_mean=0.3, offset_std=(0.5, 0.1))
    for i, ((shape, distances, ht), space) in enumerate(zip(legacy, spaces)):
        m_legacy.add_fluctuations(shape, distances, harmonic_type=ht, prefix=f"a{i}", **kw)
        m_spaces.add_fluctuations(space, prefix=f"a{i}", **kw)
    cf_legacy, cf_spaces = m_legacy.finalize(), m_spaces.finalize()
    assert cf_legacy.domain.keys() == cf_spaces.domain.keys()
    for k in cf_legacy.domain:
        assert cf_legacy.domain[k].shape == cf_spaces.domain[k].shape
    p = jft.random_like(jax.random.PRNGKey(3), cf_legacy.domain)
    assert_allclose(cf_legacy(p), cf_spaces(p), rtol=0.0, atol=0.0)
    assert cf_spaces(p).shape == sum((s.shape for s in spaces), ())


def test_space_rejects_distances():
    m = jft.CorrelatedFieldMaker("cf")
    with pytest.raises(TypeError):
        m.add_fluctuations(jft.RegularSpace((8,), 1.0), 1.0, (1.0, 0.1), (-3.0, 0.5))


@pmp("dim", [1, 2, 3])
def test_hankel_weights_gaussian(dim):
    """The spectrum-to-covariance transform against the Gaussian pair
    ``P(k) = exp(-(k / kc)^2)`` <-> ``C(r) = (sqrt(pi) kc)^dim exp(-(pi kc r)^2)``."""
    kc = 0.7
    k_bins = np.concatenate(([0.0], np.geomspace(1e-3, 8.0, 4000)))
    radial_bins = np.concatenate(([0.0], np.geomspace(1e-2, 3.0, 50)))
    grid = jft.make_isotropic_grid(dim, radial_bins=radial_bins, k_bins=k_bins)
    space = jft.KernelSpace((1,), dim, radial_bins=radial_bins, k_bins=k_bins, jitter=0.0)
    amplitude = jnp.exp(-0.5 * (k_bins / kc) ** 2)
    cov = np.asarray(space.covariance(amplitude))
    truth = (np.sqrt(np.pi) * kc) ** dim * np.exp(-((np.pi * kc * radial_bins) ** 2))
    assert_allclose(cov, truth, rtol=5e-3, atol=5e-3 * truth[0])
    # C(0) is the multiplicity-weighted sum of the amplitude squared
    mult = grid.harmonic_grid.mode_multiplicity
    assert_allclose(cov[0], np.sum(mult[1:] * np.asarray(amplitude[1:]) ** 2), rtol=1e-12)


@pmp(
    "make_space",
    [
        lambda: jft.ICRSpace(_open_grid_1d(), n_bins=128, chunk_size=7),
        lambda: jft.ICRSpace(_open_grid_2d(), n_bins=256, chunk_size=5),  # 128 bins sit at the positive-definiteness limit here
        lambda: jft.ICRSpace(_open_grid_1d(), n_bins=128, compress=None),
        lambda: jft.GraphGPSpace(
            np.random.default_rng(0).uniform(0.0, 10.0, (200, 2)), n0=32, k=12, n_bins=128, cuda=False
        ),
    ],
    ids=["icr1d", "icr2d", "icr1d-uncompressed", "graphgp2d"],
)
def test_kernel_space_covariance(make_space):
    if "GraphGP" in make_space.__code__.co_names:
        pytest.importorskip("graphgp")
    space = make_space()
    m = _maker([space])
    cf = m.finalize()
    xi_dom = cf.domain["cfxi"]
    shapes = tuple(x.shape for x in xi_dom) if isinstance(xi_dom, list) else (xi_dom.shape,)
    assert shapes == space.excitation_shapes and cf.domain["cfxi0"].shape == ()
    p = jft.random_like(jax.random.PRNGKey(1), cf.domain)
    f = jax.jit(lambda cf, p: cf(p))(cf, p)
    assert f.shape == space.shape and np.all(np.isfinite(f))
    # covariance of the generated fluctuations versus the covariance function
    C = _fluctuation_covariance(cf, p, space)
    cov = np.asarray(space.covariance(m._fluctuations[0](p)))
    fluct = float(m._fluctuations[0].fluctuations(p))
    assert_allclose(cov[0], fluct**2 * (1.0 + space.jitter), rtol=1e-12)
    coords = _coordinates(space)
    r = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    truth = np.interp(r, space.radial_bins, cov)
    assert np.linalg.norm(C - truth) / np.linalg.norm(truth) < 3e-2
    # the zero-mode block is the constant one, scaled by the zero-mode amplitude
    q = dict(p)
    q["cfxi"] = jax.tree_util.tree_map(jnp.zeros_like, p["cfxi"])
    q["cfxi0"] = jnp.asarray(1.5)
    assert_allclose(cf(q), 1.5 * m.azm(p), rtol=1e-12)
    # gradients and vmap over samples
    g = jax.jit(jax.grad(lambda p: (cf(p) ** 2).sum()))(p)
    assert all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(g))
    ps = jax.vmap(lambda k: jft.random_like(k, cf.domain))(jax.random.split(jax.random.PRNGKey(2), 3))
    fs = jax.jit(jax.vmap(cf))(ps)
    for i in range(3):
        assert_allclose(fs[i], cf(jax.tree_util.tree_map(lambda x: x[i], ps)), rtol=1e-10, atol=1e-10)


def test_icr_space_healpix_log_r():
    space = jft.ICRSpace(_hp_grid(), n_bins=256, chunk_size=100)
    assert space.dim == 3
    assert space.shape == tuple(int(s) for s in space.multigrid.at(space.depth).shape)
    assert space.excitation_shapes == tuple(tuple(int(s) for s in space.multigrid.at(l).shape) for l in range(space.depth + 1))
    m = _maker([space])
    cf = m.finalize()
    p = jft.random_like(jax.random.PRNGKey(0), cf.domain)
    f = jax.jit(lambda cf, p: cf(p))(cf, p)
    assert f.shape == space.shape and np.all(np.isfinite(f))
    # the chunked assembly equals the layout of the grid's `resort`
    C = _fluctuation_covariance(cf, p, space)
    cov = np.asarray(space.covariance(m._fluctuations[0](p)))
    coords = _coordinates(space)
    r = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    truth = np.interp(r, space.radial_bins, cov)
    assert np.linalg.norm(C - truth) / np.linalg.norm(truth) < 5e-2


def test_product_regular_icr():
    reg = jft.RegularSpace((16,), 0.5)
    icr = jft.ICRSpace(_open_grid_1d(), n_bins=128)
    m = _maker([reg, icr], offset_std=(0.7, 1e-6))
    cf = m.finalize()
    assert cf.spaces == (reg, icr)
    # main block: the regular modes times every ICR level; zero block of the ICR factor: the regular modes
    assert [x.shape for x in cf.domain["cfxi"]] == [(16,) + sh for sh in icr.excitation_shapes]
    assert cf.domain["cfxi0"].shape == (16,)
    p = jft.random_like(jax.random.PRNGKey(5), cf.domain)
    f = jax.jit(lambda cf, p: cf(p))(cf, p)
    assert f.shape == reg.shape + icr.shape
    azm = float(m.azm(p))
    zero_main = jax.tree_util.tree_map(jnp.zeros_like, p["cfxi"])
    # both zero modes: the constant field (regular zero mode = index 0 of the zero block)
    q = dict(p)
    q["cfxi"] = zero_main
    q["cfxi0"] = jnp.zeros(16).at[0].set(2.0)
    assert_allclose(cf(q), 2.0 * azm, rtol=1e-12)
    # ICR zero block with regular modes: constant along the ICR axis
    q["cfxi0"] = p["cfxi0"]
    g = cf(q)
    assert_allclose(g, np.broadcast_to(g[:, :1], g.shape), rtol=1e-12, atol=1e-12)
    assert np.std(g[:, 0]) > 0
    # regular zero mode (index 0 of the main block's regular axis) with ICR excitations: constant along the regular axis
    q["cfxi0"] = jnp.zeros(16)
    q["cfxi"] = [jnp.zeros_like(x).at[0].set(x[0]) for x in p["cfxi"]]
    g = cf(q)
    assert_allclose(g, np.broadcast_to(g[:1], g.shape), rtol=1e-12, atol=1e-12)
    assert np.std(g[0]) > 0
    # the map is linear in the excitation blocks
    q2 = dict(p)
    q2["cfxi"] = jax.tree_util.tree_map(lambda x: 2.0 * x, p["cfxi"])
    q2["cfxi0"] = 2.0 * p["cfxi0"]
    assert_allclose(cf(q2), 2.0 * f, rtol=1e-12)
    jax.jit(jax.grad(lambda p: (cf(p) ** 2).sum()))(p)


def test_product_icr_graphgp_spherical():
    pytest.importorskip("graphgp")
    pytest.importorskip("jaxbind")
    pts = np.random.default_rng(1).uniform(0.0, 5.0, (50, 3))
    spaces = [jft.SphericalSpace(1), jft.GraphGPSpace(pts, n0=16, k=8, n_bins=256, cuda=False), jft.ICRSpace(_open_grid_1d(12), n_bins=256)]
    cf = _maker(spaces).finalize()
    names = [b[0] for b in cf.blocks]
    assert names == ["cfxi", "cfxi0_1", "cfxi0_2", "cfxi0_1_2"]
    assert cf.domain["cfxi0_1_2"].shape == spaces[0].excitation_shape  # spherical harmonic modes only
    p = jft.random_like(jax.random.PRNGKey(0), cf.domain)
    f = jax.jit(lambda cf, p: cf(p))(cf, p)
    assert f.shape == sum((s.shape for s in spaces), ())
    assert np.all(np.isfinite(f))


def test_matern_amplitude_on_kernel_space():
    space = jft.ICRSpace(_open_grid_1d(), n_bins=128)
    m = jft.CorrelatedFieldMaker("cf")
    m.set_amplitude_total_offset(offset_mean=1.0, offset_std=(1.0, 1e-6))
    m.add_fluctuations_matern(space, scale=(1.5, 1e-6), cutoff=(0.2, 1e-6), loglogslope=(-4.0, 1e-6), renormalize_amplitude=True)
    cf = m.finalize()
    p = jft.random_like(jax.random.PRNGKey(0), cf.domain)
    cov = np.asarray(space.covariance(m._fluctuations[0](p)))
    assert_allclose(cov[0], 1.5**2 * (1.0 + space.jitter), rtol=1e-6)
    assert np.all(np.isfinite(cf(p)))


def _spaces_for_adjoint():
    out = {
        "icr1d": lambda: [jft.ICRSpace(_open_grid_1d(), n_bins=128)],
        "icr-healpix": lambda: [jft.ICRSpace(_hp_grid(), n_bins=256)],
        "regular-x-icr": lambda: [jft.RegularSpace((6,), 0.5), jft.ICRSpace(_open_grid_1d(12), n_bins=128)],
    }
    try:
        import graphgp  # noqa: F401

        out["graphgp2d"] = lambda: [
            jft.GraphGPSpace(np.random.default_rng(0).uniform(0.0, 10.0, (120, 2)), n0=16, k=8, n_bins=128, cuda=False)
        ]
        out["graphgp-x-icr"] = lambda: [
            jft.GraphGPSpace(np.random.default_rng(1).uniform(0.0, 5.0, (20, 3)), n0=8, k=6, n_bins=64, cuda=False),
            jft.ICRSpace(_open_grid_1d(12), n_bins=128),
        ]
    except ImportError:
        pass
    return out


@pmp("name", list(_spaces_for_adjoint()))
def test_jvp_vjp_adjoint(name):
    """<J v, w> == <v, J^T w> for the full model (jvp against vjp) and the
    linear transpose of the excitation map against its vjp."""
    spaces = _spaces_for_adjoint()[name]()
    m = _maker(spaces, fluctuations=(1.0, 0.3), slope=(-3.0, 0.5), offset_std=(0.8, 0.2))
    cf = m.finalize()
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(7), 3)
    p = jft.random_like(k0, cf.domain)
    v = jft.random_like(k1, cf.domain)
    w = jax.random.normal(k2, cf.target.shape, dtype=cf.target.dtype)
    f, jv = jax.jvp(cf, (p,), (v,))
    _, vjp = jax.vjp(cf, p)
    (jtw,) = vjp(w)
    lhs = jnp.vdot(jv, w)
    rhs = _vdot(v, jtw)
    assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10 * abs(float(lhs)))
    # the map from the excitation blocks alone is linear
    names = [b[0] for b in cf.blocks]
    xi = {k: p[k] for k in names}
    v_xi = {k: v[k] for k in names}

    def lin(x):
        return cf({**p, **x})

    zero = jax.tree_util.tree_map(jnp.zeros_like, xi)
    assert_allclose(lin(xi) + lin(v_xi) - lin(zero), lin(jax.tree_util.tree_map(jnp.add, xi, v_xi)), rtol=1e-10, atol=1e-10)
    transpose = jax.linear_transpose(lambda t: jax.jvp(lin, (xi,), (t,))[1], xi)
    (jt_lin,) = transpose(w)
    for k in names:
        for a, b in zip(jax.tree_util.tree_leaves(jt_lin[k]), jax.tree_util.tree_leaves(jtw[k])):
            assert_allclose(a, b, rtol=1e-10, atol=1e-10)


@pmp("name", ["icr-healpix", "graphgp2d", "regular-x-icr"])
def test_models_are_pytrees_without_inlined_constants(name):
    """The arrays of the spaces are dynamic leaves of the model, so passing
    the model into a jitted function inlines no large constants and retraces
    only once."""
    spaces = _spaces_for_adjoint()
    if name not in spaces:
        pytest.importorskip("graphgp")
    cf = _maker(spaces[name]()).finalize()
    p = jft.random_like(jax.random.PRNGKey(0), cf.domain)
    leaves = jax.tree_util.tree_leaves(cf)
    n_dynamic = sum(int(np.asarray(l).nbytes) for l in leaves)
    consts = jax.make_jaxpr(lambda cf, p: cf(p))(cf, p).consts
    n_const = sum(int(np.asarray(c).nbytes) for c in consts)
    assert n_const < 2**16, f"{n_const} bytes of constants inlined"
    if any(isinstance(s, jft.KernelSpace) for s in cf.spaces):
        assert n_dynamic > n_const
    traces = []

    @jax.jit
    def f(cf, p):
        traces.append(1)
        return cf(p)

    assert_allclose(f(cf, p), cf(p), rtol=1e-12, atol=1e-12)
    f(cf, p)
    assert len(traces) == 1


_needs_devices = pytest.mark.skipif(jax.device_count() < 2, reason="needs several devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")


def test_grid_points():
    """Coordinates of a multigrid level in pixel order, shape + (dim,)."""
    grid = _hp_grid()
    pts = jft.grid_points(grid)
    shape = tuple(grid.at(-1).shape)
    assert pts.shape == shape + (3,)
    idx = np.stack(np.meshgrid(*(np.arange(s) for s in shape), indexing="ij"), axis=0)
    ref = np.moveaxis(np.asarray(grid.at(-1).index2coord(idx.reshape(2, -1))), 0, -1).reshape(shape + (3,))
    assert_allclose(pts, ref)
    assert_allclose(jft.grid_points(grid, 0), np.moveaxis(np.asarray(grid.at(0).index2coord(np.stack(np.meshgrid(*(np.arange(s) for s in grid.at(0).shape), indexing="ij"), 0).reshape(2, -1))), 0, -1).reshape(tuple(grid.at(0).shape) + (3,)))


@_needs_devices
@pytest.mark.parametrize("kind", ["icr", "graphgp"])
def test_correlated_field_sharded_matches_unsharded(kind):
    """`finalize(mesh=)`: a correlated field with a kernel factor on the mesh
    (ICR or graphgp, the latter with its graph built on the devices from sharded
    points, whose mesh is inferred) times a harmonic factor on the second axis
    equals the unsharded field, values and gradients; excitations drawn by
    `init` and placed by `shard` carry the field's shardings."""
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(jax.devices()), ("x",))
    grid = jft.HPLogRGrid(nside=4, r_min_shape=4, r_min=1.0, r_max=10.0, nside0=1)
    reg = jft.RegularSpace((6,), 0.5)
    if kind == "icr":
        kernel_space = jft.ICRSpace(grid, n_bins=256)
        kernel_space_sharded = kernel_space
        explicit = mesh
    else:
        pytest.importorskip("graphgp")
        pts = jft.grid_points(grid)
        kernel_space = jft.GraphGPSpace(pts, n0=32, k=6, n_bins=256, cuda=False)
        kernel_space_sharded = jft.GraphGPSpace(jft.grid_points(grid, mesh=mesh), n0=32, k=6, n_bins=256, cuda=False)
        assert kernel_space_sharded.mesh is not None
        explicit = None  # inferred from the sharded points
    cf = _maker([kernel_space, reg]).finalize()
    cfs = _maker([kernel_space_sharded, reg]).finalize(mesh=explicit)
    assert cfs.mesh is not None and cfs.domain.keys() == cf.domain.keys()
    assert set(cfs.shardings) == {"cfxi", "cfxi0"}
    p = jft.random_like(jax.random.PRNGKey(0), cf.domain)
    ps = cfs.shard(p)
    for name, s in cfs.shardings.items():
        if isinstance(s, list):
            assert all(a.sharding.is_equivalent_to(si, a.ndim) for a, si in zip(ps[name], s))
        else:
            assert ps[name].sharding.is_equivalent_to(s, ps[name].ndim)
    f, fs = jax.jit(lambda cf, p: cf(p))(cf, p), jax.jit(lambda cf, p: cf(p))(cfs, ps)
    assert fs.shape == f.shape == kernel_space.shape + reg.shape
    assert fs.sharding.is_equivalent_to(cfs.sharding, fs.ndim)
    if kind == "graphgp":  # different graphs (global vs device-built): same statistics, not the same field
        return_early = True
    else:
        return_early = False
    if not return_early:
        assert_allclose(fs, f, rtol=1e-9, atol=1e-10)
        w = jax.random.normal(jax.random.PRNGKey(3), f.shape)
        g = jax.jit(jax.grad(lambda cf, p: jnp.sum(w * cf(p)), argnums=1))(cf, p)
        gs = jax.jit(jax.grad(lambda cf, p: jnp.sum(w * cf(p)), argnums=1))(cfs, ps)
        for key in g:
            for a, b in zip(jax.tree_util.tree_leaves(gs[key]), jax.tree_util.tree_leaves(g[key])):
                assert_allclose(a, b, rtol=1e-8, atol=1e-10, err_msg=key)
    # the amplitude parameters placed on the mesh (replicated), as an optimizer running on the
    # mesh holds them: the same field (the covariance table must not reach the manual region
    # of the generator through a closure)
    rep = NamedSharding(mesh, P())
    pr = {k: (v if k in cfs.shardings else jax.device_put(np.asarray(v), rep)) for k, v in ps.items()}
    fr = jax.jit(lambda cf, p: cf(p))(cfs, pr)
    assert_allclose(fr, fs, rtol=1e-12, atol=1e-12)
    gr = jax.jit(jax.grad(lambda cf, p: jnp.sum(cf(p) ** 2), argnums=1))(cfs, pr)
    assert all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(gr))
    # differentiation of a jitted inner function inside an outer jit (NIFTy's optimizers jit
    # the likelihood pieces and call them from jitted KL steps): the custom derivatives of
    # the sharded generators must not close over the model's arrays
    inner = jax.jit(lambda cf, p: jnp.sum(cf(p) ** 2))
    vg = jax.jit(lambda cf, p: jax.value_and_grad(lambda p: inner(cf, p))(p))(cfs, pr)
    assert np.isfinite(vg[0]) and all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(vg[1]))
    jax.block_until_ready(jax.jit(lambda cf, p: jax.jvp(lambda p: inner(cf, p), (p,), (p,)))(cfs, pr))
    # init draws straight onto the mesh
    q = cfs.init(jax.random.PRNGKey(7))
    for name, s in cfs.shardings.items():
        if isinstance(s, list):
            assert all(a.sharding.is_equivalent_to(si, a.ndim) for a, si in zip(q[name], s))
        else:
            assert q[name].sharding.is_equivalent_to(s, q[name].ndim)
    assert np.all(np.isfinite(jax.jit(lambda cf, p: cf(p))(cfs, q)))


@_needs_devices
def test_correlated_field_sharded_graphgp_same_graph():
    """With the same (global) graph the sharded graphgp field equals the unsharded one."""
    pytest.importorskip("graphgp")
    from jax.sharding import Mesh

    mesh = Mesh(np.array(jax.devices()), ("x",))
    grid = jft.HPLogRGrid(nside=4, r_min_shape=4, r_min=1.0, r_max=10.0, nside0=1)
    pts = jft.grid_points(grid)
    graph = jft.GraphGP.build_graph(pts, n0=32, k=6, cuda=False)
    reg = jft.RegularSpace((6,), 0.5)
    space = jft.GraphGPSpace(pts, graph=graph, n_bins=256, cuda=False)
    cf = _maker([space, reg]).finalize()
    cfs = _maker([space, reg]).finalize(mesh=mesh)
    p = jft.random_like(jax.random.PRNGKey(0), cf.domain)
    ps = cfs.shard(p)
    assert_allclose(jax.jit(lambda cf, p: cf(p))(cfs, ps), jax.jit(lambda cf, p: cf(p))(cf, p), rtol=1e-9, atol=1e-10)
    w = jax.random.normal(jax.random.PRNGKey(3), cf.target.shape if hasattr(cf.target, "shape") else sum((s.shape for s in (space, reg)), ()))
    g = jax.jit(jax.grad(lambda cf, p: jnp.sum(w * cf(p)), argnums=1))(cf, p)
    gs = jax.jit(jax.grad(lambda cf, p: jnp.sum(w * cf(p)), argnums=1))(cfs, ps)
    for key in g:
        assert_allclose(gs[key], g[key], rtol=1e-8, atol=1e-10, err_msg=key)
