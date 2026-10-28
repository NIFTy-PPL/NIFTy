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
    product of its Jacobian with respect to the excitations without the zero
    mode."""
    xi = p["cfxi"]

    def fluct(rest):
        q = dict(p)
        q["cfxi"] = xi.at[0].set(0.0).at[1:].set(rest)
        return cf(q).ravel()

    A = jax.jacfwd(fluct)(jnp.zeros(space.n_points))
    return np.asarray(A @ A.T)


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
        lambda: jft.ICRSpace(_open_grid_2d(), n_bins=256, chunk_size=5),
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
    assert cf.domain["cfxi"].shape == space.excitation_shape
    p = jft.random_like(jax.random.PRNGKey(1), cf.domain)
    f = jax.jit(cf)(p)
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
    # the zero mode is the constant one, scaled by the zero-mode amplitude
    q = dict(p)
    q["cfxi"] = jnp.zeros_like(p["cfxi"]).at[0].set(1.5)
    assert_allclose(cf(q), 1.5 * m.azm(p), rtol=1e-12)
    # gradients and vmap over samples
    g = jax.jit(jax.grad(lambda p: (cf(p) ** 2).sum()))(p)
    assert all(np.all(np.isfinite(v)) for v in g.values())
    ps = jax.vmap(lambda k: jft.random_like(k, cf.domain))(jax.random.split(jax.random.PRNGKey(2), 3))
    fs = jax.jit(jax.vmap(cf))(ps)
    for i in range(3):
        assert_allclose(fs[i], cf(jax.tree_util.tree_map(lambda x: x[i], ps)), rtol=1e-10, atol=1e-10)


def test_icr_space_healpix_log_r():
    space = jft.ICRSpace(_hp_grid(), n_bins=256, chunk_size=100)
    assert space.dim == 3
    assert space.shape == tuple(int(s) for s in space.multigrid.at(space.depth).shape)
    assert space.excitation_shape == (sum(int(space.multigrid.at(l).size) for l in range(space.depth + 1)) + 1,)
    cf = _maker([space]).finalize()
    p = jft.random_like(jax.random.PRNGKey(0), cf.domain)
    f = jax.jit(cf)(p)
    assert f.shape == space.shape and np.all(np.isfinite(f))
    # the chunked assembly equals the layout of the grid's `resort`
    C = _fluctuation_covariance(cf, p, space)
    cov = np.asarray(space.covariance(cf.normalized_amplitudes and _maker([space])._fluctuations[0](p)))
    coords = _coordinates(space)
    r = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    truth = np.interp(r, space.radial_bins, cov)
    assert np.linalg.norm(C - truth) / np.linalg.norm(truth) < 5e-2


def test_product_regular_icr():
    reg = jft.RegularSpace((16,), 0.5)
    icr = jft.ICRSpace(_open_grid_1d(), n_bins=128)
    m = _maker([reg, icr], offset_std=(0.7, 1e-6))
    cf = m.finalize()
    assert cf.domain["cfxi"].shape == reg.excitation_shape + icr.excitation_shape
    assert cf.spaces == (reg, icr)
    p = jft.random_like(jax.random.PRNGKey(5), cf.domain)
    f = jax.jit(cf)(p)
    assert f.shape == reg.shape + icr.shape
    azm = float(m.azm(p))
    xi = p["cfxi"]
    # zero mode of both factors: the constant field
    q = dict(p)
    q["cfxi"] = jnp.zeros_like(xi).at[0, 0].set(2.0)
    assert_allclose(cf(q), 2.0 * azm, rtol=1e-12)
    # zero mode of the regular factor only: constant along the regular axis
    q["cfxi"] = jnp.zeros_like(xi).at[0, 1:].set(xi[0, 1:])
    g = cf(q)
    assert_allclose(g, np.broadcast_to(g[:1], g.shape), rtol=1e-12, atol=1e-12)
    assert np.std(g[0]) > 0
    # zero mode of the ICR factor only: constant along the ICR axis
    q["cfxi"] = jnp.zeros_like(xi).at[1:, 0].set(xi[1:, 0])
    g = cf(q)
    assert_allclose(g, np.broadcast_to(g[:, :1], g.shape), rtol=1e-12, atol=1e-12)
    assert np.std(g[:, 0]) > 0
    # the map is linear in the excitations
    q2 = dict(p)
    q2["cfxi"] = 2.0 * xi
    assert_allclose(cf(q2), 2.0 * f, rtol=1e-12)
    jax.jit(jax.grad(lambda p: (cf(p) ** 2).sum()))(p)


def test_product_icr_graphgp_spherical():
    pytest.importorskip("graphgp")
    pytest.importorskip("jaxbind")
    pts = np.random.default_rng(1).uniform(0.0, 5.0, (50, 3))
    spaces = [jft.SphericalSpace(1), jft.GraphGPSpace(pts, n0=16, k=8, n_bins=64, cuda=False), jft.ICRSpace(_open_grid_1d(12), n_bins=256)]
    cf = _maker(spaces).finalize()
    p = jft.random_like(jax.random.PRNGKey(0), cf.domain)
    f = jax.jit(cf)(p)
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
    rhs = sum(jnp.vdot(v[k], jtw[k]) for k in v)
    assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10 * abs(float(lhs)))
    # the map from the excitations alone is linear
    xi_name = m._prefix + "xi"

    def lin(xi):
        return cf({**p, xi_name: xi})

    xi = p[xi_name]
    assert_allclose(lin(xi) + lin(v[xi_name]) - lin(jnp.zeros_like(xi)), lin(xi + v[xi_name]), rtol=1e-10, atol=1e-10)
    transpose = jax.linear_transpose(lambda t: jax.jvp(lin, (xi,), (t,))[1], xi)
    (jt_lin,) = transpose(w)
    assert_allclose(jt_lin, jtw[xi_name], rtol=1e-10, atol=1e-10)


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
