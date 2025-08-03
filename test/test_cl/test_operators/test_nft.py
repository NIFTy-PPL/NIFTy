# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2019-2021 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Copyright(C) 2025 LambdaFields GmbH
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty.cl as ift
import numpy as np
import pytest

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


@pmp('eps', [1e-2, 1e-4, 1e-7, 1e-10, 1e-11, 1e-12, 2e-13])
@pmp('nxdirty', [32, 128])
@pmp('nydirty', [32, 48, 128])
@pmp('N', [1, 10, 100])
def test_gridding(nxdirty, nydirty, N, eps):
    uv = ift.random.current_rng().random((N, 2)) - 0.5
    vis = (ift.random.current_rng().standard_normal(N)
           + 1j*ift.random.current_rng().standard_normal(N))

    if N > 2:
        uv[-1] = 0
        uv[-2] = 1e-5
    # Nifty
    dom = ift.RGSpace((nxdirty, nydirty), distances=(0.2, 1.12))
    dstx, dsty = dom.distances
    uv[:, 0] = uv[:, 0]/dstx
    uv[:, 1] = uv[:, 1]/dsty
    Op = ift.Gridder(dom, uv=uv, eps=eps)
    vis2 = ift.makeField(ift.UnstructuredDomain(vis.shape), vis)

    pynu = Op(vis2).asnumpy()
    # DFT
    x, y = np.meshgrid(
        *[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty]], indexing='ij')
    dft = pynu*0.
    for i in range(N):
        dft += (
            vis[i]*np.exp(2j*np.pi*(x*uv[i, 0]*dstx + y*uv[i, 1]*dsty))).real
    ift.myassert(_l2error(dft, pynu) < eps)


def test_cartesian():
    nx, ny = 32, 42
    dstx, dsty = 0.3, 0.2
    dom = ift.RGSpace((nx, ny), (dstx, dsty))

    kx = np.fft.fftfreq(nx, dstx)
    ky = np.fft.fftfreq(ny, dsty)
    uu, vv = np.meshgrid(kx, ky)
    tmp = np.vstack([uu[None, :], vv[None, :]])
    uv = np.transpose(tmp, (2, 1, 0)).reshape(-1, 2)

    op = ift.Gridder(dom, uv=uv).adjoint

    fld = ift.from_random(dom, 'normal')
    arr = fld.asnumpy()

    fld2 = ift.makeField(dom, np.roll(arr, (nx//2, ny//2), axis=(0, 1)))
    res = op(fld2).asnumpy().reshape(nx, ny)

    fft = ift.FFTOperator(dom.get_default_codomain(), target=dom).adjoint
    vol = ift.full(dom, 1.).s_integrate()
    res1 = fft(fld).asnumpy()

    np.testing.assert_allclose(res, res1*vol)


@pmp('eps', [1e-2, 1e-6, 2e-13])
@pmp('nxdirty', [32, 128])
@pmp('nydirty', [32, 48, 128])
@pmp('N', [1, 10, 100])
def test_build(nxdirty, nydirty, N, eps):
    dom = ift.RGSpace([nxdirty, nydirty])
    uv = ift.random.current_rng().random((N, 2)) - 0.5
    RF = ift.Gridder(dom, uv=uv, eps=eps)

    # Consistency checks
    flt = np.float64
    cmplx = np.complex128
    # We set rtol=eps here, because the gridder operator only guarantees
    # adjointness to this accuracy.
    ift.extra.check_linear_operator(RF, cmplx, flt, only_r_linear=True,
                                    rtol=eps, atol=eps, no_device_copies=False)


@pmp('eps', [1e-2, 1e-4, 1e-7, 1e-10, 1e-11, 1e-12, 2e-13])
@pmp('nxdirty', [32, 128])
@pmp('N', [1, 10, 100])
def test_nu1d(nxdirty, N, eps):
    pos = ift.random.current_rng().random((N)) - 0.5
    vis = (ift.random.current_rng().standard_normal(N)
           + 1j*ift.random.current_rng().standard_normal(N))

    if N > 2:
        pos[-1] = 0
        pos[-2] = 1e-5
    # Nifty
    dom = ift.RGSpace((nxdirty), distances=0.2)
    dstx = dom.distances
    pos = pos / dstx
    Op = ift.Nufft(dom, pos=pos[:, None], eps=eps)
    vis2 = ift.makeField(ift.UnstructuredDomain(vis.shape), vis)
    pynu = Op(vis2).asnumpy()
    # DFT
    x = -nxdirty/2 + np.arange(nxdirty)

    dft = pynu*0
    for i in range(N):
        dft += (vis[i]*np.exp(2j*np.pi*(x*pos[i]*dstx))).real
    ift.myassert(_l2error(dft, pynu) < eps*10)


@pmp('eps', [1e-2, 1e-4, 1e-7, 1e-10, 1e-11, 1e-12, 2e-13])
@pmp('nxdirty', [32, 128])
@pmp('nydirty', [32, 48, 128])
@pmp('N', [1, 10, 100])
def test_nu2d(nxdirty, nydirty, N, eps):
    uv = ift.random.current_rng().random((N, 2)) - 0.5
    vis = (ift.random.current_rng().standard_normal(N)
           + 1j*ift.random.current_rng().standard_normal(N))

    if N > 2:
        uv[-1] = 0
        uv[-2] = 1e-5
    # Nifty
    dom = ift.RGSpace((nxdirty, nydirty), distances=(0.2, 1.12))
    dstx, dsty = dom.distances
    uv[:, 0] = uv[:, 0]/dstx
    uv[:, 1] = uv[:, 1]/dsty
    Op = ift.Nufft(dom, pos=uv, eps=eps)
    vis2 = ift.makeField(ift.UnstructuredDomain(vis.shape), vis)

    pynu = Op(vis2).asnumpy()
    # DFT
    x, y = np.meshgrid(
        *[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty]], indexing='ij')
    dft = pynu*0.
    for i in range(N):
        dft += (
            vis[i]*np.exp(2j*np.pi*(x*uv[i, 0]*dstx + y*uv[i, 1]*dsty))).real
    ift.myassert(_l2error(dft, pynu) < eps*10)


@pmp('eps', [1e-2, 1e-4, 1e-7, 1e-11])
@pmp('nxdirty', [32, 128])
@pmp('nydirty', [32, 48])
@pmp('nzdirty', [32, 54])
@pmp('N', [1, 10])
def test_nu3d(nxdirty, nydirty, nzdirty, N, eps):
    pos = ift.random.current_rng().random((N, 3)) - 0.5
    vis = (ift.random.current_rng().standard_normal(N)
           + 1j*ift.random.current_rng().standard_normal(N))
    # Nifty
    dom = ift.RGSpace((nxdirty, nydirty, nzdirty), distances=(0.2, 1.12, 0.7))
    dstx, dsty, dstz = dom.distances
    pos[:, 0] = pos[:, 0]/dstx
    pos[:, 1] = pos[:, 1]/dsty
    pos[:, 2] = pos[:, 2]/dstz
    Op = ift.Nufft(dom, pos=pos, eps=eps)
    vis2 = ift.makeField(ift.UnstructuredDomain(vis.shape), vis)

    pynu = Op(vis2).asnumpy()
    # DFT
    x, y, z = np.meshgrid(
        *[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty, nzdirty]], indexing='ij')
    dft = pynu*0.
    for i in range(N):
        dft += (
            vis[i]*np.exp(2j*np.pi*(x*pos[i, 0]*dstx + y*pos[i, 1]*dsty + z*pos[i, 2]*dstz))).real
    ift.myassert(_l2error(dft, pynu) < eps*10)


@pmp('eps', [1e-2, 1e-6, 3e-13])
@pmp('space', [ift.RGSpace(128),
               ift.RGSpace([32, 64]),
               ift.RGSpace([10, 27, 32])])
@pmp('N', [1, 10, 100])
def test_build_nufft(space, N, eps):
    pos = ift.random.current_rng().random((N, len(space.shape))) - 0.5
    RF = ift.Nufft(space, pos=pos, eps=eps)
    flt = np.float64
    cmplx = np.complex128
    ift.extra.check_linear_operator(RF, cmplx, flt, only_r_linear=True,
                                    rtol=eps, atol=eps, no_device_copies=False)


@pmp('eps', [1e-4, 1e-7, 1e-10, 1e-11, 1e-12, 2e-13])
@pmp('shape', [(34,), (127,), (34, 50), (129, 100), (10, 10, 10)])
@pmp('N', [1, 10, 100])
@pmp('pre_domain', [None, ift.UnstructuredDomain(3)])
def test_variable_position_nufft_consistency(shape, N, eps, pre_domain):
    dst = (0.2, 0.99, 9.213)
    ndim = len(shape)
    dom = ift.RGSpace(shape, dst[:ndim])
    dst = np.array(dom.distances)

    ddom = ift.UnstructuredDomain(N), ift.UnstructuredDomain(ndim)
    coord = (ift.from_random(ddom).asnumpy() - 0.5) / dst[None]
    coord = ift.makeField(ddom, coord)

    op = ift.VariablePositionNufft(dom, N, epsilon=eps, pre_domain=pre_domain)
    inp = ift.MultiField.from_dict({"grid": ift.from_random(op.domain["grid"], dtype=complex),
                                    "coord": coord})
    res = op(inp).asnumpy()

    ref = np.empty_like(res)
    grid = inp["grid"].asnumpy()
    coord = inp["coord"].asnumpy()

    if pre_domain is None:
        grid = grid[None]
        ref = ref[None]
    corfact = -2j*np.pi*dst
    xyz = np.meshgrid(*[(-(ss//2) + np.arange(ss))*cf
                        for ss, cf in zip(dom.shape, corfact)],
                      indexing='ij')
    for islc in range(grid.shape[0]):
        for i in range(N):
            phase = sum([a*b for a,b in zip(xyz, coord[i])])
            ref[islc, i] = np.sum(grid[islc]*np.exp(phase))

    if pre_domain is None:
        ref = ref[0]

    ift.myassert(_l2error(ref, res) < eps)
    ift.extra.check_operator(op, inp, ntries=5, tol=10*eps)


@pmp('eps', [1e-4, 1e-7, 1e-10, 1e-11, 1e-12, 2e-13])
@pmp('shape', [(34,), (1001,), (100, 100), (35, 199), (20, 20, 21)])
@pmp('pre_domain', [None, ift.UnstructuredDomain(3)])
def test_variable_position_nufft_vs_fft(shape, eps, pre_domain):
    dst = (0.2, 0.99, 28.1)
    ndim = len(shape)
    dom = ift.RGSpace(shape, dst[:ndim])
    if pre_domain is not None:
        dom = (pre_domain, dom)
    dom = ift.DomainTuple.make(dom)

    op1 = ift.ShiftedPositionFFT(dom[-1], eps, pre_domain)
    op0 = ift.FFTOperator(dom, space=None if pre_domain is None else 1)
    assert op1.domain["grid"] == ift.makeDomain(dom)
    assert op1.domain["delta_coord"] == ift.makeDomain((dom[-1], ift.UnstructuredDomain(ndim)))
    inp = ift.MultiField.from_dict({"grid": ift.from_random(op1.domain["grid"], dtype=complex)},
                                   domain=op1.domain)
    ref = op0(inp["grid"]).asnumpy()
    res = op1(inp).asnumpy()
    ift.myassert(_l2error(ref, res) < eps)

    delta_coord = np.zeros(op1.domain["delta_coord"].shape)
    if ndim == 1:
        delta_coord[1, 0] = -1
        delta_coord[2, 0] = 2
    elif ndim == 2:
        delta_coord[1, 0, 0] = -1
        delta_coord[0, 2, 1] = 2
        delta_coord[3, 3, 0] = 1
        delta_coord[3, 3, 1] = -2
    elif ndim == 3:
        delta_coord[6, 4, 3, 0] = 3
        delta_coord[6, 4, 3, 1] = -1
        delta_coord[6, 4, 3, 2] = -2
        delta_coord[2, 1, 10, 0] = -5
        delta_coord[2, 1, 10, 1] = 20
        delta_coord[2, 1, 10, 2] = 30

    delta_coord = ift.makeField(op1.domain["delta_coord"], delta_coord)
    inp = ift.MultiField.from_dict(
        {
            "grid": ift.from_random(dom, dtype=complex),
            "delta_coord": delta_coord,
        }
    )
    res = op1(inp).asnumpy()
    if pre_domain is None:
        res = res[None]
        ntrans = 1
    else:
        ntrans = pre_domain.shape[0]

    for itrans in range(ntrans):
        if ndim == 1:
            np.testing.assert_allclose(res[itrans, 0], res[itrans, 1])
            np.testing.assert_allclose(res[itrans, 2], res[itrans, 4])
        elif ndim == 2:
            np.testing.assert_allclose(res[itrans, 1, 0], res[itrans, 0, 0])
            np.testing.assert_allclose(res[itrans, 0, 2], res[itrans, 0, 4])
            np.testing.assert_allclose(res[itrans, 3, 3], res[itrans, 4, 1])
        elif ndim == 3:
            np.testing.assert_allclose(res[itrans, 6, 4, 3], res[itrans, 9, 3, 1])
            np.testing.assert_allclose(res[itrans, 2, 1, 10], res[itrans, 17, 1, 19]) # Test wraparound
