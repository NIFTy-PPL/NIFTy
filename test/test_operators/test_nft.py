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
# Copyright(C) 2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_

import nifty5 as ift

np.random.seed(40)

pmp = pytest.mark.parametrize


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


@pmp('eps', [1e-2, 1e-4, 1e-7, 1e-10, 1e-11, 1e-12, 2e-13])
@pmp('nu', [12, 128])
@pmp('nv', [4, 12, 128])
@pmp('N', [1, 10, 100])
def test_gridding(nu, nv, N, eps):
    uv = np.random.rand(N, 2) - 0.5
    vis = np.random.randn(N) + 1j*np.random.randn(N)

    # Nifty
    dom = ift.RGSpace((nu, nv), distances=(0.2, 1.12))
    dstx, dsty = dom.distances
    uv[:, 0] = uv[:, 0]/dstx
    uv[:, 1] = uv[:, 1]/dsty
    GM = ift.GridderMaker(dom, uv=uv, eps=eps)
    vis2 = ift.from_global_data(ift.UnstructuredDomain(vis.shape), vis)

    Op = GM.getFull()
    pynu = Op(vis2).to_global_data()
    # DFT
    x, y = np.meshgrid(
        *[-ss/2 + np.arange(ss) for ss in [nu, nv]], indexing='ij')
    dft = pynu*0.
    for i in range(N):
        dft += (
            vis[i]*np.exp(2j*np.pi*(x*uv[i, 0]*dstx + y*uv[i, 1]*dsty))).real
    assert_(_l2error(dft, pynu) < eps)


def test_cartesian():
    nx, ny = 2, 6
    dstx, dsty = 0.3, 0.2
    dom = ift.RGSpace((nx, ny), (dstx, dsty))

    kx = np.fft.fftfreq(nx, dstx)
    ky = np.fft.fftfreq(ny, dsty)
    uu, vv = np.meshgrid(kx, ky)
    tmp = np.vstack([uu[None, :], vv[None, :]])
    uv = np.transpose(tmp, (2, 1, 0)).reshape(-1, 2)

    GM = ift.GridderMaker(dom, uv=uv)
    op = GM.getFull().adjoint

    fld = ift.from_random('normal', dom)
    arr = fld.to_global_data()

    fld2 = ift.from_global_data(dom, np.roll(arr, (nx//2, ny//2), axis=(0, 1)))
    res = op(fld2).to_global_data().reshape(nx, ny)

    fft = ift.FFTOperator(dom.get_default_codomain(), target=dom).adjoint
    vol = ift.full(dom, 1.).integrate()
    res1 = fft(fld).to_global_data()

    # FIXME: we don't understand the conjugate() yet
    np.testing.assert_allclose(res, res1.conjugate()*vol)


@pmp('eps', [1e-2, 1e-6, 2e-13])
@pmp('nu', [12, 128])
@pmp('nv', [4, 12, 128])
@pmp('N', [1, 10, 100])
def test_build(nu, nv, N, eps):
    dom = ift.RGSpace([nu, nv])
    uv = np.random.rand(N, 2) - 0.5
    GM = ift.GridderMaker(dom, uv=uv, eps=eps)
    R0 = GM.getGridder()
    R1 = GM.getRest()
    R = R1@R0
    RF = GM.getFull()

    # Consistency checks
    flt = np.float64
    cmplx = np.complex128
    ift.extra.consistency_check(R0, cmplx, flt, only_r_linear=True)
    ift.extra.consistency_check(R1, flt, flt)
    ift.extra.consistency_check(R, cmplx, flt, only_r_linear=True)
    ift.extra.consistency_check(RF, cmplx, flt, only_r_linear=True)
