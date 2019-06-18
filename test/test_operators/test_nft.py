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
from numpy.testing import assert_allclose, assert_

import nifty5 as ift

np.random.seed(40)

pmp = pytest.mark.parametrize


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))

speedOfLight = 299792458.

@pmp('eps', [1e-2, 1e-4, 1e-7, 1e-10, 1e-11, 1e-12, 2e-13])
@pmp('nu', [12, 128])
@pmp('nv', [4, 12, 128])
@pmp('N', [1, 10, 100])
@pmp('freq', [1e9])
def test_gridding(nu, nv, N, eps, freq):
    fovx = 0.0001
    fovy = 0.0002
    uvw = (np.random.rand(N, 3) - 0.5)
    uvw[:,0] /= fovx*freq/speedOfLight
    uvw[:,1] /= fovy*freq/speedOfLight
    vis = (np.random.randn(N) + 1j*np.random.randn(N)).reshape((-1,1))

    # Nifty
    GM = ift.GridderMaker(ift.RGSpace((nu, nv)), uvw=uvw,
                          freq=np.array([freq]), eps=eps, fovx=fovx, fovy=fovy,
                          flags=np.zeros((N, 1), dtype=np.bool))
    vis2 = ift.from_global_data(ift.UnstructuredDomain(vis.shape), vis)

    Op = GM.getFull()
    pynu = Op(vis2).to_global_data()
    # DFT
    x, y = np.meshgrid(
        *[-ss/2 + np.arange(ss) for ss in [nu, nv]], indexing='ij')
    x *= fovx*freq/speedOfLight
    y *= fovy*freq/speedOfLight
    dft = pynu*0.
    for i in range(N):
        dft += (vis[i]*np.exp(2j*np.pi*(x*uvw[i, 0] + y*uvw[i, 1]))).real
    assert_(_l2error(dft, pynu) < eps)


@pmp('eps', [1e-2, 1e-6, 2e-13])
@pmp('nu', [12, 128])
@pmp('nv', [4, 12, 128])
@pmp('N', [1, 10, 100])
@pmp('freq', [np.array([1e9]), np.array([1e9, 2e9, 2.5e9])])
def test_build(nu, nv, N, eps, freq):
    dom = ift.RGSpace([nu, nv])
    fov = np.pi/180/60
    uvw = np.random.rand(N, 3) - 0.5
    flags=np.zeros((N, freq.shape[0]), dtype=np.bool)
    flags[0,0]=True
    GM = ift.GridderMaker(dom, uvw=uvw, freq=freq, eps=eps,
                          flags=flags, fovx=fov, fovy=fov)
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
