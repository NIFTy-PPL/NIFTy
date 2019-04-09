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
from numpy.testing import assert_allclose

import nifty5 as ift

np.random.seed(40)

pmp = pytest.mark.parametrize


@pmp('nu', [12, 128])
@pmp('nv', [4, 12, 128])
@pmp('N', [1, 10, 100])
def test_gridding(nu, nv, N):
    uv = np.random.rand(N, 2) - 0.5
    vis = np.random.randn(N) + 1j*np.random.randn(N)

    # Nifty
    GM = ift.GridderMaker(ift.RGSpace((nu, nv)))
    # re-order for performance
    idx = GM.getReordering(uv)
    uv, vis = uv[idx], vis[idx]
    vis = ift.from_global_data(ift.UnstructuredDomain(vis.shape), vis)

    Op = GM.getFull(uv)
    pynu = Op(vis).to_global_data()
    # DFT
    x, y = np.meshgrid(
        *[-ss/2 + np.arange(ss) for ss in [nu, nv]], indexing='ij')
    dft = pynu*0.
    for i in range(N):
        dft += vis.val[i]*np.exp(2j*np.pi*(x*uv[i, 0] + y*uv[i, 1]))
    assert_allclose(dft, pynu)


@pmp('eps', [1e-2, 1e-6, 1e-15])
@pmp('nu', [12, 128])
@pmp('nv', [4, 12, 128])
@pmp('N', [1, 10, 100])
def test_build(nu, nv, N, eps):
    dom = ift.RGSpace([nu, nv])
    uv = np.random.rand(N, 2) - 0.5
    GM = ift.GridderMaker(dom)
    # re-order for performance
    idx = GM.getReordering(uv)
    uv = uv[idx]
    R0 = GM.getGridder(uv)
    R1 = GM.getRest()
    R = R1@R0
    RF = GM.getFull(uv)

    # Consistency checks
    dt = np.complex
    ift.extra.consistency_check(R0, domain_dtype=dt, target_dtype=dt)
    ift.extra.consistency_check(R1, domain_dtype=dt, target_dtype=dt)
    ift.extra.consistency_check(R, domain_dtype=dt, target_dtype=dt)
    ift.extra.consistency_check(RF, domain_dtype=dt, target_dtype=dt)
