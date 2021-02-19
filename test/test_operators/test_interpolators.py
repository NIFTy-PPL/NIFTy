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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest

import nifty7 as ift

pmp = pytest.mark.parametrize


@pmp("interpolator", ["FFTInterpolator", "LinearInterpolator"])
def test_grid_points(interpolator):
    res = 64
    vol = 2
    sp = ift.RGSpace([res, res], [vol / res, vol / res])
    mg = np.mgrid[(slice(0, res),) * 2]
    mg = np.array(list(map(np.ravel, mg)))

    dist = [list(sp.distances)]
    dist = np.array(dist).reshape(-1, 1)

    sampling_points = dist * mg
    R = getattr(ift, interpolator)(sp, sampling_points)

    ift.extra.check_linear_operator(R, atol=1e-7, rtol=1e-7)
    inp = ift.from_random(R.domain)
    out = R(inp).val
    np.testing.assert_allclose(out, inp.val.reshape(-1), rtol=1e-7)


@pmp("npix", [37, 100])
@pmp("vol", [1., 4.23])
@pmp("phi", [0., 180., 31.2])
@pmp("k", [0.32, 1.0, 1.32, 10.0])
def test_fourier_interpolation(npix, vol, phi, k):
    sp = ift.RGSpace(npix, vol / npix)

    f = lambda x: np.sin(k * x / npix * 2 * np.pi + phi * np.pi / 180)
    xs = np.mgrid[0:npix]*vol/npix
    ys = ift.makeField(sp, f(xs))

    pos = ift.random.Random.uniform(np.float, (1, 5)) * vol

    pos = (np.arange(2*npix)/2/npix*vol)[None]
    op = ift.FFTInterpolator(sp, pos)
    res0 = op(ys)
    res1 = f(ift.makeField(op.target, pos[0]))
    tol = 1e-4
    # import pylab as plt
    # plt.plot(res0.val)
    # plt.plot(res1.val)
    # plt.show()
    ift.extra.assert_allclose(res0, res1, rtol=tol, atol=tol)
