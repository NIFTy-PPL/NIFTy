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

import nifty8 as ift
import numpy as np
import pytest

pmp = pytest.mark.parametrize


def test_grid_points():
    res = 64
    vol = 2
    sp = ift.RGSpace([res, res], [vol / res, vol / res])
    mg = np.mgrid[(slice(0, res),) * 2]
    mg = np.array(list(map(np.ravel, mg)))

    dist = [list(sp.distances)]
    dist = np.array(dist).reshape(-1, 1)

    sampling_points = dist * mg
    R = ift.LinearInterpolator(sp, sampling_points)

    ift.extra.check_linear_operator(R, atol=1e-7, rtol=1e-7)
    inp = ift.from_random(R.domain)
    out = R(inp).val
    np.testing.assert_allclose(out, inp.val.reshape(-1), rtol=1e-7)
