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



@pmp('interpolator', ["FFTInterpolator", "LinearInterpolator"])
def test_grid_points(interpolator):
    res = 64
    vol = 2
    sp = ift.RGSpace([res, res], [vol/res, vol/res])
    mg = np.mgrid[(slice(0,res),)*2]
    mg = np.array(list(map(np.ravel, mg)))

    dist = [list(sp.distances)]
    dist = np.array(dist).reshape(-1, 1)

    sampling_points = dist * mg
    R = getattr(ift, interpolator)(sp, sampling_points)

    ift.extra.check_linear_operator(R, atol=1e-7, rtol=1e-7)
    inp = ift.from_random(R.domain)
    out = R(inp).val
    np.testing.assert_allclose(out, inp.val.reshape(-1), rtol=1e-7) #Fails otherwise....

# sampling_points = np.array([[0.25], [0.]])
# R = ift.FFTInterpolator(sp, sampling_points)
# R1 = ift.LinearInterpolator(sp, sampling_points)

# p = ift.Plot()
# p.add(R.adjoint(ift.full(R.target, 1)), title="FFT")
# p.add(R1.adjoint(ift.full(R.target, 1)), title="Linear")
# p.output(name="debug.png", ny=1, xsize=12)



#TODO Generate one Fourriermode, read out between gridpoints, check if right value
#FIXME unfortunately this fails with relative tol of 1e-6
