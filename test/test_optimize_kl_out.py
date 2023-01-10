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
# Copyright(C) 2023 Vincent Eberle
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.
import os

import pytest
import h5py
import astropy.io.fits as ast
from numpy.testing import assert_array_equal

outroot = "demos/getting_started_3_results/signal/"

@pytest.mark.skipif(not os.path.exists(outroot), reason="This test only works after running the demos")
def test_optimize_kl_operator_output():

    with ast.open(outroot+"last_std.fits") as f:
        f1 = f[0].data
    with ast.open(outroot+"last_mean.fits") as f:
        f3 = f[0].data
    with h5py.File(outroot+"last.hdf5", "r") as g:
        f2 = g["stats"]["standard deviation"][:]
        f4 = g["stats"]["mean"][:]

    assert_array_equal(f1, f2.T)
    assert_array_equal(f3, f4.T)
