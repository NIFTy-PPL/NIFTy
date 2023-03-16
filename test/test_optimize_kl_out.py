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

import h5py
import astropy.io.fits as ast
from numpy.testing import assert_array_equal

import nifty8 as ift


def test_optimize_kl_operator_output():
    sam_list = []
    dom = ift.RGSpace([32, 32])
    for i in range(10):
        tmp = ift.from_random(dom)
        sam_list.append(tmp)
    samples = ift.SampleList(sam_list)
    fname = "test_consistency"
    samples.save_to_fits(file_name_base=fname,
                         op=None,
                         samples=True,
                         mean=True,
                         std=True,
                         overwrite=True)
    samples.save_to_hdf5(file_name=fname+'.hdf5',
                         op=None,
                         samples=True,
                         mean=True,
                         std=True,
                         overwrite=True)

    mean, var = samples.sample_stat()
    mean = mean.val
    std = var.sqrt().val

    with ast.open(fname+"_mean.fits") as f:
        mean_fits = f[0].data
    with ast.open(fname+"_std.fits") as f:
        std_fits = f[0].data
    with h5py.File(fname+".hdf5", "r") as g:
        mean_hdf5 = g["stats"]["mean"][:]
        std_hdf5 = g["stats"]["standard deviation"][:]

    assert_array_equal(mean, mean_fits.T)
    assert_array_equal(std, std_fits.T)
    assert_array_equal(mean, mean_hdf5)
    assert_array_equal(std, std_hdf5)
