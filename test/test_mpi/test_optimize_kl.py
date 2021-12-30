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
from numpy.testing import assert_allclose, assert_raises

import nifty8 as ift
from nifty8 import myassert

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize

point_estimates = []

dom = ift.RGSpace([20, 20])
logsky = ift.SimpleCorrelatedField(dom, 0., None, (1, 1), None, None, (-3, 1))
sky = logsky.exp()
mock_pos = ift.from_random(logsky.domain)

def cstfunc(iglobal):
    if iglobal == 0:
        return []
    if iglobal == 1:
        return ["xi"]
    if iglobal in [2, 3]:
        return ["fluctuations"]
    if iglobal == 4:
        return []
    return []

def pefunc(iglobal):
    if iglobal == 0:
        return []
    if iglobal == 1:
        return []
    if iglobal == 2:
        return ["xi"]
    if iglobal == 3:
        return ["fluctuations"]
    if iglobal == 4:
        return ["xi"]
    return []

ic = ift.GradientNormController(iteration_limit=5)

@pmp("constants", [[], ["fluctuations"], cstfunc])
@pmp("point_estimates", [[], ["fluctuations"], pefunc])
@pmp("kl_minimizer", [ift.SteepestDescent(ic),
                      lambda n: ift.NewtonCG(ic) if n < 2 else ift.VL_BFGS(ic)])
@pmp("sampling_iteration_controller", [ic])
@pmp("nonlinear_sampling_minimizer", [None, ift.NewtonCG(ift.GradInfNormController(1e-3, iteration_limit=1))])
def test_optimize_kl(constants, point_estimates, kl_minimizer,
                     sampling_iteration_controller, nonlinear_sampling_minimizer):
    n_samples = 2
    global_iterations = 5
    output_directory = "out"

    initial_position = None
    initial_index = 0
    ground_truth_position = None
    comm = ift.utilities.get_MPI_params()[0]
    overwrite = True

    d = sky(mock_pos)
    likelihood_energy = ift.GaussianEnergy(mean=d) @ sky
    callback = None
    plot_latent = False
    plottable_operators = {}
    ift.optimize_kl(likelihood_energy, global_iterations, n_samples, kl_minimizer,
                    sampling_iteration_controller, nonlinear_sampling_minimizer, constants,
                    point_estimates, plottable_operators, output_directory, initial_position,
                    initial_index, ground_truth_position, comm, overwrite, callback, plot_latent)
