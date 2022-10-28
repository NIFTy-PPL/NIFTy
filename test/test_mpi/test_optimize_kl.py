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
# Copyright(C) 2013-2022 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from tempfile import TemporaryDirectory

import nifty8 as ift
import numpy as np
import pytest

from ..common import setup_function, teardown_function

comm = ift.utilities.get_MPI_params()[0]
master = True if comm is None else comm.Get_rank() == 0

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
@pmp("n_samples", [0, 2])
@pmp("dry_run", [False, True])
def test_optimize_kl(constants, point_estimates, kl_minimizer, n_samples,
                     sampling_iteration_controller, nonlinear_sampling_minimizer,
                     dry_run):
    final_index = 5

    foo, output_directory = _create_temp_outputdir()
    bar, output_directory1 = _create_temp_outputdir()

    initial_position = None
    initial_index = 0

    d = sky(mock_pos)
    likelihood_energy = ift.GaussianEnergy(d) @ sky
    inspect_callback = None
    terminate_callback = None
    transitions = None
    export_operator_outputs = {}
    rand_state = ift.random.getState()
    sl = ift.optimize_kl(likelihood_energy, final_index, n_samples, kl_minimizer,
                         sampling_iteration_controller, nonlinear_sampling_minimizer, constants,
                         point_estimates, transitions, export_operator_outputs, output_directory, initial_position,
                         initial_index, comm, inspect_callback,
                         terminate_callback, save_strategy="all", plot_energy_history=False,
                         plot_minisanity_history=False, dry_run=dry_run)

    ift.random.setState(rand_state)

    def terminate_callback(iglobal):
        return iglobal in [0, 3]

    for _ in range(5):
        sl1 = ift.optimize_kl(likelihood_energy, final_index, n_samples, kl_minimizer,
                              sampling_iteration_controller, nonlinear_sampling_minimizer, constants,
                              point_estimates, transitions, export_operator_outputs, output_directory1, initial_position,
                              initial_index, comm, inspect_callback,
                              terminate_callback, resume=True, save_strategy="last",
                              plot_energy_history=False, plot_minisanity_history=False, dry_run=dry_run)

    for aa, bb in zip(sl.iterator(), sl1.iterator()):
        ift.extra.assert_allclose(aa, bb)


def test_transitions():
    final_index = 3

    foo, output_directory = _create_temp_outputdir()

    initial_position = None
    initial_index = 0
    n_samples = 2

    kl_minimizer = ift.SteepestDescent(ic)
    # ift.NewtonCG(ift.GradInfNormController(1e-3, iteration_limit=1))
    sampling_iteration_controller = ic
    nonlinear_sampling_minimizer = None
    constants = []
    point_estimates = []

    dom0 = ift.UnstructuredDomain(5)
    dom1 = ift.UnstructuredDomain(7)
    mdom0 = ift.makeDomain({"in0": dom0})
    mdom1 = ift.makeDomain({"in0": dom1})
    mdom2 = ift.makeDomain({"in1": dom1})


    def likelihood_energy(iglobal):
        if iglobal == 0:
            return ift.GaussianEnergy(domain=mdom0, sampling_dtype=float)
        elif iglobal == 1:
            return ift.GaussianEnergy(domain=mdom1, sampling_dtype=float)
        else:
            return ift.GaussianEnergy(domain=mdom2, sampling_dtype=float)


    def transitions(iglobal):
        if iglobal == 0:
            return None
        elif iglobal == 1:
            mask = np.zeros(dom1.shape)
            mask[:2] = 1
            op = ift.MaskOperator(ift.makeField(dom1, mask)).adjoint.ducktape("in0").ducktape_left("in0")
            return lambda sl: op(sl._m)
        elif iglobal == 2:
            fa = ift.FieldAdapter(dom1, "in0")
            op = fa.ducktape_left("in1")
            return lambda sl: op(sl._m)

        raise RuntimeError


    inspect_callback = None
    terminate_callback = None
    export_operator_outputs = {}

    sl = ift.optimize_kl(likelihood_energy, final_index, n_samples, kl_minimizer,
                         sampling_iteration_controller, nonlinear_sampling_minimizer, constants,
                         point_estimates, transitions, export_operator_outputs, output_directory, initial_position,
                         initial_index, comm, inspect_callback, terminate_callback, save_strategy="all",
                         plot_energy_history=False, plot_minisanity_history=False)
    assert sl.domain is mdom2


def _create_temp_outputdir():
    if master:
        output_directory = TemporaryDirectory()
        name = output_directory.name
    else:
        name = output_directory = None
    if comm is not None:
        name = comm.bcast(name, 0)
    return output_directory, name
