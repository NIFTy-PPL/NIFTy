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
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

############################################################
# Non-linear tomography
#
# The signal is a sigmoid-normal distributed field.
# The data is the field integrated along lines of sight that are
# randomly (set mode=0) or radially (mode=1) distributed
#
# Demo takes a while to compute
#############################################################

import sys

import numpy as np

import nifty8 as ift

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
except ImportError:
    comm = None
    master = True


class SingleDomain(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return ift.makeField(self._tgt(mode), x.val)


def random_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends


def radial_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(0.5 + 0*ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends


def main():
    # Choose between random line-of-sight response (mode=0) and radial lines
    # of sight (mode=1)
    if len(sys.argv) == 2:
        mode = int(sys.argv[1])
    else:
        mode = 0

    # Preparing the filename string for store results
    filename = "getting_started_mf_mode_{}_".format(mode) + "{}.png"

    # Set up signal domain
    npix1, npix2 = 128, 128
    position_space = ift.RGSpace([npix1, npix2])
    sp1 = ift.RGSpace(npix1)
    sp2 = ift.RGSpace(npix2)

    # Set up signal model
    cfmaker = ift.CorrelatedFieldMaker('')
    cfmaker.add_fluctuations(sp1, (0.1, 1e-2), (2, .2), (.01, .5), (-4, 2.),
                             'amp1')
    cfmaker.add_fluctuations(sp2, (0.1, 1e-2), (2, .2), (.01, .5), (-3, 1),
                             'amp2')
    cfmaker.set_amplitude_total_offset(0., (1e-2, 1e-6))
    correlated_field = cfmaker.finalize()

    normalized_amp = cfmaker.get_normalized_amplitudes()
    pspec1 = normalized_amp[0]**2
    pspec2 = normalized_amp[1]**2
    DC = SingleDomain(correlated_field.target, position_space)

    # Apply a nonlinearity
    signal = DC @ ift.sigmoid(correlated_field)

    # Build the line-of-sight response and define signal response
    LOS_starts, LOS_ends = random_los(100) if mode == 0 else radial_los(100)
    R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    signal_response = R(signal)

    # Specify noise
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(data_space, noise, float)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response.domain, 'normal')
    data = signal_response(mock_position) + N.draw_sample()

    plot = ift.Plot()
    plot.add(signal(mock_position), title='Ground Truth')
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([pspec1.force(mock_position)], title='Power Spectrum 1')
    plot.add([pspec2.force(mock_position)], title='Power Spectrum 2')
    plot.output(ny=2, nx=2, xsize=10, ysize=10, name=filename.format("setup"))

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name='Sampling', deltaE=0.01, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.01, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton, enable_logging=True)

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = ift.GaussianEnergy(data, inverse_covariance=N.inverse) @ signal_response

    n_samples = 20
    n_iterations = 5
    samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples,
                              minimizer, ic_sampling, None, comm=comm,
                              output_directory="getting_started_5_results",
                              export_operator_outputs={"signal": signal, "power spectrum 1": pspec1,
                                                       "power spectrum 2": pspec2})

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    mean, var = samples.sample_stat(signal)
    plot.add(mean, title="Posterior Mean")
    plot.add(ift.sqrt(var), title="Posterior Standard Deviation")

    n_samples = samples.n_samples
    plot.add(list(samples.iterator(pspec1)) + [samples.average(pspec1.log()).exp(),
              pspec1.force(mock_position)],
             title="Sampled Posterior Power Spectrum 1",
             linewidth=[1.]*n_samples + [3., 3.])
    plot.add(list(samples.iterator(pspec2)) + [samples.average(pspec2.log()).exp(),
              pspec2.force(mock_position)],
             title="Sampled Posterior Power Spectrum 2",
             linewidth=[1.]*n_samples + [3., 3.])
    if master:
        plot.output(ny=2, nx=2, xsize=15, ysize=15, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))


if __name__ == '__main__':
    main()
