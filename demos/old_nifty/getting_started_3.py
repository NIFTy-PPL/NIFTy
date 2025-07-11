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

ift.random.push_sseq_from_seed(27)

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
except ImportError:
    comm = None
    master = True


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
    filename = "getting_started_3_mode_{}_".format(mode) + "{}.png"
    position_space = ift.RGSpace([128, 128])

    #  For a detailed showcase of the effects the parameters
    #  of the CorrelatedField model have on the generated fields,
    #  see 'getting_started_4_CorrelatedFields.ipynb'.

    args = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),

        # Amplitude of field fluctuations
        'fluctuations': (1., 0.8),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-3., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (2, 1.),  # 1.0, 0.5

        # How ragged the integrated Wiener process component is
        'asperity': (0.5, 0.4)  # 0.1, 0.5
    }

    correlated_field = ift.SimpleCorrelatedField(position_space, **args)
    pspec = correlated_field.power_spectrum

    # Apply a nonlinearity
    signal = ift.sigmoid(correlated_field)

    # Build the line-of-sight response and define signal response
    LOS_starts, LOS_ends = random_los(100) if mode == 0 else radial_los(100)
    R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    signal_response = R(signal)

    # Specify noise
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(data_space, noise, np.float64)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response.domain, 'normal')
    data = signal_response(mock_position) + N.draw_sample()

    # Plot setup
    plot = ift.Plot()
    plot.add(signal(mock_position), title='Ground Truth', vmin=0, vmax=1)
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([pspec.force(mock_position)], title='Power Spectrum')
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename.format("setup"))

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
                                               deltaE=0.05, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5,
                                             convergence_level=2, iteration_limit=35)
    ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
                                                  deltaE=0.5, iteration_limit=15,
                                                  convergence_level=2)
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Set up likelihood energy
    likelihood_energy = (ift.GaussianEnergy(data, inverse_covariance=N.inverse) @
                         signal_response)

    # Minimize KL
    n_iterations = 6
    n_samples = lambda iiter: 10 if iiter < 5 else 20
    # TODO Write callback for nice plots
    samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples,
                              minimizer, ic_sampling, minimizer_sampling,
                              export_operator_outputs={"signal": signal},
                              output_directory="getting_started_3_results",
                              comm=comm)

    if True:
        # Load result from disk. May be useful for long inference runs, where
        # inference and posterior analysis are split into two steps
        samples = ift.ResidualSampleList.load("getting_started_3_results/pickle/last", comm=comm)

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    mean, var = samples.sample_stat(signal)
    plot.add(mean, title="Posterior Mean", vmin=0, vmax=1)
    plot.add(var.sqrt(), title="Posterior Standard Deviation", vmin=0)

    nsamples = samples.n_samples
    logspec = pspec.log()
    plot.add(list(samples.iterator(pspec)) +
             [pspec.force(mock_position), samples.average(logspec).exp()],
             title="Sampled Posterior Power Spectrum",
             linewidth=[1.]*nsamples + [3., 3.],
             label=[None]*nsamples + ['Ground truth', 'Posterior mean'])
    if master:
        plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename_res)
        print("Saved results as '{}'.".format(filename_res))


if __name__ == '__main__':
    main()
