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
# Copyright(C) 2013-2020 Max-Planck-Society
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

import nifty7 as ift


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
        'fluctuations': (2., 1.),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-4., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (5, 2.),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': (0.5, 0.5)  # 0.1, 0.5
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
    sig = ift.ScalingOperator(data_space, np.sqrt(noise))
    N = ift.ScalingOperator(data_space, noise)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response.domain, 'normal')
    data = signal_response(mock_position) + N.draw_sample_with_dtype(dtype=np.float64)

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(
        deltaE=0.05, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(
        name='Newton', deltaE=0.5, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)

    # Set up likelihood and information Hamiltonian
    likelihood = (ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @
                  signal_response)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    plot = ift.Plot()
    plot.add(signal(mock_position), title='Ground Truth')
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([pspec.force(mock_position)], title='Power Spectrum')
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename.format("setup"))

    # number of samples used to estimate the KL
    N_samples = 20

    # Draw new samples to approximate the KL five times
    for i in range(5):
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL.make(mean, H, N_samples, True)
        KL, convergence = minimizer(KL)
        mean = KL.position
        ift.extra.minisanity(data, lambda x: N.inverse, signal_response,
                             KL.position, list(KL.samples))

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(signal(KL.position), title="Latent mean")
        plot.add([pspec.force(KL.position + ss) for ss in KL.samples],
                 title="Samples power spectrum")
        plot.output(ny=1, ysize=6, xsize=16,
                    name=filename.format("loop_{:02d}".format(i)))

    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    plot.add(sc.mean, title="Posterior Mean")
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    powers = [pspec.force(s + KL.position) for s in KL.samples]
    sc = ift.StatCalculator()
    for pp in powers:
        sc.add(pp)
    plot.add(
        powers + [pspec.force(mock_position),
                  pspec.force(KL.position), sc.mean],
        title="Sampled Posterior Power Spectrum",
        linewidth=[1.]*len(powers) + [3., 3., 3.],
        label=[None]*len(powers) + ['Ground truth', 'Posterior latent mean', 'Posterior mean'])
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))


if __name__ == '__main__':
    main()
