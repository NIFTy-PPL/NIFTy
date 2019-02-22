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
# Copyright(C) 2013-2019 Max-Planck-Society
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

import nifty5 as ift


def random_los(n_los):
    starts = list(np.random.uniform(0, 1, (n_los, 2)).T)
    ends = list(np.random.uniform(0, 1, (n_los, 2)).T)
    return starts, ends


def radial_los(n_los):
    starts = list(np.random.uniform(0, 1, (n_los, 2)).T)
    ends = list(0.5 + 0*np.random.uniform(0, 1, (n_los, 2)).T)
    return starts, ends


if __name__ == '__main__':
    np.random.seed(420)

    # Choose between random line-of-sight response (mode=0) and radial lines
    # of sight (mode=1)
    if len(sys.argv) == 2:
        mode = int(sys.argv[1])
    else:
        mode = 0
    filename = "getting_started_3_mode_{}_".format(mode) + "{}.png"

    position_space = ift.RGSpace([128, 128])
    harmonic_space = position_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(harmonic_space, position_space)
    power_space = ift.PowerSpace(harmonic_space)

    # Set up an amplitude operator for the field
    dct = {
        'target': power_space,
        'n_pix': 64,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im':  0,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3   # y-intercept variance
    }
    A = ift.SLAmplitude(**dct)

    # Build the operator for a correlated signal
    power_distributor = ift.PowerDistributor(harmonic_space, power_space)
    vol = harmonic_space.scalar_dvol**-0.5
    xi = ift.ducktape(harmonic_space, None, 'xi')
    correlated_field = ht(vol*power_distributor(A)*xi)
    # Alternatively, one can use:
    # correlated_field = ift.CorrelatedField(position_space, A)

    # Apply a nonlinearity
    signal = ift.sigmoid(correlated_field)

    # Build the line-of-sight response and define signal response
    LOS_starts, LOS_ends = random_los(100) if mode == 0 else radial_los(100)
    R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    signal_response = R(signal)

    # Specify noise
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(noise, data_space)

    # Generate mock signal and data
    mock_position = ift.from_random('normal', signal_response.domain)
    data = signal_response(mock_position) + N.draw_sample()

    # Minimization parameters
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(
        name='Newton', tol=1e-7, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)

    # Set up likelihood and information Hamiltonian
    likelihood = ift.GaussianEnergy(mean=data, covariance=N)(signal_response)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    plot = ift.Plot()
    plot.add(signal(mock_position), title='Ground Truth')
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([A.force(mock_position)], title='Power Spectrum')
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename.format("setup"))

    # number of samples used to estimate the KL
    N_samples = 20

    # Draw new samples to approximate the KL five times
    for i in range(5):
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL(mean, H, N_samples)
        KL, convergence = minimizer(KL)
        mean = KL.position

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(signal(KL.position), title="reconstruction")
        plot.add([A.force(KL.position), A.force(mock_position)], title="power")
        plot.output(ny=1, ysize=6, xsize=16,
                    name=filename.format("loop_{:02d}".format(i)))

    # Draw posterior samples
    KL = ift.MetricGaussianKL(mean, H, N_samples)
    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    plot.add(sc.mean, title="Posterior Mean")
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    powers = [A.force(s + KL.position) for s in KL.samples]
    plot.add(
        powers + [A.force(KL.position),
                  A.force(mock_position)],
        title="Sampled Posterior Power Spectrum",
        linewidth=[1.]*len(powers) + [3., 3.])
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))
