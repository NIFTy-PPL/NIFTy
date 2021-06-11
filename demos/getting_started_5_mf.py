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
    N = ift.ScalingOperator(data_space, noise)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response.domain, 'normal')
    data = signal_response(mock_position) + N.draw_sample_with_dtype(dtype=np.float64)

    plot = ift.Plot()
    plot.add(signal(mock_position), title='Ground Truth')
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([pspec1.force(mock_position)], title='Power Spectrum 1')
    plot.add([pspec2.force(mock_position)], title='Power Spectrum 2')
    plot.output(ny=2, nx=2, xsize=10, ysize=10, name=filename.format("setup"))

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name='Sampling',
                                               deltaE=0.01,
                                               iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton',
                                             deltaE=0.01,
                                             iteration_limit=35)
    ic_sampling.enable_logging()
    ic_newton.enable_logging()
    minimizer = ift.NewtonCG(ic_newton, enable_logging=True)

    # number of samples used to estimate the KL
    N_samples = 20

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @ signal_response
    H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

    # Begin minimization
    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    for i in range(5):
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL(mean, H, N_samples, True)
        KL, convergence = minimizer(KL)
        mean = KL.position

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(signal(mock_position), title="ground truth")
        plot.add(signal(KL.position), title="reconstruction")
        plot.add([pspec1.force(KL.position),
                  pspec1.force(mock_position)],
                 title="power1")
        plot.add([pspec2.force(KL.position),
                  pspec2.force(mock_position)],
                 title="power2")
        plot.add((ic_newton.history, ic_sampling.history,
                  minimizer.inversion_history),
                 label=['KL', 'Sampling', 'Newton inversion'],
                 title='Cumulative energies', s=[None, None, 1],
                 alpha=[None, 0.2, None])
        plot.output(nx=3,
                    ny=2,
                    ysize=10,
                    xsize=15,
                    name=filename.format("loop_{:02d}".format(i)))

    # Done, draw posterior samples
    sc = ift.StatCalculator()
    scA1 = ift.StatCalculator()
    scA2 = ift.StatCalculator()
    powers1 = []
    powers2 = []
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))
        p1 = pspec1.force(sample + KL.position)
        p2 = pspec2.force(sample + KL.position)
        scA1.add(p1)
        powers1.append(p1)
        scA2.add(p2)
        powers2.append(p2)

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    plot.add(sc.mean, title="Posterior Mean")
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    powers1 = [pspec1.force(s + KL.position) for s in KL.samples]
    powers2 = [pspec2.force(s + KL.position) for s in KL.samples]
    plot.add(powers1 + [scA1.mean, pspec1.force(mock_position)],
             title="Sampled Posterior Power Spectrum 1",
             linewidth=[1.]*len(powers1) + [3., 3.])
    plot.add(powers2 + [scA2.mean, pspec2.force(mock_position)],
             title="Sampled Posterior Power Spectrum 2",
             linewidth=[1.]*len(powers2) + [3., 3.])
    plot.output(ny=2, nx=2, xsize=15, ysize=15, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))


if __name__ == '__main__':
    main()
