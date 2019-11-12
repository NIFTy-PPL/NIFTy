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

class SingleDomain(ift.LinearOperator):
    def __init__(self,domain,target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
    def apply(self,x,mode):
        self._check_input(x,mode)
        return ift.from_global_data(self._tgt(mode),x.to_global_data())

def random_los(n_los):
    starts = list(np.random.uniform(0, 1, (n_los, 2)).T)
    ends = list(np.random.uniform(0, 1, (n_los, 2)).T)
    return starts, ends


def radial_los(n_los):
    starts = list(np.random.uniform(0, 1, (n_los, 2)).T)
    ends = list(0.5 + 0*np.random.uniform(0, 1, (n_los, 2)).T)
    return starts, ends


if __name__ == '__main__':
    np.random.seed(45)

    # Choose between random line-of-sight response (mode=0) and radial lines
    # of sight (mode=1)
    if len(sys.argv) == 2:
        mode = int(sys.argv[1])
    else:
        mode = 0
    filename = "getting_started_mf_mode_{}_".format(mode) + "{}.png"

    npix1, npix2 = 128, 128
    position_space = ift.RGSpace([npix1, npix2])
    sp1 = ift.RGSpace(npix1)
    sp2 = ift.RGSpace(npix2)
    
    power_space1 = ift.PowerSpace(sp1.get_default_codomain())
    power_space2 = ift.PowerSpace(sp2.get_default_codomain())

    cfmaker = ift.CorrelatedFieldMaker()
    amp1 = 0.5
    cfmaker.add_fluctuations(power_space1,
                             amp1, 1e-2,
                             1, .1,
                             .01, .5,
                             -2, 1.,
                             'amp1')
    cfmaker.add_fluctuations(power_space2,
                             np.sqrt(1.-amp1**2), 1e-2,
                             1, .1,
                             .01, .5,
                             -1.5, .5,
                             'amp2')
    correlated_field = cfmaker.finalize(1e-3, 1e-6, '')
    sams = [ift.from_random('normal',correlated_field.domain)
            for _ in range(20)]

    print("Prior expected total fluctuations: "+str(
          cfmaker.stats(cfmaker.total_fluctuation,sams)[0]))
    
    A1 = cfmaker.amplitudes[0]
    A2 = cfmaker.amplitudes[1]
    DC = SingleDomain(correlated_field.target,position_space)

    # Apply a nonlinearity
    signal = DC @ ift.sigmoid(correlated_field)

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
    ic_sampling = ift.AbsDeltaEnergyController(
        name='Sampling', deltaE=0.01, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(
        name='Newton', deltaE=0.01, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)

    # Set up likelihood and information Hamiltonian
    likelihood = ift.GaussianEnergy(mean=data,
                                    inverse_covariance=N.inverse)(signal_response)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    plot = ift.Plot()
    plot.add(signal(mock_position), title='Ground Truth')
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([A1.force(mock_position)], title='Power Spectrum 1')
    plot.add([A2.force(mock_position)], title='Power Spectrum 2')
    plot.output(ny=2, nx=2, xsize=10, ysize=10, name=filename.format("setup"))

    # number of samples used to estimate the KL
    N_samples = 20

    # Draw new samples to approximate the KL five times
    for i in range(10):
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL(mean, H, N_samples)
        KL, convergence = minimizer(KL)
        mean = KL.position

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(signal(mock_position), title="ground truth")
        plot.add(signal(KL.position), title="reconstruction")
        plot.add([A1.force(KL.position), A1.force(mock_position)], title="power1")
        plot.add([A2.force(KL.position), A2.force(mock_position)], title="power2")
        plot.output(nx = 2, ny=2, ysize=10, xsize=10,
                    name=filename.format("loop_{:02d}".format(i)))

    # Draw posterior samples
    Nsamples = 20
    KL = ift.MetricGaussianKL(mean, H, N_samples)
    sc = ift.StatCalculator()
    scA1 = ift.StatCalculator()
    scA2 = ift.StatCalculator()
    powers1 = []
    powers2 = []
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))
        p1 = A1.force(sample + KL.position)
        p2 = A2.force(sample + KL.position)
        scA1.add(p1)
        powers1.append(p1)
        scA2.add(p2)
        powers2.append(p2)

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    plot.add(sc.mean, title="Posterior Mean")
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    powers1 = [A1.force(s + KL.position) for s in KL.samples]
    powers2 = [A2.force(s + KL.position) for s in KL.samples]
    plot.add(
        powers1 + [scA1.mean,
                   A1.force(mock_position)],
        title="Sampled Posterior Power Spectrum 1",
        linewidth=[1.]*len(powers1) + [3., 3.])
    plot.add(
        powers2 + [scA2.mean,
                   A2.force(mock_position)],
        title="Sampled Posterior Power Spectrum 2",
        linewidth=[1.]*len(powers2) + [3., 3.])
    plot.output(ny=2, nx=2, xsize=15, ysize=15, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))
