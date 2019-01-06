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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

############################################################
# Non-linear tomography 
# data is line of sight (LOS) field
# random lines (set mode=0), radial lines (mode=1)
#############################################################
mode = 0

import nifty5 as ift
import numpy as np


def get_random_LOS(n_los):
    # Setting up LOS
    starts = list(np.random.uniform(0, 1, (n_los, 2)).T)
    if mode == 0:
        ends = list(np.random.uniform(0, 1, (n_los, 2)).T)
    else:
        ends = list(0.5+0*np.random.uniform(0, 1, (n_los, 2)).T)
    return starts, ends


if __name__ == '__main__':
    # FIXME description of the tutorial
    np.random.seed(420)
    np.seterr(all='raise')
    position_space = ift.RGSpace([128, 128])

    # Setting up an amplitude model for the field
    A = ift.AmplitudeModel(position_space, 64, 3, 0.4, -5., 0.5, 0.4, 0.3)
    # made choices:
    # 64 spectral bins 
    #
    # Spectral smoothness (affects Gaussian process part)
    # 3 = relatively high variance of spectral curbvature 
    # 0.4 = quefrency mode below which cepstrum flattens
    #
    # power law part of spectrum:
    # -5= preferred power-law slope
    # 0.5 = low variance of power-law slope
    #
    # Gaussian process part of log-spectrum
    # 0.4 = y-intercept mean of additional power 
    # 0.3 = y-intercept variance of additional power
    
    # Building the model for a correlated signal
    harmonic_space = position_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(harmonic_space, position_space)
    power_space = A.target[0]
    power_distributor = ift.PowerDistributor(harmonic_space, power_space)

    vol = harmonic_space.scalar_dvol
    vol = ift.ScalingOperator(vol**(-0.5), harmonic_space)
    correlated_field = ht(
        vol(power_distributor(A))*ift.ducktape(harmonic_space, None, 'xi'))
    # alternatively to the block above one can do:
    #correlated_field = ift.CorrelatedField(position_space, A)

    # apply some nonlinearity
    signal = ift.positive_tanh(correlated_field)

    # Building the Line of Sight response
    LOS_starts, LOS_ends = get_random_LOS(100)
    R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    # build signal response model and model likelihood
    signal_response = R(signal)
    # specify noise
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(noise, data_space)

    # generate mock signal and data
    MOCK_POSITION = ift.from_random('normal', signal_response.domain)
    data = signal_response(MOCK_POSITION) + N.draw_sample()

    # set up model likelihood
    likelihood = ift.GaussianEnergy(mean=data, covariance=N)(signal_response)

    # set up minimization and inversion schemes
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(
        name='Newton', tol=1e-7, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)

    # build model Hamiltonian
    H = ift.Hamiltonian(likelihood, ic_sampling)

    INITIAL_POSITION = ift.MultiField.full(H.domain, 0.)
    position = INITIAL_POSITION

    plot = ift.Plot()
    plot.add(signal(MOCK_POSITION), title='Ground Truth')
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([A.force(MOCK_POSITION)], title='Power Spectrum')
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name="setup.png")

    # number of samples used to estimate the KL
    N_samples = 20
    
    # five intermediate steps in minimization to illustrate progress
    for i in range(5):
        # set up KL
        KL = ift.KL_Energy(position, H, N_samples)
        # minimize KL until iteration limit reached
        KL, convergence = minimizer(KL)
        # read out position
        position = KL.position
        # plot momentariy field
        plot = ift.Plot()
        plot.add(signal(KL.position), title="reconstruction")
        plot.add([A.force(KL.position), A.force(MOCK_POSITION)], title="power")
        plot.output(ny=1, ysize=6, xsize=16, name="loop-{:02}.png".format(i))

    # final plot 
    KL = ift.KL_Energy(position, H, N_samples)
    plot = ift.Plot()
    # do statistics
    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))
    plot.add(sc.mean, title="Posterior Mean")
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    powers = [A.force(s + KL.position) for s in KL.samples]
    plot.add(
        powers + [A.force(KL.position), A.force(MOCK_POSITION)],
        title="Sampled Posterior Power Spectrum",
        linewidth=[1.]*len(powers) + [3., 3.])
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name="results.png")
