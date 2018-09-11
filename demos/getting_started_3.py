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
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import nifty5 as ift
import numpy as np


def get_random_LOS(n_los):
    starts = list(np.random.uniform(0, 1, (n_los, 2)).T)
    ends = list(np.random.uniform(0, 1, (n_los, 2)).T)
    return starts, ends


if __name__ == '__main__':
    # FIXME description of the tutorial
    np.random.seed(42)
    position_space = ift.RGSpace([128, 128])

    # Setting up an amplitude model
    A = ift.AmplitudeModel(position_space, 16, 1, 10, -4., 1, 0., 1.)
    dummy = ift.from_random('normal', A.domain)

    # Building the model for a correlated signal
    harmonic_space = position_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(harmonic_space, position_space)
    power_space = A.target[0]
    power_distributor = ift.PowerDistributor(harmonic_space, power_space)
    dummy = ift.Field.from_random('normal', harmonic_space)
    domain = ift.MultiDomain.union((A.domain,
                                    ift.MultiDomain.make({
                                        'xi': harmonic_space
                                    })))

    correlated_field = ht(power_distributor(A)*ift.FieldAdapter(domain, "xi"))
    # alternatively to the block above one can do:
    # correlated_field = ift.CorrelatedField(position_space, A)

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

    # generate mock data
    MOCK_POSITION = ift.from_random('normal', domain)
    data = signal_response(MOCK_POSITION) + N.draw_sample()

    # set up model likelihood
    likelihood = ift.GaussianEnergy(mean=data, covariance=N)(signal_response)

    # set up minimization and inversion schemes
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(
        name='Newton', tol=1e-7, iteration_limit=1000)
    minimizer = ift.NewtonCG(ic_newton)

    # build model Hamiltonian
    H = ift.Hamiltonian(likelihood, ic_sampling)

    INITIAL_POSITION = ift.from_random('normal', domain)
    position = INITIAL_POSITION

    plot = ift.Plot()
    plot.add(signal(MOCK_POSITION), title='Ground Truth')
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([A(MOCK_POSITION.extract(A.domain))], title='Power Spectrum')
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name="setup.png")

    # number of samples used to estimate the KL
    N_samples = 20
    for i in range(2):
        KL = ift.KL_Energy(position, H, N_samples)
        KL, convergence = minimizer(KL)
        position = KL.position

        plot = ift.Plot()
        plot.add(signal(KL.position), title="reconstruction")
        plot.add(
            [
                A(KL.position.extract(A.domain)),
                A(MOCK_POSITION.extract(A.domain))
            ],
            title="power")
        plot.output(ny=1, ysize=6, xsize=16, name="loop.png")

    plot = ift.Plot()
    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))
    plot.add(sc.mean, title="Posterior Mean")
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    powers = [A((s + KL.position).extract(A.domain)) for s in KL.samples]
    plot.add(
        [A(KL.position.extract(A.domain)),
         A(MOCK_POSITION.extract(A.domain))] + powers,
        title="Sampled Posterior Power Spectrum")
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name="results.png")
