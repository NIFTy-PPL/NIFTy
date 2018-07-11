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
    A, amplitude_internals = ift.make_amplitude_model(
        position_space, 16, 1, 10, -4., 1, 0., 1.)

    # Building the model for a correlated signal
    harmonic_space = position_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(harmonic_space, position_space)
    power_space = A.value.domain[0]
    power_distributor = ift.PowerDistributor(harmonic_space, power_space)
    position = ift.MultiField.from_dict(
        {'xi': ift.Field.from_random('normal', harmonic_space)})

    xi = ift.Variable(position)['xi']
    Amp = power_distributor(A)
    correlated_field_h = Amp * xi
    correlated_field = ht(correlated_field_h)
    # alternatively to the block above one can do:
    # correlated_field,_ = ift.make_correlated_field(position_space, A)

    # apply some nonlinearity
    signal = ift.PointwisePositiveTanh(correlated_field)

    # Building the Line of Sight response
    LOS_starts, LOS_ends = get_random_LOS(100)
    R = ift.LOSResponse(position_space, starts=LOS_starts,
                        ends=LOS_ends)
    # build signal response model and model likelihood
    signal_response = R(signal)
    # specify noise
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(noise, data_space)

    # generate mock data
    MOCK_POSITION = ift.from_random('normal', signal.position.domain)
    data = signal_response.at(MOCK_POSITION).value + N.draw_sample()

    # set up model likelihood
    likelihood = ift.GaussianEnergy(signal_response, mean=data, covariance=N)

    # set up minimization and inversion schemes
    ic_cg = ift.GradientNormController(iteration_limit=10)
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradientNormController(name='Newton', iteration_limit=100)
    minimizer = ift.RelaxedNewton(ic_newton)

    # build model Hamiltonian
    H = ift.Hamiltonian(likelihood, ic_sampling)

    INITIAL_POSITION = ift.from_random('normal', H.position.domain)
    position = INITIAL_POSITION

    ift.plot(signal.at(MOCK_POSITION).value, name='truth.png')
    ift.plot(R.adjoint_times(data), name='data.png')
    ift.plot([A.at(MOCK_POSITION).value], name='power.png')

    # number of samples used to estimate the KL
    N_samples = 20
    for i in range(5):
        H = H.at(position)
        samples = [H.metric.draw_sample(from_inverse=True)
                   for _ in range(N_samples)]

        KL = ift.SampledKullbachLeiblerDivergence(H, samples)
        KL = KL.make_invertible(ic_cg)
        KL, convergence = minimizer(KL)
        position = KL.position

        ift.plot(signal.at(position).value, name='reconstruction.png')

        ift.plot([A.at(position).value, A.at(MOCK_POSITION).value],
                 name='power.png')

    sc = ift.StatCalculator()
    for sample in samples:
        sc.add(signal.at(sample+position).value)
    ift.plot(sc.mean, name='avrg.png')
    ift.plot(ift.sqrt(sc.var), name='std.png')

    powers = [A.at(s+position).value for s in samples]
    ift.plot([A.at(position).value, A.at(MOCK_POSITION).value]+powers,
             name='power.png')
