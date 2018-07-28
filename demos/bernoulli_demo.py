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


if __name__ == '__main__':
    # FIXME ABOUT THIS CODE
    np.random.seed(41)

    # Set up the position space of the signal
    #
    # # One dimensional regular grid with uniform exposure
    # position_space = ift.RGSpace(1024)
    # exposure = np.ones(position_space.shape)

    # Two-dimensional regular grid with inhomogeneous exposure
    position_space = ift.RGSpace([512, 512])

    # # Sphere with with uniform exposure
    # position_space = ift.HPSpace(128)
    # exposure = ift.Field.full(position_space, 1.)

    # Defining harmonic space and transform
    harmonic_space = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(harmonic_space, position_space)

    position = ift.from_random('normal', harmonic_space)

    # Define power spectrum and amplitudes
    def sqrtpspec(k):
        return 1. / (20. + k**2)

    p_space = ift.PowerSpace(harmonic_space)
    pd = ift.PowerDistributor(harmonic_space, p_space)
    a = ift.PS_field(p_space, sqrtpspec)
    A = pd(a)

    # Set up a sky model
    sky = lambda inp: HT(A*inp).positive_tanh()

    GR = ift.GeometryRemover(position_space)
    # Set up instrumental response
    R = GR

    # Generate mock data
    d_space = R.target[0]
    p = R.chain(sky)
    mock_position = ift.from_random('normal', harmonic_space)
    pp = p(mock_position)
    data = np.random.binomial(1, pp.to_global_data().astype(np.float64))
    data = ift.Field.from_global_data(d_space, data)

    # Compute likelihood and Hamiltonian
    position = ift.from_random('normal', harmonic_space)
    likelihood = ift.BernoulliEnergy(p, data)
    ic_cg = ift.GradientNormController(iteration_limit=50)
    ic_newton = ift.GradientNormController(name='Newton', iteration_limit=30,
                                           tol_abs_gradnorm=1e-3)
    minimizer = ift.RelaxedNewton(ic_newton)
    ic_sampling = ift.GradientNormController(iteration_limit=100)

    # Minimize the Hamiltonian
    H = ift.Hamiltonian(likelihood, ic_sampling)
    H = ift.EnergyAdapter(position, H)
    H = H.make_invertible(ic_cg)
    # minimizer = ift.SteepestDescent(ic_newton)
    H, convergence = minimizer(H)

    reconstruction = sky(H.position)

    ift.plot(reconstruction, title='reconstruction')
    ift.plot(GR.adjoint_times(data), title='data')
    ift.plot(sky(mock_position), title='truth')
    ift.plot_finish(nx=3, xsize=16, ysize=5, title="results",
                    name="bernoulli.png")
