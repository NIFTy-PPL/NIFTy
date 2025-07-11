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

#####################################################################
# Bernoulli reconstruction
# Reconstruct an event probability field with values between 0 and 1
# from the observed events
# 1D (set mode=0), 2D (mode=1), or on the sphere (mode=2)
#####################################################################

import numpy as np

import nifty8 as ift


def main():
    # Set up the position space of the signal
    mode = 2
    if mode == 0:
        # One-dimensional regular grid
        position_space = ift.RGSpace(1024)
    elif mode == 1:
        # Two-dimensional regular grid
        position_space = ift.RGSpace([512, 512])
    else:
        # Sphere
        position_space = ift.HPSpace(128)

    # Define harmonic space and transform
    harmonic_space = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(harmonic_space, position_space)

    position = ift.from_random(harmonic_space, 'normal')

    # Define power spectrum and amplitudes
    def sqrtpspec(k):
        return 1./(20. + k**2)

    A = ift.create_power_operator(harmonic_space, sqrtpspec)

    # Set up a sky operator and instrumental response
    sky = ift.sigmoid(HT(A))
    GR = ift.GeometryRemover(position_space)
    R = GR

    # Generate mock data
    p = R(sky)
    mock_position = ift.from_random(harmonic_space, 'normal')
    tmp = p(mock_position).val.astype(np.float64)
    data = ift.random.current_rng().binomial(1, tmp)
    data = ift.Field.from_raw(R.target, data)

    # Compute likelihood energy and Hamiltonian
    position = ift.from_random(harmonic_space, 'normal')
    likelihood_energy = ift.BernoulliEnergy(data) @ p
    ic_newton = ift.DeltaEnergyController(
        name='Newton', iteration_limit=100, tol_rel_deltaE=1e-8)
    minimizer = ift.NewtonCG(ic_newton)
    ic_sampling = ift.GradientNormController(iteration_limit=100)

    # Minimize the Hamiltonian
    H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)
    H = ift.EnergyAdapter(position, H, want_metric=True)
    # minimizer = ift.L_BFGS(ic_newton)
    H, convergence = minimizer(H)

    reconstruction = sky(H.position)

    plot = ift.Plot()
    plot.add(reconstruction, title='reconstruction')
    plot.add(GR.adjoint_times(data), title='data')
    plot.add(sky(mock_position), title='truth')
    plot.output(nx=3, xsize=16, ysize=9, title="results", name="bernoulli.png")


if __name__ == '__main__':
    main()
