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

###############################################################################
# Log-normal field reconstruction from Poissonian data with inhomogenous
# exposure (in case for 2D mode)
# 1D (set mode=0), 2D (mode=1), or on the sphere (mode=2)
###############################################################################

import sys

import numpy as np

import nifty8 as ift


def exposure_2d(domain):
    # Structured exposure for 2D mode
    x_shape, y_shape = domain.shape
    exposure = np.ones(domain.shape)
    exposure[x_shape//3:x_shape//2, :] *= 2.
    exposure[x_shape*4//5:x_shape, :] *= .1
    exposure[x_shape//2:x_shape*3//2, :] *= 3.
    exposure[:, x_shape//3:x_shape//2] *= 2.
    exposure[:, x_shape*4//5:x_shape] *= .1
    exposure[:, x_shape//2:x_shape*3//2] *= 3.
    return ift.Field.from_raw(domain, exposure)


def main():
    # Choose space on which the signal field is defined
    if len(sys.argv) == 2:
        mode = int(sys.argv[1])
    else:
        mode = 1

    if mode == 0:
        # One-dimensional regular grid with uniform exposure of 10
        position_space = ift.RGSpace(1024)
        exposure = ift.Field.full(position_space, 10.)
    elif mode == 1:
        # Two-dimensional regular grid with inhomogeneous exposure
        position_space = ift.RGSpace([512, 512])
        exposure = exposure_2d(position_space)
    else:
        # Sphere with uniform exposure of 100
        position_space = ift.HPSpace(128)
        exposure = ift.Field.full(position_space, 100.)

    # Define harmonic space and harmonic transform
    harmonic_space = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(harmonic_space, position_space)

    # Domain on which the field's degrees of freedom are defined
    domain = ift.DomainTuple.make(harmonic_space)

    # Define amplitude (square root of power spectrum)
    def sqrtpspec(k):
        return 1./(20. + k**2)

    p_space = ift.PowerSpace(harmonic_space)
    pd = ift.PowerDistributor(harmonic_space, p_space)
    a = ift.PS_field(p_space, sqrtpspec)
    A = pd(a)

    # Define sky operator
    sky = ift.exp(HT(ift.makeOp(A)))

    M = ift.DiagonalOperator(exposure)
    GR = ift.GeometryRemover(position_space)
    # Define instrumental response
    R = GR(M)

    # Generate mock data and define likelihood energy operator
    d_space = R.target[0]
    lamb = R(sky)
    mock_position = ift.from_random(domain, 'normal')
    data = lamb(mock_position)
    data = ift.random.current_rng().poisson(data.val.astype(np.float64))
    data = ift.Field.from_raw(d_space, data)
    likelihood_energy = ift.PoissonianEnergy(data) @ lamb

    # Settings for minimization
    ic_newton = ift.DeltaEnergyController(
        name='Newton', iteration_limit=100, tol_rel_deltaE=1e-8)
    minimizer = ift.NewtonCG(ic_newton)

    # Compute MAP solution by minimizing the information Hamiltonian
    sl = ift.optimize_kl(likelihood_energy, 1, 0, minimizer, None, None,
                         output_directory="getting_started_2_results", overwrite=True,
                         plottable_operators={"signal": sky})


if __name__ == '__main__':
    main()
