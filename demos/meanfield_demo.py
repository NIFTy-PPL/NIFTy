import nifty7 as ift
from matplotlib import pyplot as plt
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

###############################################################################
# Log-normal field reconstruction from Poissonian data with inhomogenous
# exposure (in case for 2D mode)
# 1D (set mode=0), 2D (mode=1), or on the sphere (mode=2)
###############################################################################

import sys

import numpy as np


# def exposure_2d(domain):
#     # Structured exposure for 2D mode
#     x_shape, y_shape = domain.shape
#     exposure = np.ones(domain.shape)
#     exposure[x_shape//3:x_shape//2, :] *= 2.
#     exposure[x_shape*4//5:x_shape, :] *= .1
#     exposure[x_shape//2:x_shape*3//2, :] *= 3.
#     exposure[:, x_shape//3:x_shape//2] *= 2.
#     exposure[:, x_shape*4//5:x_shape] *= .1
#     exposure[:, x_shape//2:x_shape*3//2] *= 3.
#     return ift.Field.from_raw(domain, exposure)


if __name__ == '__main__':

    # Two-dimensional regular grid with inhomogeneous exposure
    position_space = ift.RGSpace([100])

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
    sky = ift.exp(HT(ift.makeOp(A))).ducktape('xi')

    # M = ift.DiagonalOperator(exposure)
    GR = ift.GeometryRemover(position_space)
    # Define instrumental response
    # R = GR(M)
    R = GR

    # Generate mock data and define likelihood operator
    d_space = R.target[0]
    lamb = R(sky)
    mock_position = ift.from_random(sky.domain, 'normal')
    data = lamb(mock_position)
    data = ift.random.current_rng().poisson(data.val.astype(np.float64))
    data = ift.Field.from_raw(d_space, data)
    likelihood = ift.PoissonianEnergy(data) @ lamb

    # Settings for minimization
    ic_newton = ift.DeltaEnergyController(
        name='Newton', iteration_limit=1, tol_rel_deltaE=1e-8)
    # minimizer = ift.L_BFGS(ic_newton)
    minimizer = ift.ADVIOptimizer(steps=10)

    # Compute MAP solution by minimizing the information Hamiltonian
    H = ift.StandardHamiltonian(likelihood)
    initial_position = ift.from_random(domain, 'normal')

    # meanfield_model = ift.MeanfieldModel(H.domain)
    fullcov_model = ift.FullCovarianceModel(H.domain)
    initial_position = fullcov_model.get_initial_pos()
    position = initial_position
    KL = ift.ParametricGaussianKL.make(initial_position,H,fullcov_model,3,False)
    plt.figure('data')
    plt.imshow(sky(mock_position).val)
    plt.pause(0.001)
    for i in range(300):
        # KL = ParametricGaussianKL.make(position,H,meanfield_model,3,True)
        KL, _ = minimizer(KL)
        position = KL.position
        plt.figure('result')
        plt.cla()
        plt.plot(sky(fullcov_model.generator(KL.position)).val)
        for samp in KL.samples:
            plt.plot(sky(fullcov_model.generator(KL.position + samp)).val)
        plt.plot(data.val,'kx')
        plt.pause(0.001)