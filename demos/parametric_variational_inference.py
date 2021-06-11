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
# Meanfield and fullcovariance variational inference
#
# The signal is a 1-D lognormal distributed field.
# The  data follows a Poisson likelihood.
# The posterior distribution is approximated with a diagonal, as well as a
# full covariance Gaussian distribution. This is achieved by minimizing
# a stochastic estimate of the KL-Divergence
#
# Note that the fullcovariance approximation scales quadratically with the
# number of parameters. 
###############################################################################

import numpy as np
from matplotlib import pyplot as plt

import nifty8 as ift

ift.random.push_sseq_from_seed(27)


if __name__ == "__main__":
    # Space and model setup
    position_space = ift.RGSpace([100])
    harmonic_space = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(harmonic_space, position_space)
    p_space = ift.PowerSpace(harmonic_space)

    pd = ift.PowerDistributor(harmonic_space, p_space)
    a = ift.PS_field(p_space, lambda k: 1.0 / (1.0 + k ** 2))
    A = pd(a)
    sky = 10 * ift.exp(HT(ift.makeOp(A))).ducktape("xi")
    R = ift.GeometryRemover(position_space)

    mask = np.zeros(position_space.shape)
    mask[mask.shape[0]//3:2*mask.shape[0]//3] = 1
    mask = ift.Field.from_raw(position_space, mask)
    R = ift.MaskOperator(mask)

    d_space = R.target[0]
    lamb = R(sky)

    # Generate simulated signal and data and build likelihood energy
    mock_position = ift.from_random(sky.domain, "normal")
    data = ift.random.current_rng().poisson(lamb(mock_position).val)
    data = ift.makeField(d_space, data)
    likelihood_energy = ift.PoissonianEnergy(data) @ lamb
    H = ift.StandardHamiltonian(likelihood_energy)

    # Settings for minimization
    IC = ift.StochasticAbsDeltaEnergyController(5, iteration_limit=200,
                                                name='advi')
    minimizer_fc = ift.ADVIOptimizer(IC, eta=0.1)
    minimizer_mf = ift.ADVIOptimizer(IC)

    # Initial positions 
    position_fc = ift.from_random(H.domain)*0.1
    position_mf = ift.from_random(H.domain)*0.1

    # Setup of the variational models
    fc = ift.FullCovarianceVI(position_fc, H, 3, True, initial_sig=0.01)
    mf = ift.MeanFieldVI(position_mf, H, 3, True, initial_sig=0.01)


    niter = 10
    for ii in range(niter):
        # Plotting
        plt.plot(sky(fc.mean).val, "b-", label="Full covariance")
        plt.plot(sky(mf.mean).val, "r-", label="Mean field")
        for _ in range(5):
            plt.plot(sky(fc.draw_sample()).val, "b-", alpha=0.3)
            plt.plot(sky(mf.draw_sample()).val, "r-", alpha=0.3)
        plt.plot(R.adjoint(data).val, "kx")
        plt.plot(sky(mock_position).val, "k-", label="Ground truth")
        plt.legend()
        plt.ylim(0.1, data.val.max() + 10)
        fname = f"meanfield_{ii:03d}.png"
        plt.savefig(fname)
        print(f"Saved results as '{fname}' ({ii}/{niter-1}).")
        plt.close()
        # /Plotting

        # Run minimization
        fc.minimize(minimizer_fc)
        mf.minimize(minimizer_mf)
