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
# Authors: Reimar Leike, Philipp Arras, Philipp Frank
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

###############################################################################
# Variational Inference (VI)
#
# This script demonstrates how MGVI and GeoVI work for an inference problem
# with only two real quantities of interest. This enables us to plot the
# posterior probability density as two-dimensional plot. The approximate
# posterior samples are contrasted with the maximum-a-posterior (MAP) solution
# together with samples drawn with the Laplace method. This method uses the
# local curvature at the MAP solution as inverse covariance of a Gaussian
# probability density.
###############################################################################

import numpy as np
import pylab as plt
from matplotlib.colors import LogNorm

import nifty7 as ift


def main():
    dom = ift.UnstructuredDomain(1)
    scale = 10

    a = ift.FieldAdapter(dom, 'a')
    b = ift.FieldAdapter(dom, 'b')
    lh = (a.adjoint @ a).scale(scale) + (b.adjoint @ b).scale(-1.35*2).exp()
    lh = ift.VariableCovarianceGaussianEnergy(dom, 'a', 'b', np.float64) @ lh
    icsamp = ift.AbsDeltaEnergyController(deltaE=0.1, iteration_limit=2)
    ham = ift.StandardHamiltonian(lh, icsamp)

    x_limits = [-8/scale, 8/scale]
    x_limits_scaled = [-8, 8]
    y_limits = [-4, 4]
    x = np.linspace(*x_limits, num=401)
    y = np.linspace(*y_limits, num=401)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    def np_ham(x, y):
        prior = x**2 + y**2
        mean = x*scale
        lcov = 1.35*2*y
        lh = .5*(mean**2*np.exp(-lcov) + lcov)
        return lh + prior

    z = np.exp(-1*np_ham(xx, yy))
    plt.plot(y, np.sum(z, axis=0))
    plt.xlabel('y')
    plt.ylabel('unnormalized pdf')
    plt.title('Marginal density')
    plt.pause(2.0)
    plt.close()
    plt.plot(x*scale, np.sum(z, axis=1))
    plt.xlabel('x')
    plt.ylabel('unnormalized pdf')
    plt.title('Marginal density')
    plt.pause(2.0)
    plt.close()

    pos = ift.from_random(ham.domain, 'normal')
    MAP = ift.EnergyAdapter(pos, ham, want_metric=True)
    minimizer = ift.NewtonCG(
        ift.GradientNormController(iteration_limit=20, name='Mini'))
    MAP, _ = minimizer(MAP)
    map_xs, map_ys = [], []
    for ii in range(10):
        samp = (MAP.metric.draw_sample(from_inverse=True) + MAP.position).val
        map_xs.append(samp['a'])
        map_ys.append(samp['b'])

    minimizer = ift.NewtonCG(
        ift.GradientNormController(iteration_limit=2, name='Mini'))
    pos = pos1 = ift.from_random(ham.domain, 'normal')
    fig, axs = plt.subplots(2, 1, figsize=[12, 8])
    for ii in range(15):
        if ii % 3 == 0:
            # Resample
            mgkl = ift.MetricGaussianKL(pos, ham, 100, False)
            mini_samp = ift.NewtonCG(ift.GradientNormController(iteration_limit=5))
            geokl = ift.GeoMetricKL(pos1, ham, 100, mini_samp, False)

        for axx in axs:
            axx.clear()
            im = axx.imshow(z.T, origin='lower', norm=LogNorm(vmin=1e-3, vmax=np.max(z)),
                               cmap='gist_earth_r', extent=x_limits_scaled + y_limits)
            if ii == 0:
                cbar = plt.colorbar(im, ax=axx)
                cbar.ax.set_ylabel('pdf')

        for jj, nn, kl, pp in ((0, "MGVI", mgkl, pos), (1, "GeoVI", geokl, pos1)):
            xs, ys = [], []
            for samp in kl.samples:
                samp = (samp + pp).val
                xs.append(samp['a'])
                ys.append(samp['b'])
            axs[jj].scatter(np.array(xs)*scale, np.array(ys), label=f'{nn} samples')
            axs[jj].scatter(pp.val['a']*scale, pp.val['b'], label=f'{nn} latent mean')
            axs[jj].set_title(nn)

        for axx in axs:
            axx.scatter(np.array(map_xs)*scale, np.array(map_ys),
                        label='Laplace samples')
            axx.scatter(MAP.position.val['a']*scale, MAP.position.val['b'],
                        label='Maximum a posterior solution')
            axx.set_xlim(x_limits_scaled)
            axx.set_ylim(y_limits)
            axx.set_ylabel('y')
            axx.legend(loc='lower right')
        axs[0].xaxis.set_visible(False)
        axs[1].set_xlabel('x')
        plt.tight_layout()
        plt.draw()
        plt.pause(1.0)

        mgkl, _ = minimizer(mgkl)
        geokl, _ = minimizer(geokl)
        pos = mgkl.position
        pos1 = geokl.position
    ift.logger.info('Finished')
    # Uncomment the following line in order to leave the plots open
    # plt.show()


if __name__ == '__main__':
    main()
