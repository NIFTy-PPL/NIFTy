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
# This script demonstrates how MGVI, GeoVI, MeanfieldVI and FullCovarianceVI
# work for an inference problem with only two real quantities of interest. This
# enables us to plot the posterior probability density as two-dimensional plot.
###############################################################################

import numpy as np
import pylab as plt
from matplotlib.colors import LogNorm

import nifty8 as ift


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
    plt.pause(2.)
    plt.close()
    plt.plot(x*scale, np.sum(z, axis=1))
    plt.xlabel('x')
    plt.ylabel('unnormalized pdf')
    plt.title('Marginal density')
    plt.pause(2.)
    plt.close()

    mapx = xx[z == np.max(z)]
    mapy = yy[z == np.max(z)]
    meanx = (xx*z).sum()/z.sum()
    meany = (yy*z).sum()/z.sum()

    n_samples = 100
    minimizer = ift.NewtonCG(
        ift.GradientNormController(iteration_limit=3, name='Mini'))
    IC = ift.StochasticAbsDeltaEnergyController(0.5, iteration_limit=20,
                                                name='advi')
    stochastic_minimizer_mf = ift.ADVIOptimizer(IC, eta=0.3)
    stochastic_minimizer_fc = ift.ADVIOptimizer(IC, eta=0.3)
    posmg = posgeo = posmf = posfc = ift.from_random(ham.domain, 'normal')
    fc = ift.FullCovarianceVI(posfc, ham, 10, False, initial_sig=0.01)
    mf = ift.MeanFieldVI(posmf, ham, 10, False, initial_sig=0.01)

    fig, axs = plt.subplots(2, 2, figsize=[12, 8])
    axs = axs.flatten()

    def update_plot(runs):
        for axx, (nn, kl) in zip(axs, runs):
            axx.clear()
            axx.imshow(z.T, origin='lower',  cmap='gist_earth_r',
                       norm=LogNorm(vmin=1e-3, vmax=np.max(z)),
                       extent=x_limits_scaled + y_limits)
            xs, ys = [], []
            if isinstance(kl, ift.SampledKLEnergyClass):
                samples = kl.samples.iterator()
            else:
                samples = (kl.draw_sample() for _ in range(n_samples))
            mx, my = 0., 0.
            for samp in samples:
                a = samp.val['a']
                xs.append(a)
                mx += a
                b = samp.val['b']
                ys.append(b)
                my += b
            mx /= n_samples
            my /= n_samples
            axx.scatter(np.array(xs)*scale, np.array(ys),
                        label=f'{nn} samples')
            axx.scatter(mx*scale, my, label=f'{nn} mean')
            axx.scatter(mapx*scale, mapy, label='MAP')
            axx.scatter(meanx*scale, meany, label='Posterior mean')
            axx.set_title(nn)
            axx.set_xlim(x_limits_scaled)
            axx.set_ylim(y_limits)
            axx.legend(loc='lower right')
        axs[0].xaxis.set_visible(False)
        axs[1].xaxis.set_visible(False)
        axs[1].yaxis.set_visible(False)
        axs[2].set_xlabel('x')
        axs[2].set_ylabel('y')
        axs[3].yaxis.set_visible(False)
        axs[3].set_xlabel('x')
        plt.tight_layout()
        plt.draw()
        plt.pause(2.)

    for ii in range(10):
        if ii % 2 == 0:
            # Resample GeoVI and MGVI
            mgkl = ift.SampledKLEnergy(posmg, ham, n_samples, None, False)
            mini_samp = ift.NewtonCG(
                    ift.AbsDeltaEnergyController(1E-8, iteration_limit=5))
            geokl = ift.SampledKLEnergy(posgeo, ham, n_samples, mini_samp, False)

            runs = (("MGVI", mgkl), ("GeoVI", geokl),
                    ("MeanfieldVI", mf), ("FullCovarianceVI", fc))
            update_plot(runs)

        mgkl, _ = minimizer(mgkl)
        geokl, _ = minimizer(geokl)
        mf.minimize(stochastic_minimizer_mf)
        fc.minimize(stochastic_minimizer_fc)
        posmg = mgkl.position
        posgeo = geokl.position
        runs = (("MGVI", mgkl), ("GeoVI", geokl),
                ("MeanfieldVI", mf), ("FullCovarianceVI", fc))
        update_plot(runs)
    ift.logger.info('Finished')
    # Uncomment the following line in order to leave the plots open
    # plt.show()


if __name__ == '__main__':
    main()
