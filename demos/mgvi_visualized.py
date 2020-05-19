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
# Authors: Reimar Leike, Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pylab as plt

import nifty6 as ift

if __name__ == '__main__':
    dom = ift.UnstructuredDomain(1)
    a = ift.FieldAdapter(dom, 'a')
    b = ift.FieldAdapter(dom, 'b')

    uninformative_scaling = 10
    lh = (a.adjoint @ a).scale(uninformative_scaling) + (b.adjoint @ b).scale(-1.35*2).exp()
    lh = ift.VariableCovarianceGaussianEnergy(dom, 'a', 'b', np.float64) @ lh
    icsamp = ift.AbsDeltaEnergyController(deltaE=0.1, iteration_limit=2)
    ham = ift.StandardHamiltonian(lh, icsamp)

    x_limits = [-8/uninformative_scaling, 8/uninformative_scaling]
    y_limits = [-4, 4]
    x = np.linspace(*x_limits, num=201)
    y = np.linspace(*y_limits, num=201)
    z = np.empty((x.size, y.size))
    for ii, xx in enumerate(x):
        for jj, yy in enumerate(y):
            inp = ift.MultiField.from_raw(lh.domain, {'a': xx, 'b': yy})
            z[ii, jj] = np.exp(-1*ham(inp).val)
    plt.plot(y, np.sum(z, axis=0))
    plt.xlabel('y')
    plt.ylabel('pdf')
    plt.title('marginal density')
    plt.show()
    plt.plot(x*uninformative_scaling, np.sum(z, axis=1))
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.title('marginal density')
    plt.show()

    pos = ift.from_random('normal', ham.domain)
    MAP = ift.EnergyAdapter(pos, ham, want_metric=True)
    minimizer = ift.NewtonCG(ift.GradientNormController(iteration_limit=20,
                                                               name='Mini'))
    MAP, _ = minimizer(MAP)
    map_xs, map_ys = [], []
    for ii in range(10):
        samp = (MAP.metric.draw_sample(from_inverse=True) + MAP.position).val
        map_xs.append(samp['a'])
        map_ys.append(samp['b'])

    minimizer = ift.NewtonCG(
            ift.GradientNormController(iteration_limit=2, name='Mini'))
    pos = ift.from_random('normal', ham.domain)
    for ii in range(15):
        if ii % 3 == 0:
            mgkl = ift.MetricGaussianKL(pos, ham, 40)

        plt.cla()
        plt.imshow(z.T, origin='lower', 
                extent=(x_limits[0]*uninformative_scaling,
                    x_limits[1]*uninformative_scaling)+tuple(y_limits), vmin=0., vmax=np.amax(z))
        if ii==0:
            plt.colorbar()
        xs, ys = [], []
        for samp in mgkl.samples:
            samp = (samp + pos).val
            xs.append(samp['a'])
            ys.append(samp['b'])
        plt.scatter(np.array(xs)*uninformative_scaling, np.array(ys))
        plt.scatter(pos.val['a']*uninformative_scaling, pos.val['b'])
        plt.scatter(np.array(map_xs)*uninformative_scaling, np.array(map_ys))
        plt.scatter(MAP.position.val['a']*uninformative_scaling,
                    MAP.position.val['b'])
        plt.draw()
        plt.pause(1.0)

        mgkl, _ = minimizer(mgkl)
        pos = mgkl.position
    ift.logger.info('Finished')
    # Uncomment the following line in order to leave the plots open
    # plt.show()
