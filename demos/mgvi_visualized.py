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
    icnewton = ift.GradientNormController(iteration_limit=1, name='Mini')
    ham = ift.StandardHamiltonian(lh, icsamp)
    newton = ift.SteepestDescent(icnewton)
    pos = ift.from_random('normal', ham.domain)

    x_limits = [-2/uninformative_scaling, 2/uninformative_scaling]
    y_limits = [-4, 2]
    x = np.linspace(*x_limits, num=101)
    y = np.linspace(*y_limits, num=101)
    z = np.empty((x.size, y.size))
    for ii, xx in enumerate(x):
        for jj, yy in enumerate(y):
            inp = ift.MultiField.from_raw(lh.domain, {'a': xx, 'b': yy})
            z[ii, jj] = np.exp(-1*ham(inp).val)

    for ii in range(10):
        if ii % 2 == 0:
            mgkl = ift.MetricGaussianKL(pos, ham, 40)

        plt.cla()
        plt.contour(x*uninformative_scaling, y, z.T)
        xs, ys = [], []
        for samp in mgkl.samples:
            samp = (samp + pos).val
            xs.append(samp['a'])
            ys.append(samp['b'])
        plt.scatter(np.array(xs)*uninformative_scaling, np.array(ys))
        plt.scatter(pos.val['a'], pos.val['b'], color='red')
        plt.draw()
        plt.pause(0.01)

        mgkl, _ = newton(mgkl)
        pos = mgkl.position
    ift.logger.info('Finished')
    # plt.show()
