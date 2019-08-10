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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

import nifty5 as ift


def test_plots():
    # FIXME Write to temporary folder?
    rg_space1 = ift.makeDomain(ift.RGSpace((100,)))
    rg_space2 = ift.makeDomain(ift.RGSpace((80, 60), distances=1))
    hp_space = ift.makeDomain(ift.HPSpace(64))
    gl_space = ift.makeDomain(ift.GLSpace(128))

    fft = ift.FFTOperator(rg_space2)

    field_rg1_1 = ift.Field.from_global_data(rg_space1, np.random.randn(100))
    field_rg1_2 = ift.Field.from_global_data(rg_space1, np.random.randn(100))
    field_rg2 = ift.Field.from_global_data(
        rg_space2, np.random.randn(80*60).reshape((80, 60)))
    field_hp = ift.Field.from_global_data(hp_space, np.random.randn(12*64**2))
    field_gl = ift.Field.from_global_data(gl_space, np.random.randn(32640))
    field_ps = ift.power_analyze(fft.times(field_rg2))

    plot = ift.Plot()
    plot.add(field_rg1_1, title='Single plot')
    plot.output(name='plot1.png')

    plot = ift.Plot()
    plot.add(field_rg2, title='2d rg')
    plot.add([field_rg1_1, field_rg1_2], title='list 1d rg', label=['1', '2'])
    plot.add(field_rg1_2, title='1d rg, xmin, ymin', xmin=0.5, ymin=0.,
             xlabel='xmin=0.5', ylabel='ymin=0')
    plot.output(title='Three plots', name='plot2.png')

    plot = ift.Plot()
    plot.add(field_hp, title='HP planck-color', colormap='Planck-like')
    plot.add(field_rg1_2, title='1d rg')
    plot.add(field_ps)
    plot.add(field_gl, title='GL')
    plot.add(field_rg2, title='2d rg')
    plot.output(nx=2, ny=3, title='Five plots', name='plot3.png')
