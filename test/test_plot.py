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

import nifty6 as ift
from .common import setup_function, teardown_function

nr = 0
def name():
    global nr
    nr += 1
    return 'plot{}.png'.format(nr)

def test_plots():
    # FIXME Write to temporary folder?
    rg_space1 = ift.makeDomain(ift.RGSpace((100,)))
    rg_space2 = ift.makeDomain(ift.RGSpace((80, 60), distances=1))
    hp_space = ift.makeDomain(ift.HPSpace(64))
    gl_space = ift.makeDomain(ift.GLSpace(128))

    fft = ift.FFTOperator(rg_space2)

    field_rg1_1 = ift.Field(rg_space1, ift.random.current_rng().standard_normal(100))
    field_rg1_2 = ift.Field(rg_space1, ift.random.current_rng().standard_normal(100))
    field_rg2 = ift.Field(
        rg_space2, ift.random.current_rng().standard_normal((80,60)))
    field_hp = ift.Field(hp_space, ift.random.current_rng().standard_normal(12*64**2))
    field_gl = ift.Field(gl_space, ift.random.current_rng().standard_normal(32640))
    field_ps = ift.power_analyze(fft.times(field_rg2))

    plot = ift.Plot()
    plot.add(field_rg1_1, title='Single plot')
    plot.output(name=name())

    plot = ift.Plot()
    plot.add(field_rg2, title='2d rg')
    plot.add([field_rg1_1, field_rg1_2], title='list 1d rg', label=['1', '2'])
    plot.add(field_rg1_2, title='1d rg, xmin, ymin', xmin=0.5, ymin=0.,
             xlabel='xmin=0.5', ylabel='ymin=0')
    plot.output(title='Three plots', name=name())

    plot = ift.Plot()
    plot.add(field_hp, title='HP planck-color', colormap='Planck-like')
    plot.add(field_rg1_2, title='1d rg')
    plot.add(field_ps)
    plot.add(field_gl, title='GL')
    plot.add(field_rg2, title='2d rg')
    plot.output(nx=2, ny=3, title='Five plots', name=name())


def test_mf_plot():
    x_space = ift.RGSpace((16, 16))
    f_space = ift.RGSpace(4)

    d1 = ift.DomainTuple.make([x_space, f_space])
    d2 = ift.DomainTuple.make([f_space, x_space])

    f1 = ift.from_random('normal', d1)
    f2 = ift.makeField(d2, np.moveaxis(f1.val, -1, 0))

    plot = ift.Plot()
    plot.add(f1, block=False, title='f_space_idx = 1')
    plot.add(f2, freq_space_idx=0, title='f_space_idx = 0')
    plot.output(nx=2, ny=1, title='MF-Plots, should look identical', name=name())
