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

from itertools import count

import nifty.cl as ift
import numpy as np
import pytest
from nifty.cl.plot import _make_rgb_data

from .common import setup_function, teardown_function

name = (f'plot{nr}.png' for nr in count())


def test_plots():
    rg_space1 = ift.makeDomain(ift.RGSpace((10,)))
    rg_space2 = ift.makeDomain(ift.RGSpace((8, 6), distances=1))
    hp_space = ift.makeDomain(ift.HPSpace(5))
    gl_space = ift.makeDomain(ift.GLSpace(10))
    us_space = ift.makeDomain(ift.UnstructuredDomain(10))

    fft = ift.FFTOperator(rg_space2)

    field_rg1_1 = ift.from_random(rg_space1, 'normal')
    field_rg1_2 = ift.from_random(rg_space1, 'normal')
    field_rg2 = ift.from_random(rg_space2, 'normal')
    field_hp = ift.from_random(hp_space, 'normal')
    field_gl = ift.from_random(gl_space, 'normal')
    field_ps = ift.power_analyze(fft.times(field_rg2))
    field_us = ift.from_random(us_space, 'normal')

    plot = ift.Plot()
    plot.add(field_rg1_1, title='Single plot')
    plot.output(name=next(name))

    plot = ift.Plot()
    plot.add(field_rg2, title='2d rg')
    plot.add([field_rg1_1, field_rg1_2], title='list 1d rg', label=['1', '2'])
    plot.add(field_rg1_2, title='1d rg, xmin, ymin', xmin=0.5, ymin=0.,
             xlabel='xmin=0.5', ylabel='ymin=0')
    plot.output(title='Three plots', name=next(name))
    field_us1 = ift.from_random(us_space, 'normal')

    plot = ift.Plot()
    plot.add(field_hp, title='HP planck-color', cmap='Planck-like')
    plot.add(field_rg1_2, title='1d rg')
    plot.add(field_ps)
    plot.add(field_gl, title='GL')
    plot.add(field_rg2, title='2d rg')
    plot.add(field_us, title='Histogram')
    plot.add([field_us, field_us1], label="ii", title='Two Histograms')
    plot.output(title='Seven plots', name=next(name))



def test_mf_plot():
    x_space = ift.RGSpace((16, 16))
    f_space = ift.RGSpace(4)

    d1 = ift.DomainTuple.make([x_space, f_space])
    d2 = ift.DomainTuple.make([f_space, x_space])

    f1 = ift.from_random(d1, 'normal').exp()
    f2 = ift.makeField(d2, np.moveaxis(f1.asnumpy(), -1, 0))

    plot = ift.Plot()
    plot.add(f1, block=False, title='f_space_idx = 1')
    plot.add(f2, freq_space_idx=0, title='f_space_idx = 0')
    plot.output(nx=2, ny=1, title='MF-Plots, should look identical',
                name=next(name))


def test_iter_plot():
    for ind in range(2):
        x_space = ift.RGSpace((16, 16))
        f_space = ift.UnstructuredDomain(4)

        if ind == 0:
            dom = ift.DomainTuple.make([x_space, f_space])
        else:
            dom = ift.DomainTuple.make([f_space, x_space])

        fld = ift.from_random(dom, 'normal')

        plot = ift.Plot()
        plot.add(fld)
        plot.output(title='Should be 4 plots', name=next(name))


def test_EnergyHistory_plot():
    eh = ift.minimization.iteration_controllers.EnergyHistory()
    for i in range(5):
        eh.append((i, (i + 1)**-2))
    plot = ift.Plot()
    plot.add(eh)
    plot.output(title='EnergyHistory plot', name=next(name))


# --- spectro-chromatic plotting: color_mapping_kwargs validation ---

def _spectral_cube(n_bins=8):
    f_space = ift.RGSpace(n_bins, distances=1.)
    val = np.exp(np.random.default_rng(0).standard_normal((16, 16, n_bins)))
    return val, f_space


def _rgb(**color_mapping_kwargs):
    val, f_space = _spectral_cube()
    return _make_rgb_data(val, f_space, color_mapping_kwargs)


@pytest.mark.parametrize('orphan', [
    dict(black=1.),
    dict(dynamic_range=100.),
    dict(highlights='clip_channels'),
    dict(black=1., highlights='clamp'),
])
def test_range_options_without_white_point_raise(orphan):
    # These only mean something relative to a white point; silently dropping them
    # would render the image with an auto white point and no hint that the
    # requested settings were ignored.
    with pytest.raises(ValueError, match="no white point"):
        _rgb(flux_convention='bin_integrated_flux', spectral_axis_type='energy',
             visible_bin_width='uniform', **orphan)


def test_range_options_are_accepted_alongside_a_white_point():
    setup = dict(flux_convention='bin_integrated_flux', spectral_axis_type='energy',
                 visible_bin_width='uniform')
    for range_kwargs in (dict(quantiles=(0.3, 0.99), highlights='clip_channels'),
                         dict(white=10., black=1.),
                         dict(white=10., dynamic_range=100.),
                         dict(quantiles=(0.3, 0.99), dynamic_range=50.)):
        rgb = _rgb(**setup, **range_kwargs)
        assert rgb.shape == (16, 16, 3)
        assert np.all(rgb >= 0.) and np.all(rgb <= 1.)


def test_log_compression_without_white_point_raises():
    # would otherwise fail deeper inside the projector, phrased in terms of
    # set_luminance_range rather than of color_mapping_kwargs
    with pytest.raises(ValueError, match="log_compression"):
        _rgb(flux_convention='bin_integrated_flux', spectral_axis_type='energy',
             visible_bin_width='uniform', log_compression=True)


def test_no_range_options_at_all_is_allowed():
    # the auto white point path must keep working, so that Plot.add(field) stays
    # a one-liner
    assert _rgb(flux_convention='bin_integrated_flux', spectral_axis_type='energy',
                visible_bin_width='uniform').shape == (16, 16, 3)


def test_white_and_quantiles_are_mutually_exclusive():
    with pytest.raises(ValueError, match="at most one"):
        _rgb(flux_convention='bin_integrated_flux', spectral_axis_type='energy',
             visible_bin_width='uniform', white=10., quantiles=(0.3, 0.99))


def test_unknown_color_mapping_key_raises():
    with pytest.raises(ValueError, match="unknown"):
        _rgb(flux_convention='bin_integrated_flux', saturation_flux=1.)
