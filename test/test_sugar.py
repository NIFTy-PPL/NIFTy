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

import nifty8 as ift
import numpy as np
import pytest
from numpy.testing import assert_equal

from .common import setup_function, teardown_function

pmp = pytest.mark.parametrize


def test_get_signal_variance():
    space = ift.RGSpace(3)
    hspace = space.get_default_codomain()
    sv = ift.get_signal_variance(lambda x: np.ones_like(x), hspace)
    assert_equal(sv, 3.)

    space = ift.RGSpace(3, distances=1.)
    hspace = space.get_default_codomain()

    def spec2(k):
        t = np.zeros_like(k)
        t[k == 0] = 1.
        return t
    assert_equal(ift.get_signal_variance(spec2, hspace), 1/9.)


def test_exec_time():
    dom = ift.RGSpace(12, harmonic=True)
    op = ift.HarmonicTransformOperator(dom)
    op1 = op.ptw("exp")
    lh = ift.GaussianEnergy(domain=op.target, sampling_dtype=np.float64) @ op1
    ic = ift.GradientNormController(iteration_limit=2)
    ham = ift.StandardHamiltonian(lh, ic_samp=ic)
    ham1 = ift.EnergyAdapter(ift.full(ham.domain, 0.), ham)
    kl = ift.SampledKLEnergy(ift.full(ham.domain, 0.), ham, 1, None, mirror_samples=False)
    ops = [op, op1, lh, ham, ham1, kl]
    for oo in ops:
        for wm in [True, False]:
            ift.exec_time(oo, wm)


@pmp('mf', [False, True])
@pmp('cplx', [False, True])
def test_calc_pos(mf, cplx):
    dom = ift.RGSpace(12, harmonic=True)
    op = ift.HarmonicTransformOperator(dom).ptw("exp")
    if mf:
        op = op.ducktape_left('foo')
        dom = ift.makeDomain({'': dom})
    if cplx:
        op = op + 1j*op
    fld = op(0.1 * ift.from_random(op.domain, 'normal'))
    pos = ift.calculate_position(op, fld)
    ift.extra.assert_allclose(op(pos), fld, 1e-1, 1e-1)


def test_isinstance_helpers():
    dom = ift.RGSpace(12, harmonic=True)
    op = ift.ScalingOperator(dom, 12.)
    fld = ift.full(dom, 0.)
    lin = ift.Linearization.make_var(fld)
    assert not ift.is_fieldlike(op)
    assert ift.is_fieldlike(lin)
    assert ift.is_fieldlike(fld)
    assert not ift.is_linearization(op)
    assert ift.is_linearization(lin)
    assert not ift.is_linearization(fld)
    assert ift.is_operator(op)
    assert not ift.is_operator(lin)
    assert not ift.is_operator(fld)

@pmp('dom_shape', [10, (10,20)])
@pmp('n_samples', [2, 5])
@pmp('common_colorbar', (True, False))
def test_plot_priorsamples(dom_shape, n_samples, common_colorbar):
    dom = ift.RGSpace(dom_shape)
    op = ift.ScalingOperator(dom, 1.)
    ift.plot_priorsamples(op, n_samples, common_colorbar, name=f'test_plot_priorsamples_{dom_shape}.png')
