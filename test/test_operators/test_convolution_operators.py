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

import nifty8 as ift
import numpy as np
from numpy.testing import assert_allclose

from ..common import list2fixture, setup_function, teardown_function

space = list2fixture([
    ift.RGSpace(4),
    ift.HPSpace(4),
    ift.GLSpace(4)
])


def test_const_func(space):
    sig = ift.Field.from_random(domain=space, random_type='normal')
    fco_op = ift.FuncConvolutionOperator(space, lambda x: np.ones(x.shape))
    vals = fco_op(sig).val
    vals = np.round(vals, decimals=5)
    assert len(np.unique(vals)) == 1


def gauss(x, sigma):
    normalization = np.sqrt(2. * np.pi) * sigma
    return np.exp(-0.5 * x * x / sigma**2) / normalization


def test_gaussian_smoothing():
    N = 128
    sigma = N / 10**4
    dom = ift.RGSpace(N)
    sig = ift.Field.from_random(dom, 'normal').ptw("exp")
    fco_op = ift.FuncConvolutionOperator(dom, lambda x: gauss(x, sigma))
    sm_op = ift.HarmonicSmoothingOperator(dom, sigma)
    assert_allclose(fco_op(sig).val,
                    sm_op(sig).val,
                    rtol=1e-05)
    assert_allclose(fco_op.adjoint_times(sig).val,
                    sm_op.adjoint_times(sig).val,
                    rtol=1e-05)
