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

from numpy.testing import assert_allclose, assert_equal

import nifty5 as ift
import numpy as np

from ..common import list2fixture

space = list2fixture([
    ift.RGSpace(4),
    ift.HPSpace(4),
    ift.GLSpace(4)
])


def test_const_func(space):
    ones = lambda x: np.ones(x.shape)
    sig = ift.Field.from_random('normal', domain=space)
    fco_op = ift.FuncConvolutionOperator(space, ones)
    vals = fco_op(sig).to_global_data()
    vals = np.round(vals, decimals=5)
    assert len(np.unique(vals)) == 1


def gauss(x, sigma):
    normalization = np.sqrt(2. * np.pi) * sigma
    return np.exp(-0.5 * x * x / sigma**2) / normalization


def test_gaussian_smoothing():
    N = 128
    sigma = N / 10**4
    dom = ift.RGSpace(N)
    sig = ift.exp(ift.Field.from_random('normal', dom))
    fco_op = ift.FuncConvolutionOperator(dom, lambda x: gauss(x, sigma))
    sm_op = ift.HarmonicSmoothingOperator(dom, sigma)
    assert_allclose(fco_op(sig).to_global_data(),
                    sm_op(sig).to_global_data(),
                    rtol=1e-05)
    assert_allclose(fco_op.adjoint_times(sig).to_global_data(),
                    sm_op.adjoint_times(sig).to_global_data(),
                    rtol=1e-05)
