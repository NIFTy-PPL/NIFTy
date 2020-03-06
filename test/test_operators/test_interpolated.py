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

import pytest
from numpy.testing import assert_allclose
from scipy.stats import invgamma, norm

import nifty6 as ift

from ..common import list2fixture

pmp = pytest.mark.parametrize
pmp = pytest.mark.parametrize
space = list2fixture([ift.GLSpace(15),
                      ift.RGSpace(64, distances=.789),
                      ift.RGSpace([32, 32], distances=.789)])
seed = list2fixture([4, 78, 23])


def testInverseGammaAccuracy(space, seed):
    S = ift.ScalingOperator(space, 1.)
    pos = S.draw_sample()
    alpha = 1.5
    q = 0.73
    op = ift.InverseGammaOperator(space, alpha, q)
    arr1 = op(pos).val
    arr0 = q*invgamma.ppf(norm.cdf(pos.val), alpha)
    assert_allclose(arr0, arr1)
