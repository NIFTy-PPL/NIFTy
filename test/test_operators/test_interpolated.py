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
# Copyright(C) 2013-2022 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import gamma, invgamma, norm

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize
space = list2fixture([ift.GLSpace(15),
                      ift.RGSpace(64, distances=.789),
                      ift.RGSpace([32, 32], distances=.789)])
seed = list2fixture([4, 78, 23])


def testInterpolationAccuracy(space, seed):
    ift.random.push_sseq_from_seed(seed)
    pos = ift.from_random(space, 'normal')
    alpha = 1.5
    qs = [0.73, pos.ptw("exp").val]
    for q in qs:
        qfld = q
        if not np.isscalar(q):
            qfld = ift.makeField(space, q)
        op = ift.InverseGammaOperator(space, alpha, qfld)
        op1 = ift.LogInverseGammaOperator(space, alpha, qfld).exp()
        arr1 = op(pos).val
        arr2 = op1(pos).val
        arr0 = invgamma.ppf(norm.cdf(pos.val), alpha, scale=q)
        assert_allclose(arr0, arr1)
        assert_allclose(arr0, arr2)

        op2 = ift.GammaOperator(space, alpha=alpha, theta=qfld)
        arr1 = op2(pos).val
        arr0 = gamma.ppf(norm.cdf(pos.val), alpha, scale=q)
        assert_allclose(arr0, arr1)
