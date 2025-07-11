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
# Copyright(C) 2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import pytest
from numpy.testing import assert_allclose

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp('mean', [3., -2])
@pmp('std', [1.5, 0.1])
@pmp('seed', [7, 21])
def test_normal_transform(mean, std, seed):
    op = ift.NormalTransform(mean, std, 'dom')
    assert op.target is ift.DomainTuple.make(())

    op = ift.NormalTransform(mean, std, 'dom', 500)

    with ift.random.Context(seed):
        res = op(ift.from_random(op.domain))
        assert_allclose(res.val.mean(), mean, rtol=0.1)
        assert_allclose(res.val.std(), std, rtol=0.1)

        loc = ift.from_random(op.domain)
        ift.extra.check_operator(op, loc)


@pmp('mean', [0.01, 10.])
@pmp('std_fct', [0.01, 0.1])
@pmp('seed', [7, 21])
def test_lognormal_transform(mean, std_fct, seed):
    std = mean * std_fct
    op = ift.LognormalTransform(mean, std, 'dom', 500)

    with ift.random.Context(seed):
        res = op(ift.from_random(op.domain))
        assert_allclose(res.val.mean(), mean, rtol=0.1)
        assert_allclose(res.val.std(), std, rtol=0.1)

        loc = ift.from_random(op.domain)
        ift.extra.check_operator(op, loc)
