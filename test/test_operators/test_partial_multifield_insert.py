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
import pytest
from numpy.testing import assert_allclose

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize
dtype = list2fixture([np.float64, np.float32, np.complex64, np.complex128])
ntries = 10


def test_part_mf_insert():
    dom = ift.RGSpace(3)
    op1 = ift.ScalingOperator(dom, 1.32).ducktape('a').ducktape_left('a1')
    op2 = ift.ScalingOperator(dom, 1).ptw("exp").ducktape('b').ducktape_left('b1')
    op3 = ift.ScalingOperator(dom, 1).ptw("sin").ducktape('c').ducktape_left('c1')
    op4 = ift.ScalingOperator(dom, 1).ducktape('c0').ducktape_left('c')**2
    op5 = ift.ScalingOperator(dom, 1).ptw("tan").ducktape('d0').ducktape_left('d')
    a = op1 + op2 + op3
    b = op4 + op5
    op = a.partial_insert(b)
    fld = ift.from_random(op.domain, 'normal')
    ift.extra.check_operator(op, fld, ntries=ntries)
    ift.myassert(op.domain is ift.MultiDomain.union(
        [op1.domain, op2.domain, op4.domain, op5.domain]))
    ift.myassert(op.target is ift.MultiDomain.union(
        [op1.target, op2.target, op3.target, op5.target]))
    x, y = fld.val, op(fld).val
    assert_allclose(y['a1'], x['a']*1.32)
    assert_allclose(y['b1'], np.exp(x['b']))
    assert_allclose(y['c1'], np.sin(x['c0']**2))
    assert_allclose(y['d'], np.tan(x['d0']))
