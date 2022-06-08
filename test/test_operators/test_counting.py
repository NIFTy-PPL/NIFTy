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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest
from numpy.testing import assert_equal

from ..common import list2fixture, setup_function, teardown_function


def test_counting_operator():
    dom = ift.UnstructuredDomain(2)
    inp = ift.Operator.identity_operator(dom)

    counting = ift.CountingOperator(dom)

    op = inp.exp() @ counting

    assert_equal(counting.count_apply, 0)
    assert_equal(counting.count_apply_lin, 0)
    assert_equal(counting.count_jac, 0)
    assert_equal(counting.count_jac_adj, 0)

    fld = ift.from_random(op.domain)
    lin = ift.Linearization.make_var(fld)
    op(fld)
    op(fld)

    assert_equal(counting.count_apply, 2)
    assert_equal(counting.count_apply_lin, 0)
    assert_equal(counting.count_jac, 0)
    assert_equal(counting.count_jac_adj, 0)

    res = op(lin)

    assert_equal(counting.count_apply, 2)
    assert_equal(counting.count_apply_lin, 1)
    assert_equal(counting.count_jac, 0)
    assert_equal(counting.count_jac_adj, 0)

    for _ in range(3):
        res.jac(fld)

    assert_equal(counting.count_apply, 2)
    assert_equal(counting.count_apply_lin, 1)
    assert_equal(counting.count_jac, 3)
    assert_equal(counting.count_jac_adj, 0)

    for _ in range(5):
        res.jac.adjoint(res.val)

    assert_equal(counting.count_apply, 2)
    assert_equal(counting.count_apply_lin, 1)
    assert_equal(counting.count_jac, 3)
    assert_equal(counting.count_jac_adj, 5)

    counting.__repr__()

    ift.extra.check_operator(counting, fld)
