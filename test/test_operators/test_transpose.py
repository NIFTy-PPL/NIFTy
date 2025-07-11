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
# Copyright(C) 2022 Max-Planck-Society

import nifty8 as ift
import numpy as np
import pytest

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize

dom0 = ift.RGSpace([2, 43])
dom1 = ift.UnstructuredDomain([43, 1, 1])
dom2 = ift.HPSpace(3)


def test_transpose_operator_trivial():
    op = ift.ScalingOperator(dom0, 1.).transpose((0,))
    ift.extra.check_linear_operator(op)
    fld = ift.from_random(op.domain)
    ift.extra.assert_equal(op(fld), fld)


index_pairs = list2fixture([
               [(0, 1, 2), (0, 1, 2, 3, 4, 5)],
               [(0, 2, 1), (0, 1, 5, 2, 3, 4)],
               [(1, 0, 2), (2, 3, 4, 0, 1, 5)],
               [(1, 2, 0), (2, 3, 4, 5, 0, 1)],
               [(2, 1, 0), (5, 2, 3, 4, 0, 1)],
               [(2, 0, 1), (5, 0, 1, 2, 3, 4)],
              ])


def test_transpose_operator(index_pairs):
    dom = ift.makeDomain((dom0, dom1, dom2))
    space_indices, np_indices = index_pairs

    np_indices1 = ift.operators.transpose_operator._niftyspace_to_np_indices(dom, space_indices)
    assert np_indices == np_indices1

    op = ift.ScalingOperator(dom, 1.).transpose(space_indices)
    ift.extra.check_linear_operator(op)
    fld = ift.from_random(op.domain)
    res0 = op(fld)
    res1 = ift.makeField(op.target, np.transpose(fld.val, np_indices))
    ift.extra.assert_equal(res0, res1)


def test_transpose_operator_fail():
    iden = ift.ScalingOperator(dom0, 1.)
    with pytest.raises(IndexError):
        iden.transpose((1,))
    with pytest.raises(IndexError):
        iden.transpose((0, 1))
