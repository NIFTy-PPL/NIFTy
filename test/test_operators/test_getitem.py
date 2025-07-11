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
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import pytest

from ..common import setup_function, teardown_function


def test_operator_getitem():
    dom = ift.RGSpace([2])
    key = "foo"
    op = ift.ScalingOperator(dom, 2.4)

    op1 = op.ducktape_left(key)
    op2 = op1[key]

    fld = ift.from_random(op1.domain)
    res1 = op1(fld)
    res2 = ift.MultiField.from_dict({key: op2(fld)})
    ift.extra.assert_allclose(res1, res2)
    ift.extra.assert_allclose(res1[key], res2[key])

    lin = ift.Linearization.make_var(fld)
    res1 = op1(lin)
    res2 = op2(lin)
    ift.extra.assert_allclose(res1.val[key], res2.val)

    with pytest.raises(TypeError):
        op[key]
