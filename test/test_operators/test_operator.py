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
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..common import setup_function, teardown_function


def test_ducktape():
    dom = ift.RGSpace(10)
    op = ift.FFTOperator(dom)
    fld = ift.full(dom, 2.)
    lin = ift.Linearization.make_var(fld)
    a = "foo"
    op1 = op.ducktape(a)
    mdom = ift.MultiDomain.make({a: dom})
    assert op1.domain == mdom
    assert op1.target == op.target
    op1 = op.ducktape_left(a)
    assert op1.target == ift.MultiDomain.make({a: op.target})
    assert op1.domain == op.domain
    with pytest.raises(RuntimeError):
        fld.ducktape(a)
    with pytest.raises(RuntimeError):
        lin.ducktape(a)
    fld0 = fld.ducktape_left(a)
    assert fld0.domain == mdom
    lin0 = lin.ducktape_left(a)
    assert lin0.domain == lin.domain
    assert lin0.target == mdom
