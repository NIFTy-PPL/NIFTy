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


def test_broadcast():
    dom = ift.RGSpace((12,2), (0.2,11.))

    op = ift.ScalingOperator(dom, 1.).ducktape("inp").exp()

    # Operator
    op1 = op.broadcast(0, ift.UnstructuredDomain(3))
    loc = ift.from_random(op1.domain)
    ift.extra.check_operator(op1, loc, ntries=3)
    res = op1(loc).val
    ref = np.broadcast_to(op(loc).val[None], op1.target.shape)
    assert_allclose(res, ref)

    # Linearization
    lin = op(ift.Linearization.make_var(loc))
    lin1 = lin.broadcast(0, ift.UnstructuredDomain(3))
    assert_allclose(lin1.val.val, ref)

    # Field
    fld1 = loc["inp"].exp().broadcast(0, ift.UnstructuredDomain(3))
    assert_allclose(fld1.val, ref)
