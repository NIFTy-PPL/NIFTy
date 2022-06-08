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
from numpy.testing import assert_allclose, assert_equal

from .common import setup_function, teardown_function

dom = ift.makeDomain({"d1": ift.RGSpace(10)})


def test_vdot():
    f1 = ift.from_random(domain=dom, random_type="normal", dtype=np.complex128)
    f2 = ift.from_random(domain=dom, random_type="normal", dtype=np.complex128)
    assert_allclose(f1.s_vdot(f2), np.conj(f2.s_vdot(f1)))


def test_func():
    f1 = ift.from_random(domain=dom, random_type="normal", dtype=np.complex128)
    assert_allclose(
        f1.ptw("exp").ptw("log")["d1"].val, f1["d1"].val)


def test_multifield_field_consistency():
    f1 = ift.full(dom, 27)
    f2 = ift.makeField(dom['d1'], f1['d1'].val)
    assert_equal(f1.s_sum(), f2.s_sum())
    assert_equal(f1.size, f2.size)


def test_dataconv():
    f1 = ift.full(dom, 27)
    f2 = ift.makeField(dom, f1.val)
    for key, val in f1.items():
        assert_equal(val.val, f2[key].val)
    if "d1" not in f2:
        raise KeyError()
    assert_equal({"d1": f1}, f2.to_dict())
    f3 = ift.full(dom, 27+1.j)
    f4 = ift.full(dom, 1.j)
    assert_equal(f2, f3.real)
    assert_equal(f4, f3.imag)


def test_blockdiagonal():
    op = ift.BlockDiagonalOperator(
        dom, {"d1": ift.ScalingOperator(dom["d1"], 20.)})
    op2 = op(op)
    ift.extra.check_linear_operator(op2)
    assert_equal(type(op2), ift.BlockDiagonalOperator)
    f1 = op2(ift.full(dom, 1))
    for val in f1.values():
        assert_equal((val == 400).s_all(), True)
    op2 = op + op
    assert_equal(type(op2), ift.BlockDiagonalOperator)
    f1 = op2(ift.full(dom, 1))
    for val in f1.values():
        assert_equal((val == 40).s_all(), True)


def test_blockdiagonal_nontrivial():
    dom = ift.makeDomain({"d1": ift.RGSpace(10), "d2": ift.UnstructuredDomain(2)})
    op = ift.BlockDiagonalOperator(dom, {"d1": ift.ScalingOperator(dom["d1"], 2)})
    ift.extra.check_linear_operator(op)
    assert op.domain == dom
    fld = ift.from_random(dom)
    ift.extra.assert_equal(op(fld)["d1"], 2*fld["d1"])
    ift.extra.assert_equal(op(fld)["d2"], fld["d2"])
