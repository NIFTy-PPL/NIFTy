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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import unittest

import nifty5 as ift
import numpy as np
from numpy.testing import assert_allclose, assert_equal

dom = ift.makeDomain({"d1": ift.RGSpace(10)})


class Test_Functionality(unittest.TestCase):
    def test_vdot(self):
        f1 = ift.from_random("normal", domain=dom, dtype=np.complex128)
        f2 = ift.from_random("normal", domain=dom, dtype=np.complex128)
        assert_allclose(f1.vdot(f2), np.conj(f2.vdot(f1)))

    def test_func(self):
        f1 = ift.from_random("normal", domain=dom, dtype=np.complex128)
        assert_allclose(ift.log(ift.exp((f1)))["d1"].local_data,
                        f1["d1"].local_data)

    def test_dataconv(self):
        f1 = ift.full(dom, 27)
        f2 = ift.from_global_data(dom, f1.to_global_data())
        for key, val in f1.items():
            assert_equal(val.local_data, f2[key].local_data)

    def test_blockdiagonal(self):
        op = ift.BlockDiagonalOperator(
            dom, (ift.ScalingOperator(20., dom["d1"]),))
        op2 = op*op
        ift.extra.consistency_check(op2)
        assert_equal(type(op2), ift.BlockDiagonalOperator)
        f1 = op2(ift.full(dom, 1))
        for val in f1.values():
            assert_equal((val == 400).all(), True)
        op2 = op+op
        assert_equal(type(op2), ift.BlockDiagonalOperator)
        f1 = op2(ift.full(dom, 1))
        for val in f1.values():
            assert_equal((val == 40).all(), True)
