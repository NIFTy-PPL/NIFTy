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
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from itertools import product
import nifty4 as ift
from test.common import expand

dom = ift.makeDomain({"d1": ift.RGSpace(10)})

class Test_Functionality(unittest.TestCase):
    def test_vdot(self):
        f1 = ift.from_random("normal", domain=dom, dtype=np.complex128)
        f2 = ift.from_random("normal", domain=dom, dtype=np.complex128)
        assert_allclose(f1.vdot(f2), np.conj(f2.vdot(f1)))

    def test_lock(self):
        s1 = ift.RGSpace((10,))
        f1 = ift.full(dom, 27)
        assert_equal(f1.locked, False)
        f1.lock()
        assert_equal(f1.locked, True)
        with assert_raises(ValueError):
            f1 += f1
        assert_equal(f1.locked_copy() is f1, True)

    def test_fill(self):
        s1 = ift.RGSpace((10,))
        f1 = ift.full(s1, 27)
        assert_equal((f1.fill(10) == 10).all(), True)

    def test_dataconv(self):
        s1 = ift.RGSpace((10,))
        ld = np.arange(ift.dobj.local_shape(s1.shape)[0])
        gd = np.arange(s1.shape[0])
        assert_equal(ld, ift.from_local_data(s1, ld).local_data)
        assert_equal(gd, ift.from_global_data(s1, gd).to_global_data())
