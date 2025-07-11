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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
from numpy.testing import assert_allclose

from ..common import setup_function, teardown_function


def test_conjugation_operator():
    sp = ift.RGSpace(8)
    dom = ift.makeDomain(sp)
    f = ift.from_random(dom, dtype=np.complex128)
    op = ift.ScalingOperator(sp, 1).conjugate()
    arr = f.val
    res1 = f.conjugate()
    res2 = op(f)
    res3 = arr.conjugate()
    assert_allclose(res1.val, res2.val)
    assert_allclose(res1.val, res3)
    ift.extra.check_linear_operator(op, domain_dtype=np.float64,
                                    target_dtype=np.float64)
    ift.extra.check_linear_operator(op, domain_dtype=np.complex128,
                                    target_dtype=np.complex128, only_r_linear=True)
