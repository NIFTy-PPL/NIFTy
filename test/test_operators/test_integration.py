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


def test_integration_operator():
    x1 = ift.RGSpace((9,), distances=2.)
    x2 = ift.RGSpace((2, 12), distances=(0.3,))
    dom1 = ift.makeDomain(x1)
    dom2 = ift.makeDomain((x1, x2))
    f1 = ift.from_random(dom1)
    f2 = ift.from_random(dom2)
    op1 = ift.ScalingOperator(dom1, 1).integrate()
    op2 = ift.ScalingOperator(dom2, 1).integrate()
    op3 = ift.ScalingOperator(dom2, 1).integrate(spaces=1)
    res1 = f1.integrate()
    res2 = op1(f1)
    res3 = (f1.val*x1.dvol).sum()
    assert_allclose(res1.val, res2.val)
    assert_allclose(res1.val, res3)
    res4 = f2.integrate()
    res5 = op2(f2)
    res6 = (f2.val*x1.dvol*x2.dvol).sum()
    assert_allclose(res4.val, res5.val)
    assert_allclose(res4.val, res6)
    res7 = f2.integrate(spaces=1)
    res8 = op3(f2)
    res9 = (f2.val*x2.dvol).sum(axis=(1, 2))
    assert_allclose(res7.val, res8.val)
    assert_allclose(res7.val, res9)
    for op in [op1, op2, op3]:
        ift.extra.check_linear_operator(op, domain_dtype=np.float64,
                                        target_dtype=np.float64)
        ift.extra.check_linear_operator(op, domain_dtype=np.complex128,
                                        target_dtype=np.complex128)
