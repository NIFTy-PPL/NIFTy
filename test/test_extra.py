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

from time import time

import nifty8 as ift
import numpy as np
import pytest

from .common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize


class NonPureOperator(ift.Operator):
    def __init__(self, domain):
        self._domain = self._target = ift.makeDomain(domain)

    def apply(self, x):
        self._check_input(x)
        return x*time()


class NonPureLinearOperator(ift.LinearOperator):
    def __init__(self, domain, cap):
        self._domain = self._target = ift.makeDomain(domain)
        self._capability = cap

    def apply(self, x, mode):
        self._check_input(x, mode)
        return x*time()


@pmp("cap", [ift.LinearOperator.ADJOINT_TIMES,
             ift.LinearOperator.INVERSE_TIMES | ift.LinearOperator.TIMES])
@pmp("ddtype", [np.float64, np.complex128])
@pmp("tdtype", [np.float64, np.complex128])
def test_purity_check_linear(cap, ddtype, tdtype):
    dom = ift.RGSpace(2)
    op = NonPureLinearOperator(dom, cap)
    with pytest.raises(AssertionError):
        ift.extra.check_linear_operator(op, ddtype, tdtype)


@pmp("dtype", [np.float64, np.complex128])
def test_purity_check(dtype):
    dom = ift.RGSpace(2)
    op = NonPureOperator(dom)
    with pytest.raises(AssertionError):
        ift.extra.check_operator(op, dtype)
