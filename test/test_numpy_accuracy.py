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

import numpy as np
import pytest
from ducc0.misc import vdot

from .common import setup_function, teardown_function

dt = np.float32

def _check(a, b):
    res0 = np.vdot(a, b)
    res1 = vdot(a, b)
    rel_error = np.abs((res0-res1)/(res0+res1)*2)
    assert rel_error < 1e-6


# When this is fixed in numpy, the warning in src/ducc_dispath.py is no longer
# necessary
@pytest.mark.xfail(reason="np.vdot inaccurate for single precision", strict=True)
def test_vdot():
    a = 100*np.ones((1000000,)).astype(dt)
    _check(a, a)


@pytest.mark.xfail(reason="np.vdot inaccurate for single precision", strict=True)
def test_vdot_extreme():
    a = np.array([1e8, 1, -1e8]).astype(dt)
    b = np.array([1e8, 1,  1e8]).astype(dt)
    _check(a, b)
