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

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises

import nifty5 as ift


def test_get_signal_variance():
    space = ift.RGSpace(3)
    hspace = space.get_default_codomain()
    spec1 = lambda x: np.ones_like(x)
    assert_equal(ift.get_signal_variance(spec1, hspace), 3.)

    space = ift.RGSpace(3, distances=1.)
    hspace = space.get_default_codomain()

    def spec2(k):
        t = np.zeros_like(k)
        t[k == 0] = 1.
        return t
    assert_equal(ift.get_signal_variance(spec2, hspace), 1/9.)
