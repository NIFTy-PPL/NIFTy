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
from numpy.testing import assert_allclose

from ..common import list2fixture, setup_function, teardown_function

s = list2fixture([
    ift.RGSpace(8, distances=12.9),
    ift.RGSpace(59, distances=.24, harmonic=True),
    ift.RGSpace([12, 3])
])


def test_value(s):
    Regrid = ift.RegriddingOperator(s, s.shape)
    f = ift.from_random(Regrid.domain, 'normal')
    assert_allclose(f.val, Regrid(f).val)
