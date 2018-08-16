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

from numpy.testing import assert_allclose

import nifty5 as ift


class Regridding_Tests(unittest.TestCase):
    def test_value(self):
        s = ift.RGSpace(8)
        Regrid = ift.RegriddingOperator(s, s.shape, 0)
        f = ift.from_random('normal', Regrid.domain)
        assert_allclose(f.to_global_data(), Regrid(f).to_global_data())
