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

import nifty5 as ift
import numpy as np
import unittest
from itertools import product
from test.common import expand
from numpy.testing import assert_allclose, assert_raises


class EnergySum_Tests(unittest.TestCase):
    @expand(product([ift.GLSpace(15),
                     ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789),
                     [ift.RGSpace(3), ift.RGSpace(12)],
                     {'a': ift.GLSpace(15), 'b': ift.RGSpace(64)}],
                    [4, 78, 23]))
    def testSum(self, domain, seed):
        np.random.seed(seed)
        pos = ift.from_random("normal", domain)
        A1 = ift.makeOp(ift.from_random("normal", domain))
        b1 = ift.from_random("normal", domain)
        E1 = ift.QuadraticEnergy(pos, A1, b1)
        A2 = ift.makeOp(ift.from_random("normal", domain))
        b2 = ift.from_random("normal", domain)
        E2 = ift.QuadraticEnergy(pos, A2, b2)
        assert_allclose((E1+E2).value, E1.value+E2.value)
        assert_allclose((E1-E2).value, E1.value-E2.value)
        assert_allclose((3.8*E1+E2*4).value, 3.8*E1.value+4*E2.value)
        assert_allclose((-E1).value, -(E1.value))
        with assert_raises(TypeError):
            E1*E2
        with assert_raises(TypeError):
            E1*2j
        with assert_raises(TypeError):
            E1+2
        with assert_raises(TypeError):
            E1-"hello"
        with assert_raises(ValueError):
            E1+E2.at(2*pos)
