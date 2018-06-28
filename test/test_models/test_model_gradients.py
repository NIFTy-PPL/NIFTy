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
from itertools import product
from test.common import expand

import nifty5 as ift
import numpy as np


class Model_Tests(unittest.TestCase):
    @expand(product([ift.GLSpace(15),
                     ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [4, 78, 23]))
    def testMul(self, space, seed):
        np.random.seed(seed)
        S = ift.ScalingOperator(1., space)
        s1 = S.draw_sample()
        s2 = S.draw_sample()
        s1_var = ift.Variable(ift.MultiField({'s1': s1}))['s1']
        s2_var = ift.Variable(ift.MultiField({'s2': s2}))['s2']
        ift.extra.check_value_gradient_consistency(s1_var*s2_var)
