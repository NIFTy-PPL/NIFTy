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

_h_RG_spaces = [ift.RGSpace(7, distances=0.2, harmonic=True),
                ift.RGSpace((12, 46), distances=(.2, .3), harmonic=True)]
_h_spaces = _h_RG_spaces + [ift.LMSpace(17)]

_p_RG_spaces = [ift.RGSpace(19, distances=0.7),
                ift.RGSpace((1, 2, 3, 6), distances=(0.2, 0.25, 0.34, .8))]
_p_spaces = _p_RG_spaces + [ift.HPSpace(17), ift.GLSpace(8, 13)]

_pow_spaces = [ift.PowerSpace(ift.RGSpace((17, 38), harmonic=True))]


class Consistency_Tests(unittest.TestCase):
    @expand(product(_h_spaces, [np.float64, np.complex128]))
    def testPPO(self, sp, dtype):
        op = ift.PowerDistributor(target=sp)
        ift.extra.consistency_check(op, dtype, dtype)
        ps = ift.PowerSpace(
            sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=False, nbin=3))
        op = ift.PowerDistributor(target=sp, power_space=ps)
        ift.extra.consistency_check(op, dtype, dtype)
        ps = ift.PowerSpace(
            sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=True, nbin=3))
        op = ift.PowerDistributor(target=sp, power_space=ps)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_h_RG_spaces+_p_RG_spaces,
                    [np.float64, np.complex128]))
    def testFFT(self, sp, dtype):
        op = ift.FFTOperator(sp)
        ift.extra.consistency_check(op, dtype, dtype)
        op = ift.FFTOperator(sp.get_default_codomain())
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_h_spaces, [np.float64, np.complex128]))
    def testHarmonic(self, sp, dtype):
        op = ift.HarmonicTransformOperator(sp)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_p_spaces, [np.float64, np.complex128]))
    def testMask(self, sp, dtype):
        # Create mask
        f = ift.from_random('normal', sp).to_global_data()
        mask = np.zeros_like(f)
        mask[f > 0] = 1
        mask = ift.Field.from_global_data(sp, mask)
        # Test MaskOperator
        op = ift.MaskOperator(mask)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_h_spaces+_p_spaces, [np.float64, np.complex128]))
    def testDiagonal(self, sp, dtype):
        op = ift.DiagonalOperator(ift.Field.from_random("normal", sp,
                                                        dtype=dtype))
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_pow_spaces, [np.float64, np.complex128]))
    def testLaplace(self, sp, dtype):
        op = ift.LaplaceOperator(sp)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_pow_spaces, [np.float64, np.complex128]))
    def testSmoothness(self, sp, dtype):
        op = ift.SmoothnessOperator(sp)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_h_spaces+_p_spaces+_pow_spaces,
                    [np.float64, np.complex128]))
    def testGeometryRemover(self, sp, dtype):
        op = ift.GeometryRemover(sp)
        ift.extra.consistency_check(op, dtype, dtype)
