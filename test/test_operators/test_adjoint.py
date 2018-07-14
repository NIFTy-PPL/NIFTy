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

    @expand(product(_h_RG_spaces+_p_RG_spaces,
                    [np.float64, np.complex128]))
    def testHartley(self, sp, dtype):
        op = ift.HartleyOperator(sp)
        ift.extra.consistency_check(op, dtype, dtype)
        op = ift.HartleyOperator(sp.get_default_codomain())
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

    @expand(product([0, 1, 2, 3, (0, 1), (0, 2), (0, 1, 2), (0, 2, 3), (1, 3)],
                    [np.float64, np.complex128]))
    def testDomainDistributor(self, spaces, dtype):
        dom = (ift.RGSpace(10), ift.UnstructuredDomain(13), ift.GLSpace(5),
               ift.HPSpace(4))
        op = ift.DomainDistributor(dom, spaces)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product([0, 2], [np.float64, np.complex128]))
    def testSymmetrizingOperator(self, space, dtype):
        dom = (ift.LogRGSpace(10, [2.], [1.]), ift.UnstructuredDomain(13),
               ift.LogRGSpace((5, 27), [1., 2.7], [0., 4.]), ift.HPSpace(4))
        op = ift.SymmetrizingOperator(dom, space)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product([0, 2], [2, 2.7], [np.float64, np.complex128]))
    def testZeroPadder(self, space, factor, dtype):
        dom = (ift.RGSpace(10), ift.UnstructuredDomain(13), ift.RGSpace(7, 12),
               ift.HPSpace(4))
        op = ift.FieldZeroPadder(dom, factor, space)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product([(ift.RGSpace(10, harmonic=True), 4, 0),
                     (ift.RGSpace((24, 31), distances=(0.4, 2.34),
                                  harmonic=True), (4, 3), 0),
                     ((ift.HPSpace(4), ift.RGSpace(27, distances=0.3,
                                                   harmonic=True)), (10,), 1),
                     (ift.PowerSpace(ift.RGSpace(10, distances=0.3,
                                     harmonic=True)), 6, 0)],
                    [np.float64, np.complex128]))
    def testExpTransform(self, args, dtype):
        op = ift.ExpTransform(args[0], args[1], args[2])
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product([(ift.LogRGSpace([10, 17], [2., 3.], [1., 0.]), 0),
                     ((ift.LogRGSpace(10, [2.], [1.]),
                       ift.UnstructuredDomain(13)), 0),
                     ((ift.UnstructuredDomain(13),
                       ift.LogRGSpace(17, [3.], [.7])), 1)],
                    [np.float64]))
    def testQHTOperator(self, args, dtype):
        dom = ift.DomainTuple.make(args[0])
        tgt = list(dom)
        tgt[args[1]] = tgt[args[1]].get_default_codomain()
        op = ift.QHTOperator(tgt, dom[args[1]], args[1])
        ift.extra.consistency_check(op, dtype, dtype)
