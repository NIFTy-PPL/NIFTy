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
    @expand(product(_p_RG_spaces, [np.float64, np.complex128]))
    def testLOSResponse(self, sp, dtype):
        starts = np.random.randn(len(sp.shape), 10)
        ends = np.random.randn(len(sp.shape), 10)
        sigma_low = 1e-4*np.random.randn(10)
        sigma_ups = 1e-5*np.random.randn(10)
        op = ift.LOSResponse(sp, starts, ends, sigma_low, sigma_ups)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_h_spaces + _p_spaces + _pow_spaces,
                    [np.float64, np.complex128]))
    def testOperatorCombinations(self, sp, dtype):
        a = ift.DiagonalOperator(ift.Field.from_random("normal", sp,
                                                       dtype=dtype))
        b = ift.DiagonalOperator(ift.Field.from_random("normal", sp,
                                                       dtype=dtype))
        op = ift.SandwichOperator.make(a, b)
        ift.extra.consistency_check(op, dtype, dtype)
        op = a(b)
        ift.extra.consistency_check(op, dtype, dtype)
        op = a+b
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_h_spaces + _p_spaces + _pow_spaces,
                    [np.float64, np.complex128]))
    def testVdotOperator(self, sp, dtype):
        op = ift.VdotOperator(ift.Field.from_random("normal", sp,
                                                    dtype=dtype))
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product([(ift.RGSpace(10, harmonic=True), 4, 0),
                     (ift.RGSpace((24, 31), distances=(0.4, 2.34),
                                  harmonic=True), 3, 0),
                     (ift.LMSpace(4), 10, 0)],
                    [np.float64, np.complex128]))
    def testSlopeOperator(self, args, dtype):
        tmp = ift.ExpTransform(ift.PowerSpace(args[0]), args[1], args[2])
        tgt = tmp.domain[0]
        sig = np.array([0.3, 0.13])
        op = ift.SlopeOperator(tgt, sig)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_h_spaces + _p_spaces + _pow_spaces,
                    [np.float64, np.complex128]))
    def testOperatorAdaptor(self, sp, dtype):
        op = ift.DiagonalOperator(ift.Field.from_random("normal", sp,
                                                        dtype=dtype))
        ift.extra.consistency_check(op.adjoint, dtype, dtype)
        ift.extra.consistency_check(op.inverse, dtype, dtype)
        ift.extra.consistency_check(op.inverse.adjoint, dtype, dtype)
        ift.extra.consistency_check(op.adjoint.inverse, dtype, dtype)

    @expand(product(_h_spaces + _p_spaces + _pow_spaces,
                    _h_spaces + _p_spaces + _pow_spaces,
                    [np.float64, np.complex128]))
    def testNullOperator(self, sp1, sp2, dtype):
        op = ift.NullOperator(sp1, sp2)
        ift.extra.consistency_check(op, dtype, dtype)
        mdom1 = ift.MultiDomain.make({'a': sp1})
        mdom2 = ift.MultiDomain.make({'b': sp2})
        op = ift.NullOperator(mdom1, mdom2)
        ift.extra.consistency_check(op, dtype, dtype)
        op = ift.NullOperator(sp1, mdom2)
        ift.extra.consistency_check(op, dtype, dtype)
        op = ift.NullOperator(mdom1, sp2)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_p_RG_spaces,
                    [np.float64, np.complex128]))
    def testHarmonicSmoothingOperator(self, sp, dtype):
        op = ift.HarmonicSmoothingOperator(sp, 0.1)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product(_h_spaces + _p_spaces + _pow_spaces,
                    [np.float64, np.complex128]))
    def testDOFDistributor(self, sp, dtype):
        # TODO: Test for DomainTuple
        if sp.size < 4:
            return
        dofdex = np.arange(sp.size).reshape(sp.shape) % 3
        dofdex = ift.Field.from_global_data(sp, dofdex)
        op = ift.DOFDistributor(dofdex)
        ift.extra.consistency_check(op, dtype, dtype)

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
                    [0, 1, 2, -1], [np.float64, np.complex128]))
    def testContractionOperator(self, spaces, wgt, dtype):
        dom = (ift.RGSpace(10), ift.RGSpace(13), ift.GLSpace(5),
               ift.HPSpace(4))
        op = ift.ContractionOperator(dom, spaces, wgt)
        ift.extra.consistency_check(op, dtype, dtype)

    def testDomainTupleFieldInserter(self):
        domain = ift.DomainTuple.make((ift.UnstructuredDomain(12),
                                       ift.RGSpace([4, 22])))
        new_space = ift.UnstructuredDomain(7)
        pos = (5,)
        op = ift.DomainTupleFieldInserter(domain, new_space, 0, pos)
        ift.extra.consistency_check(op)

    @expand(product([0, 2], [np.float64, np.complex128]))
    def testSymmetrizingOperator(self, space, dtype):
        dom = (ift.LogRGSpace(10, [2.], [1.]), ift.UnstructuredDomain(13),
               ift.LogRGSpace((5, 27), [1., 2.7], [0., 4.]), ift.HPSpace(4))
        op = ift.SymmetrizingOperator(dom, space)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product([0, 2], [2, 2.7], [np.float64, np.complex128],
                    [False, True]))
    def testZeroPadder(self, space, factor, dtype, central):
        dom = (ift.RGSpace(10), ift.UnstructuredDomain(13), ift.RGSpace(7, 12),
               ift.HPSpace(4))
        newshape = [int(factor*l) for l in dom[space].shape]
        op = ift.FieldZeroPadder(dom, newshape, space, central)
        ift.extra.consistency_check(op, dtype, dtype)

    @expand(product([0, 2], [2, 2.7], [np.float64, np.complex128]))
    def testZeroPadder2(self, space, factor, dtype):
        dom = (ift.RGSpace(10), ift.UnstructuredDomain(13), ift.RGSpace(7, 12),
               ift.HPSpace(4))
        newshape = [int(factor*l) for l in dom[space].shape]
        op = ift.CentralZeroPadder(dom, newshape, space)
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
        tgt = ift.DomainTuple.make(args[0])
        op = ift.QHTOperator(tgt, args[1])
        ift.extra.consistency_check(op, dtype, dtype)

    @expand([[ift.RGSpace((13, 52, 40)), (4, 6, 25), None],
             [ift.RGSpace((128, 128)), (45, 48), 0],
             [ift.RGSpace(13), (7,), None],
             [(ift.HPSpace(3), ift.RGSpace((12, 24), distances=0.3)),
              (12, 12), 1]])
    def testRegridding(self, domain, shape, space):
        op = ift.RegriddingOperator(domain, shape, space)
        ift.extra.consistency_check(op)

    @expand(product([ift.DomainTuple.make((ift.RGSpace((3, 5, 4)),
                                           ift.RGSpace((16,), distances=(7.,))),),
                     ift.DomainTuple.make(ift.HPSpace(12),)],
                    [ift.DomainTuple.make((ift.RGSpace((2,)),
                                           ift.GLSpace(10)),),
                     ift.DomainTuple.make(ift.RGSpace((10, 12), distances=(0.1, 1.)),)]
                    ))
    def testOuter(self, fdomain, domain):
        f = ift.from_random('normal', fdomain)
        op = ift.OuterProduct(f, domain)
        ift.extra.consistency_check(op)
