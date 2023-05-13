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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest

from ..common import list2fixture, setup_function, teardown_function

_h_RG_spaces = [ift.RGSpace(7, distances=0.2, harmonic=True),
                ift.RGSpace((12, 46), distances=(.2, .3), harmonic=True)]
_h_spaces = _h_RG_spaces + [ift.LMSpace(17)]
_p_RG_spaces = [ift.RGSpace(19, distances=0.7),
                ift.RGSpace((1, 2, 3, 6), distances=(0.2, 0.25, 0.34, .8))]
_p_spaces = _p_RG_spaces + [ift.HPSpace(17), ift.GLSpace(8, 13)]
_pow_spaces = [ift.PowerSpace(ift.RGSpace((17, 38), (0.99, 1340), harmonic=True)),
               ift.PowerSpace(ift.LMSpace(18), ift.PowerSpace.useful_binbounds(ift.LMSpace(18), False))]

pmp = pytest.mark.parametrize
dtype = list2fixture([np.float64, np.complex128])


@pmp('sp', _p_RG_spaces)
def testLOSResponse(sp, dtype):
    starts = ift.random.current_rng().standard_normal((len(sp.shape), 10))
    ends = ift.random.current_rng().standard_normal((len(sp.shape), 10))
    sigma_low = 1e-4*ift.random.current_rng().standard_normal(10)
    sigma_ups = 1e-5*ift.random.current_rng().standard_normal(10)
    op = ift.LOSResponse(sp, starts, ends, sigma_low, sigma_ups)
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testOperatorCombinations(sp, dtype):
    a = ift.DiagonalOperator(ift.Field.from_random(sp, "normal", dtype=dtype))
    b = ift.DiagonalOperator(ift.Field.from_random(sp, "normal", dtype=dtype))
    op = ift.SandwichOperator.make(a, b)
    ift.extra.check_linear_operator(op, dtype, dtype)
    op = a(b)
    ift.extra.check_linear_operator(op, dtype, dtype)
    op = a + b
    ift.extra.check_linear_operator(op, dtype, dtype)
    op = a - b
    ift.extra.check_linear_operator(op, dtype, dtype)


def testLinearInterpolator():
    sp = ift.RGSpace((10, 8), distances=(0.1, 3.5))
    pos = ift.random.current_rng().random((2, 23))
    pos[0, :] *= 0.9
    pos[1, :] *= 7*3.5
    op = ift.LinearInterpolator(sp, pos)
    ift.extra.check_linear_operator(op)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testRealizer(sp):
    op = ift.Realizer(sp)
    ift.extra.check_linear_operator(op, np.complex128, np.float64,
                                only_r_linear=True)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testImaginizer(sp):
    op = ift.Imaginizer(sp)
    ift.extra.check_linear_operator(op, np.complex128, np.float64,
                                only_r_linear=True)
    loc = ift.from_random(op.domain, dtype=np.complex128)
    ift.extra.check_operator(op, loc)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testConjugationOperator(sp):
    op = ift.ConjugationOperator(sp)
    ift.extra.check_linear_operator(op, np.complex128, np.complex128,
                                only_r_linear=True)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testOperatorAdaptor(sp, dtype):
    op = ift.DiagonalOperator(ift.Field.from_random(sp, "normal", dtype=dtype))
    ift.extra.check_linear_operator(op.adjoint, dtype, dtype)
    ift.extra.check_linear_operator(op.inverse, dtype, dtype)
    ift.extra.check_linear_operator(op.inverse.adjoint, dtype, dtype)
    ift.extra.check_linear_operator(op.adjoint.inverse, dtype, dtype)


@pmp('sp1', _h_spaces + _p_spaces + _pow_spaces)
@pmp('sp2', _h_spaces + _p_spaces + _pow_spaces)
def testNullOperator(sp1, sp2, dtype):
    op = ift.NullOperator(sp1, sp2)
    ift.extra.check_linear_operator(op, dtype, dtype)
    mdom1 = ift.MultiDomain.make({'a': sp1})
    mdom2 = ift.MultiDomain.make({'b': sp2})
    op = ift.NullOperator(mdom1, mdom2)
    ift.extra.check_linear_operator(op, dtype, dtype)
    op = ift.NullOperator(sp1, mdom2)
    ift.extra.check_linear_operator(op, dtype, dtype)
    op = ift.NullOperator(mdom1, sp2)
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _p_RG_spaces)
def testHarmonicSmoothingOperator(sp, dtype):
    op = ift.HarmonicSmoothingOperator(sp, 0.1)
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testDOFDistributor(sp, dtype):
    # TODO: Test for DomainTuple
    if sp.size < 4:
        return
    dofdex = np.arange(sp.size).reshape(sp.shape) % 3
    dofdex = ift.Field.from_raw(sp, dofdex)
    op = ift.DOFDistributor(dofdex)
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _h_spaces)
def testPPO(sp, dtype):
    op = ift.PowerDistributor(target=sp)
    ift.extra.check_linear_operator(op, dtype, dtype)
    ps = ift.PowerSpace(
        sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=False, nbin=3))
    op = ift.PowerDistributor(target=sp, power_space=ps)
    ift.extra.check_linear_operator(op, dtype, dtype)
    ps = ift.PowerSpace(
        sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=True, nbin=3))
    op = ift.PowerDistributor(target=sp, power_space=ps)
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _h_RG_spaces + _p_RG_spaces)
def testFFT(sp, dtype):
    op = ift.FFTOperator(sp)
    ift.extra.check_linear_operator(op, dtype, dtype)
    op = ift.FFTOperator(sp.get_default_codomain())
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _h_RG_spaces + _p_RG_spaces)
def testHartley(sp, dtype):
    op = ift.HartleyOperator(sp)
    ift.extra.check_linear_operator(op, dtype, dtype)
    op = ift.HartleyOperator(sp.get_default_codomain())
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _h_spaces)
def testHarmonic(sp, dtype):
    op = ift.HarmonicTransformOperator(sp)
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _p_spaces)
def testMask(sp, dtype):
    f = ift.from_random(sp).val
    mask = np.zeros_like(f)
    mask[f > 0] = 1
    mask = ift.Field.from_raw(sp, mask)
    op = ift.MaskOperator(mask)
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _h_spaces + _p_spaces)
def testDiagonal(sp, dtype):
    op = ift.DiagonalOperator(ift.Field.from_random(sp, dtype=dtype))
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testGeometryRemover(sp, dtype):
    op = ift.GeometryRemover(sp)
    ift.extra.check_linear_operator(op, dtype, dtype)

@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testExtractAtIndices(sp, dtype):
    min_ax = np.min(sp.shape)
    n_ax = len(sp.shape)
    inds = (list(range(min_ax//2))+list(range(min_ax//3)), )*n_ax
    op = ift.ExtractAtIndices(sp, inds)
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('spaces', [0, 1, 2, 3, (0, 1), (0, 2), (0, 1, 2), (0, 2, 3), (1, 3)])
@pmp('wgt', [0, 1, 2, -1])
def testContractionOperator(spaces, wgt, dtype):
    dom = (ift.RGSpace(1), ift.RGSpace(2), ift.GLSpace(3), ift.HPSpace(2))
    op = ift.ContractionOperator(dom, spaces, wgt)
    ift.extra.check_linear_operator(op, dtype, dtype)


def testDomainTupleFieldInserter():
    target = ift.DomainTuple.make((ift.UnstructuredDomain([3, 2]),
                                   ift.UnstructuredDomain(7),
                                   ift.RGSpace([4, 22])))
    op = ift.DomainTupleFieldInserter(target, 1, (5,))
    ift.extra.check_linear_operator(op)


@pmp('space', [0, 2])
@pmp('factor', [1, 2, 2.7])
@pmp('central', [False, True])
def testZeroPadder(space, factor, dtype, central):
    dom = (ift.RGSpace(4), ift.UnstructuredDomain(5), ift.RGSpace(3, 4),
           ift.HPSpace(2))
    newshape = [int(factor*ll) for ll in dom[space].shape]
    op = ift.FieldZeroPadder(dom, newshape, space, central)
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('args', [[ift.RGSpace((13, 52, 40)), (4, 6, 25), None],
              [ift.RGSpace((128, 128)), (45, 48), 0],
              [ift.RGSpace(13), (7,), None],
              [(ift.HPSpace(3), ift.RGSpace((12, 24), distances=0.3)), (12, 12), 1]])
def testRegridding(args):
    op = ift.RegriddingOperator(*args)
    ift.extra.check_linear_operator(op)


@pmp('fdomain', [(ift.RGSpace((2, 3, 2)), ift.RGSpace((4,), distances=(7.,))),
                 ift.HPSpace(3)])
@pmp('domain', [(ift.RGSpace(2), ift.GLSpace(10)),
                ift.RGSpace((4, 3), distances=(0.1, 1.))])
def testOuter(fdomain, domain):
    f = ift.from_random(ift.makeDomain(fdomain), 'normal')
    op = ift.OuterProduct(domain, f)
    ift.extra.check_linear_operator(op)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
@pmp('seed', [12, 3])
def testValueInserter(sp, seed):
    with ift.random.Context(seed):
        ind = []
        for ss in sp.shape:
            if ss == 1:
                ind.append(0)
            else:
                ind.append(int(ift.random.current_rng().integers(0, ss-1)))
        op = ift.ValueInserter(sp, ind)
        ift.extra.check_linear_operator(op)


@pmp('sp', _pow_spaces)
def testSlopeRemover(sp):
    op = ift.library.correlated_fields._SlopeRemover(sp)
    ift.extra.check_linear_operator(op)


@pmp('sp', _pow_spaces)
def testTwoLogIntegrations(sp):
    op = ift.library.correlated_fields._TwoLogIntegrations(sp)
    ift.extra.check_linear_operator(op)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testSpecialSum(sp):
    op = ift.library.correlated_fields._SpecialSum(sp)
    ift.extra.check_linear_operator(op)


@pmp('dofdex', [(0,), (1,), (0, 1), (1, 0)])
def testCorFldDistr(dofdex):
    tgt = ift.UnstructuredDomain(len(dofdex))
    dom = ift.UnstructuredDomain(2)
    op = ift.library.correlated_fields._Distributor(dofdex, dom, tgt)
    ift.extra.check_linear_operator(op)


def metatestMatrixProductOperator(sp, mat_shape, seed, **kwargs):
    with ift.random.Context(seed):
        mat = ift.random.current_rng().standard_normal(mat_shape)
        op = ift.MatrixProductOperator(sp, mat, **kwargs)
        ift.extra.check_linear_operator(op)
        mat = mat + 1j*ift.random.current_rng().standard_normal(mat_shape)
        op = ift.MatrixProductOperator(sp, mat, **kwargs)
        ift.extra.check_linear_operator(op)


@pmp('sp', [ift.RGSpace(10)])
@pmp('spaces', [None, (0,)])
@pmp('seed', [12, 3])
def testMatrixProductOperator_1d(sp, spaces, seed):
    mat_shape = sp.shape * 2
    metatestMatrixProductOperator(sp, mat_shape, seed, spaces=spaces)


@pmp('sp', [ift.DomainTuple.make((ift.RGSpace((2)), ift.RGSpace((10))))])
@pmp('spaces', [(0,), (1,), (0, 1)])
@pmp('seed', [12, 3])
def testMatrixProductOperator_2d_spaces(sp, spaces, seed):
    appl_shape = []
    for sp_idx in spaces:
        appl_shape += sp[sp_idx].shape
    appl_shape = tuple(appl_shape)
    mat_shape = appl_shape * 2
    metatestMatrixProductOperator(sp, mat_shape, seed, spaces=spaces)


@pmp('sp', [ift.RGSpace((2, 10))])
@pmp('seed', [12, 3])
def testMatrixProductOperator_2d_flatten(sp, seed):
    appl_shape = (ift.utilities.my_product(sp.shape),)
    mat_shape = appl_shape * 2
    metatestMatrixProductOperator(sp, mat_shape, seed, flatten=True)


@pmp('seed', [12, 3])
def testPartialExtractor(seed):
    with ift.random.Context(seed):
        tgt = {'a': ift.RGSpace(1), 'b': ift.RGSpace(2)}
        dom = tgt.copy()
        dom['c'] = ift.RGSpace(3)
        dom = ift.MultiDomain.make(dom)
        tgt = ift.MultiDomain.make(tgt)
        op = ift.PartialExtractor(dom, tgt)
        ift.extra.check_linear_operator(op)


@pmp('seed', [12, 3])
def testSlowFieldAdapter(seed):
    dom = {'a': ift.RGSpace(1), 'b': ift.RGSpace(2)}
    op = ift.operators.simple_linear_operators._SlowFieldAdapter(dom, 'a')
    ift.extra.check_linear_operator(op)

@pmp('seed', [12, 3])
def testDiagonalExtractor(seed):
    N = 42
    square_space = ift.RGSpace([N,N])
    op = ift.library.variational_models.DiagonalSelector(square_space)
    ift.extra.check_linear_operator(op)

@pmp('seed', [12, 3])
@pmp("N", [10, 17])
def testLowerTriangularInserter(seed, N):
    square_space = ift.RGSpace([N, N])
    op = ift.library.variational_models.LowerTriangularInserter(square_space)
    ift.extra.check_linear_operator(op)

@pmp('seed', [12, 3])
def test_Multifield2Vector(seed):
    dom = {'a': ift.RGSpace(1), 'b': ift.RGSpace(2)}
    dom = ift.MultiDomain.make(dom)
    op = ift.Multifield2Vector(dom)
    ift.extra.check_linear_operator(op)
