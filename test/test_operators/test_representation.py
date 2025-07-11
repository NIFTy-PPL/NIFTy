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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest

from ..common import list2fixture, setup_function, teardown_function

_h_RG_spaces = [
    ift.RGSpace(7, distances=0.2, harmonic=True),
    ift.RGSpace((12, 46), distances=(.2, .3), harmonic=True)
]
_h_spaces = _h_RG_spaces + [ift.LMSpace(17)]
_p_RG_spaces = [
    ift.RGSpace(19, distances=0.7),
    ift.RGSpace((1, 2, 3, 6), distances=(0.2, 0.25, 0.34, .8))
]
_p_spaces = _p_RG_spaces + [ift.HPSpace(17), ift.GLSpace(8, 13)]
_pow_spaces = [ift.PowerSpace(ift.RGSpace((17, 38), harmonic=True))]

pmp = pytest.mark.parametrize
dtype = list2fixture([np.float64, np.complex128])


def _check_repr(op):
    op.__repr__()


@pmp('sp', _p_RG_spaces)
def testLOSResponse(sp, dtype):
    starts = ift.random.current_rng().standard_normal((len(sp.shape), 10))
    ends = ift.random.current_rng().standard_normal((len(sp.shape), 10))
    sigma_low = 1e-4*ift.random.current_rng().standard_normal(10)
    sigma_ups = 1e-5*ift.random.current_rng().standard_normal(10)
    _check_repr(ift.LOSResponse(sp, starts, ends, sigma_low, sigma_ups))


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testOperatorCombinations(sp, dtype):
    a = ift.DiagonalOperator(ift.Field.from_random(sp, "normal", dtype=dtype))
    b = ift.DiagonalOperator(ift.Field.from_random(sp, "normal", dtype=dtype))
    _check_repr(ift.SandwichOperator.make(a, b))
    _check_repr(a(b))
    _check_repr(a+b)
    _check_repr(a-b)
    _check_repr(a*b)
    _check_repr(a**2)


def testLinearInterpolator():
    sp = ift.RGSpace((10, 8), distances=(0.1, 3.5))
    pos = ift.random.current_rng().random((2, 23))
    pos[0, :] *= 0.9
    pos[1, :] *= 7*3.5
    _check_repr(ift.LinearInterpolator(sp, pos))


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testOperatorAdaptor(sp, dtype):
    op = ift.DiagonalOperator(ift.Field.from_random(sp, "normal", dtype=dtype))
    _check_repr(op.adjoint)
    _check_repr(op.inverse)
    _check_repr(op.inverse.adjoint)


@pmp('sp1', _h_spaces + _p_spaces + _pow_spaces)
@pmp('sp2', _h_spaces + _p_spaces + _pow_spaces)
def testNullOperator(sp1, sp2, dtype):
    op = ift.NullOperator(sp1, sp2)
    ift.extra.check_linear_operator(op, dtype, dtype)
    mdom1 = ift.MultiDomain.make({'a': sp1})
    mdom2 = ift.MultiDomain.make({'b': sp2})
    _check_repr(ift.NullOperator(mdom1, mdom2))
    _check_repr(ift.NullOperator(sp1, mdom2))
    _check_repr(ift.NullOperator(mdom1, sp2))


@pmp('sp', _p_RG_spaces)
def testHarmonicSmoothingOperator(sp, dtype):
    _check_repr(ift.HarmonicSmoothingOperator(sp, 0.1))


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testDOFDistributor(sp, dtype):
    # TODO: Test for DomainTuple
    if sp.size < 4:
        return
    dofdex = np.arange(sp.size).reshape(sp.shape) % 3
    dofdex = ift.Field.from_raw(sp, dofdex)
    _check_repr(ift.DOFDistributor(dofdex))


@pmp('sp', _h_spaces)
def testPPO(sp, dtype):
    _check_repr(ift.PowerDistributor(target=sp))
    ps = ift.PowerSpace(
        sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=False, nbin=3))
    _check_repr(ift.PowerDistributor(target=sp, power_space=ps))
    ps = ift.PowerSpace(
        sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=True, nbin=3))
    _check_repr(ift.PowerDistributor(target=sp, power_space=ps))


@pmp('sp', _h_RG_spaces + _p_RG_spaces)
def testFFT(sp, dtype):
    _check_repr(ift.FFTOperator(sp))
    _check_repr(ift.FFTOperator(sp.get_default_codomain()))


@pmp('sp', _h_RG_spaces + _p_RG_spaces)
def testHartley(sp, dtype):
    _check_repr(ift.HartleyOperator(sp))
    _check_repr(ift.HartleyOperator(sp.get_default_codomain()))


@pmp('sp', _h_spaces)
def testHarmonic(sp, dtype):
    _check_repr(ift.HarmonicTransformOperator(sp))


@pmp('sp', _p_spaces)
def testMask(sp, dtype):
    # Create mask
    f = ift.from_random(sp, 'normal').val
    mask = np.zeros_like(f)
    mask[f > 0] = 1
    mask = ift.Field.from_raw(sp, mask)
    # Test MaskOperator
    _check_repr(ift.MaskOperator(mask))


@pmp('sp', _h_spaces + _p_spaces)
def testDiagonal(sp, dtype):
    op = ift.DiagonalOperator(ift.Field.from_random(sp, "normal", dtype=dtype))
    _check_repr(op)


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testGeometryRemover(sp, dtype):
    _check_repr(ift.GeometryRemover(sp))


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
def testExtractAtIndices(sp, dtype):
    min_ax = np.min(sp.shape)
    n_ax = len(sp.shape)
    inds = (list(range(min_ax//2))+list(range(min_ax//3)), )*n_ax
    _check_repr(ift.ExtractAtIndices(sp, inds))


@pmp('spaces', [0, 1, 2, 3, (0, 1), (0, 2), (0, 1, 2), (0, 2, 3), (1, 3)])
@pmp('wgt', [0, 1, 2, -1])
def testContractionOperator(spaces, wgt, dtype):
    dom = (ift.RGSpace(10), ift.RGSpace(13), ift.GLSpace(5), ift.HPSpace(4))
    _check_repr(ift.ContractionOperator(dom, spaces, wgt))


def testDomainTupleFieldInserter():
    target = ift.DomainTuple.make((ift.UnstructuredDomain([3, 2]),
                                   ift.UnstructuredDomain(7),
                                   ift.RGSpace([4, 22])))
    _check_repr(ift.DomainTupleFieldInserter(target, 1, (5,)))


@pmp('space', [0, 2])
@pmp('factor', [1, 2, 2.7])
@pmp('central', [False, True])
def testZeroPadder(space, factor, dtype, central):
    dom = (ift.RGSpace(10), ift.UnstructuredDomain(13), ift.RGSpace(7, 12),
           ift.HPSpace(4))
    newshape = [int(factor*l) for l in dom[space].shape]
    _check_repr(ift.FieldZeroPadder(dom, newshape, space, central))


@pmp('args', [[ift.RGSpace(
    (13, 52, 40)), (4, 6, 25), None], [ift.RGSpace(
        (128, 128)), (45, 48), 0], [ift.RGSpace(13), (7,), None], [
            (ift.HPSpace(3), ift.RGSpace((12, 24), distances=0.3)), (12, 12), 1
        ]])
def testRegridding(args):
    _check_repr(ift.RegriddingOperator(*args))


@pmp('fdomain', [
    ift.DomainTuple.make((ift.RGSpace(
        (3, 5, 4)), ift.RGSpace((16,), distances=(7.,))),),
    ift.DomainTuple.make(ift.HPSpace(12),)
])
@pmp('domain', [
    ift.DomainTuple.make((ift.RGSpace((2,)), ift.GLSpace(10)),),
    ift.DomainTuple.make(ift.RGSpace((10, 12), distances=(0.1, 1.)),)
])
def testOuter(fdomain, domain):
    f = ift.from_random(fdomain, 'normal')
    _check_repr(ift.OuterProduct(domain, f))


@pmp('sp', _h_spaces + _p_spaces + _pow_spaces)
@pmp('seed', [12, 3])
def testValueInserter(sp, seed):
    with ift.random.Context(seed):
        ind = []
        for ss in sp.shape:
            if ss == 1:
                ind.append(0)
            else:
                ind.append(int(ift.random.current_rng().integers(0, ss - 1)))
        _check_repr(ift.ValueInserter(sp, ind))


@pmp('dom', _h_spaces + _p_spaces)
def testGaussianEnergy(dom):
    _check_repr(ift.GaussianEnergy(domain=dom))


@pmp('dom', _h_spaces + _p_spaces)
def testHamiltonian(dom):
    lh = ift.GaussianEnergy(domain=dom)
    _check_repr(ift.StandardHamiltonian(lh))


@pmp('dom', _h_RG_spaces + _p_RG_spaces)
def testSumOperator(dom):
    op0 = ift.FFTOperator(dom)
    op1 = op0.scale(2)
    _check_repr(op0+op1)
