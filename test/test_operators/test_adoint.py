import unittest
import nifty2go as ift
import numpy as np
from itertools import product
from test.common import expand
from numpy.testing import assert_allclose


def _check_adjointness(op, dtype=np.float64):
    f1 = ift.Field.from_random("normal",domain=op.domain, dtype=dtype)
    f2 = ift.Field.from_random("normal",domain=op.target, dtype=dtype)
    assert_allclose(f1.vdot(op.adjoint_times(f2)), op.times(f1).vdot(f2),
                    rtol=1e-8)

_harmonic_spaces = [ ift.RGSpace(7, distances=0.2, harmonic=True),
                     ift.RGSpace((12,46), distances=(0.2, 0.3), harmonic=True),
                     ift.LMSpace(17) ]

_position_spaces = [ ift.RGSpace(19, distances=0.7),
                     ift.RGSpace((1,2,3,6), distances=(0.2,0.25,0.34,0.8)),
                     ift.HPSpace(17),
                     ift.GLSpace(8,13) ]

class Adjointness_Tests(unittest.TestCase):
    @expand(product(_harmonic_spaces, [np.float64, np.complex128]))
    def testPPO(self, sp, dtype):
        op = ift.PowerProjectionOperator(sp)
        _check_adjointness(op, dtype)
        ps = ift.PowerSpace(sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=False, nbin=3))
        op = ift.PowerProjectionOperator(sp, ps)
        _check_adjointness(op, dtype)
        ps = ift.PowerSpace(sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=True, nbin=3))
        op = ift.PowerProjectionOperator(sp, ps)
        _check_adjointness(op, dtype)

    @expand(product(_harmonic_spaces+_position_spaces, [np.float64, np.complex128]))
    def testFFT(self, sp, dtype):
        op = ift.FFTOperator(sp)
        _check_adjointness(op, dtype)
