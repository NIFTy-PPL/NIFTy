import unittest
from numpy.testing import assert_allclose
import nifty2go as ift
from itertools import product
from test.common import expand


class ResponseOperator_Tests(unittest.TestCase):
    spaces = [ift.RGSpace(128)]

    @expand(product(spaces, [0.,  5., 1.], [0., 1., .33]))
    def test_property(self, space, sigma, exposure):
        op = ift.ResponseOperator(space, sigma=[sigma],
                                  exposure=[exposure])
        if op.domain[0] != space:
            raise TypeError
        if op.unitary:
            raise ValueError

    @expand(product(spaces, [0.,  5., 1.], [0., 1., .33]))
    def test_times_adjoint_times(self, space, sigma, exposure):
        op = ift.ResponseOperator(space, sigma=[sigma],
                                  exposure=[exposure])
        rand1 = ift.Field.from_random('normal', domain=space)
        rand2 = ift.Field.from_random('normal', domain=op.target[0])
        tt1 = rand2.vdot(op.times(rand1))
        tt2 = rand1.vdot(op.adjoint_times(rand2))
        assert_allclose(tt1, tt2)
