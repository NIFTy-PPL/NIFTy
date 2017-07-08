import unittest

from numpy.testing import assert_approx_equal

from nifty import Field,\
    RGSpace,\
    ResponseOperator

from itertools import product
from test.common import expand

class ResponseOperator_Tests(unittest.TestCase):
    spaces = [RGSpace(100)]

    @expand(product(spaces, [0.,  5., 1.], [0., 1., .33] ))
    def test_property(self, space, sigma, exposure):
        op = ResponseOperator(space, sigma=[sigma],
                              exposure=[exposure])
        if op.domain[0] != space:
            raise TypeError
        if op.unitary != False:
            raise ValueError

    @expand(product(spaces, [0.,  5., 1.], [0., 1., .33] ))
    def test_times_adjoint_times(self, space, sigma, exposure):
        op = ResponseOperator(space, sigma=[sigma],
                              exposure=[exposure])
        rand1 = Field.from_random('normal', domain=space)
        rand2 = Field.from_random('normal', domain=op.target[0])
        tt1 = rand2.vdot(op.times(rand1))
        tt2 = rand1.vdot(op.adjoint_times(rand2))
        assert_approx_equal(tt1, tt2)
