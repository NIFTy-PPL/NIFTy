import unittest
from numpy.testing import assert_allclose
import nifty2go as ift
from itertools import product
from test.common import expand


class ResponseOperator_Tests(unittest.TestCase):
    spaces = [ift.RGSpace(128), ift.GLSpace(nlat=37)]

    @expand(product(spaces, [0.,  5., 1.], [0., 1., .33]))
    def test_property(self, space, sigma, sensitivity):
        op = ift.ResponseOperator(space, sigma=[sigma],
                                  sensitivity=[sensitivity])
        if op.domain[0] != space:
            raise TypeError

    @expand(product(spaces, [0.,  5., 1.], [0., 1., .33]))
    def test_times_adjoint_times(self, space, sigma, sensitivity):
        if not isinstance(space, ift.RGSpace):  # no smoothing supported
            sigma = 0.
        op = ift.ResponseOperator(space, sigma=[sigma],
                                  sensitivity=[sensitivity])
        rand1 = ift.Field.from_random('normal', domain=space)
        rand2 = ift.Field.from_random('normal', domain=op.target[0])
        tt1 = rand2.vdot(op.times(rand1))
        tt2 = rand1.vdot(op.adjoint_times(rand2))
        assert_allclose(tt1, tt2)
