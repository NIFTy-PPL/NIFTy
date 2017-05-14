import unittest

from numpy.testing import assert_equal,\
    assert_allclose,\
    assert_approx_equal

from nifty import Field,\
    RGSpace,\
    PowerSpace,\
    SmoothingOperator

from itertools import product
from test.common import expand

class SmoothingOperator_Tests(unittest.TestCase):
    spaces = [RGSpace(100)]

    @expand(product(spaces, [0., .5, 5.], [True, False]))
    def test_property(self, space, sigma, log_distances):
        op = SmoothingOperator(space, sigma=sigma,
                              log_distances=log_distances)
        if op.domain[0] != space:
            raise TypeError
        if op.unitary != False:
            raise ValueError
        if op.self_adjoint != True:
            raise ValueError
        if op.sigma != sigma:
            raise ValueError
        if op.log_distances != log_distances:
            raise ValueError

    @expand(product(spaces, [0., .5, 5.], [True, False]))
    def test_adjoint_times(self, space, sigma, log_distances):
        op = SmoothingOperator(space, sigma=sigma,
                              log_distances=log_distances)
        rand1 = Field.from_random('normal', domain=space)
        rand2 = Field.from_random('normal', domain=space)
        tt1 = rand1.dot(op.times(rand2))
        tt2 = rand2.dot(op.adjoint_times(rand1))
        assert_approx_equal(tt1, tt2)

    @expand(product(spaces, [0., .5, 5.], [True, False]))
    def test_times(self, space, sigma, log_distances):
        op = SmoothingOperator(space, sigma=sigma,
                              log_distances=log_distances)
        rand1 = Field(space, val=0.)
        rand1.val[0] = 1.
        tt1 = op.times(rand1)
        assert_approx_equal(1, tt1.sum())

    @expand(product(spaces, [0., .5, 5.], [True, False]))
    def test_inverse_adjoint_times(self, space, sigma, log_distances):
        op = SmoothingOperator(space, sigma=sigma,
                              log_distances=log_distances)
        rand1 = Field.from_random('normal', domain=space)
        rand2 = Field.from_random('normal', domain=space)
        tt1 = rand1.dot(op.inverse_times(rand2))
        tt2 = rand2.dot(op.inverse_adjoint_times(rand1))
        assert_approx_equal(tt1, tt2)
