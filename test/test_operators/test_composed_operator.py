import unittest

from numpy.testing import assert_equal,\
    assert_allclose,\
    assert_approx_equal

from nifty import Field,\
    DiagonalOperator,\
    ComposedOperator

from test.common import generate_spaces

from itertools import product
from test.common import expand

class ComposedOperator_Tests(unittest.TestCase):
    spaces = generate_spaces()

    @expand(product([spaces, spaces]))
    def test_property(self, space1, space2):
        rand1 = Field.from_random('normal', domain=space1)
        rand2 = Field.from_random('normal', domain=space2)
        op1 = DiagonalOperator(space1, diagonal=rand1)
        op2 = DiagonalOperator(space2, diagonal=rand2)
        op = ComposedOperator((op1, op2))
        if op.domain != (op1.domain, op2.domain):
            raise TypeError
        if op.unitary != False:
            raise ValueError

    @expand(product([spaces, spaces]))
    def test_times_adjoint_times(self, space1, space2):
        diag1 = Field.from_random('normal', domain=space1)
        diag2 = Field.from_random('normal', domain=space2)
        op1 = DiagonalOperator(space1, diagonal=diag1)
        op2 = DiagonalOperator(space2, diagonal=diag2)

        op = ComposedOperator((op1, op2))

        rand1 = Field.from_random('normal', domain=(space1,space2))
        rand2 = Field.from_random('normal', domain=(space1,space2))

        tt1 = rand2.dot(op.times(rand1))
        tt2 = rand1.dot(op.adjoint_times(rand2))
        assert_approx_equal(tt1, tt2)

    @expand(product([spaces, spaces]))
    def test_times_inverse_times(self, space1, space2):
        diag1 = Field.from_random('normal', domain=space1)
        diag2 = Field.from_random('normal', domain=space2)
        op1 = DiagonalOperator(space1, diagonal=diag1)
        op2 = DiagonalOperator(space2, diagonal=diag2)

        op = ComposedOperator((op1, op2))

        rand1 = Field.from_random('normal', domain=(space1, space2))
        tt1 = op.inverse_times(op.times(rand1))

        assert_allclose(tt1.val.get_full_data(),
                        rand1.val.get_full_data())

