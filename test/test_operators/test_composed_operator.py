import unittest
from numpy.testing import assert_allclose
from nifty2go import Field, DiagonalOperator, ComposedOperator
from test.common import generate_spaces
from itertools import product
from test.common import expand
from nifty2go.dobj import to_ndarray as to_np


class ComposedOperator_Tests(unittest.TestCase):
    spaces = generate_spaces()

    @expand(product(spaces, spaces))
    def test_times_adjoint_times(self, space1, space2):
        cspace = (space1, space2)
        diag1 = Field.from_random('normal', domain=space1)
        diag2 = Field.from_random('normal', domain=space2)
        op1 = DiagonalOperator(diag1, cspace, spaces=(0,))
        op2 = DiagonalOperator(diag2, cspace, spaces=(1,))

        op = ComposedOperator((op1, op2))

        rand1 = Field.from_random('normal', domain=(space1, space2))
        rand2 = Field.from_random('normal', domain=(space1, space2))

        tt1 = rand2.vdot(op.times(rand1))
        tt2 = rand1.vdot(op.adjoint_times(rand2))
        assert_allclose(tt1, tt2)

    @expand(product(spaces, spaces))
    def test_times_inverse_times(self, space1, space2):
        cspace = (space1, space2)
        diag1 = Field.from_random('normal', domain=space1)
        diag2 = Field.from_random('normal', domain=space2)
        op1 = DiagonalOperator(diag1, cspace, spaces=(0,))
        op2 = DiagonalOperator(diag2, cspace, spaces=(1,))

        op = ComposedOperator((op1, op2))

        rand1 = Field.from_random('normal', domain=(space1, space2))
        tt1 = op.inverse_times(op.times(rand1))

        assert_allclose(to_np(tt1.val), to_np(rand1.val))
