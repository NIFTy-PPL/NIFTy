import unittest
from numpy.testing import assert_allclose, assert_equal
import nifty2go as ift
from test.common import generate_spaces
from itertools import product
from test.common import expand


class ComposedOperator_Tests(unittest.TestCase):
    spaces = generate_spaces()

    @expand(product(spaces, spaces))
    def test_times_adjoint_times(self, space1, space2):
        cspace = (space1, space2)
        diag1 = ift.Field.from_random('normal', domain=space1)
        diag2 = ift.Field.from_random('normal', domain=space2)
        op1 = ift.DiagonalOperator(diag1, cspace, spaces=(0,))
        op2 = ift.DiagonalOperator(diag2, cspace, spaces=(1,))

        op = op2*op1

        rand1 = ift.Field.from_random('normal', domain=(space1, space2))
        rand2 = ift.Field.from_random('normal', domain=(space1, space2))

        tt1 = rand2.vdot(op.times(rand1))
        tt2 = rand1.vdot(op.adjoint_times(rand2))
        assert_allclose(tt1, tt2)

    @expand(product(spaces, spaces))
    def test_times_inverse_times(self, space1, space2):
        cspace = (space1, space2)
        diag1 = ift.Field.from_random('normal', domain=space1)
        diag2 = ift.Field.from_random('normal', domain=space2)
        op1 = ift.DiagonalOperator(diag1, cspace, spaces=(0,))
        op2 = ift.DiagonalOperator(diag2, cspace, spaces=(1,))

        op = op2*op1

        rand1 = ift.Field.from_random('normal', domain=(space1, space2))
        tt1 = op.inverse_times(op.times(rand1))

        assert_allclose(ift.dobj.to_global_data(tt1.val),
                        ift.dobj.to_global_data(rand1.val))

    @expand(product(spaces))
    def test_sum(self, space):
        op1 = ift.DiagonalOperator(ift.Field(space, 2.))
        op2 = ift.ScalingOperator(3., space)
        full_op = op1 + op2 - (op2 - op1) + op1 + op1 + op2
        x = ift.Field(space, 1.)
        res = full_op(x)
        assert_equal(isinstance(full_op, ift.DiagonalOperator), True)
        assert_allclose(ift.dobj.to_global_data(res.val), 11.)

    @expand(product(spaces))
    def test_chain(self, space):
        op1 = ift.DiagonalOperator(ift.Field(space, 2.))
        op2 = ift.ScalingOperator(3., space)
        full_op = op1 * op2 * (op2 * op1) * op1 * op1 * op2
        x = ift.Field(space, 1.)
        res = full_op(x)
        assert_equal(isinstance(full_op, ift.DiagonalOperator), True)
        assert_allclose(ift.dobj.to_global_data(res.val), 432.)
