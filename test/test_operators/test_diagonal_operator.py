from __future__ import division
import unittest
from numpy.testing import assert_equal, assert_allclose
import nifty2go as ift
from test.common import generate_spaces
from itertools import product
from test.common import expand


class DiagonalOperator_Tests(unittest.TestCase):
    spaces = generate_spaces()

    @expand(product(spaces))
    def test_property(self, space):
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        if D.domain[0] != space:
            raise TypeError
        if D.unitary:
            raise TypeError
        if not D.self_adjoint:
            raise TypeError

    @expand(product(spaces))
    def test_times_adjoint(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        rand2 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt1 = rand1.vdot(D.times(rand2))
        tt2 = rand2.vdot(D.times(rand1))
        assert_allclose(tt1, tt2)

    @expand(product(spaces))
    def test_times_inverse(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt1 = D.times(D.inverse_times(rand1))
        assert_allclose(ift.dobj.to_global_data(rand1.val),
                        ift.dobj.to_global_data(tt1.val))

    @expand(product(spaces))
    def test_times(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt = D.times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces))
    def test_adjoint_times(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt = D.adjoint_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces))
    def test_inverse_times(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt = D.inverse_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces))
    def test_adjoint_inverse_times(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt = D.adjoint_inverse_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces))
    def test_diagonal(self, space):
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        diag_op = D.diagonal()
        assert_allclose(ift.dobj.to_global_data(diag.val),
                        ift.dobj.to_global_data(diag_op.val))
