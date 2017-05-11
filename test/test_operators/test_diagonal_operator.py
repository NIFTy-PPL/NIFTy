import unittest

import numpy as np
from numpy.testing import assert_equal,\
    assert_allclose,\
    assert_approx_equal

from nifty import Field,\
    DiagonalOperator

from test.common import generate_spaces

from itertools import product
from test.common import expand

class DiagonalOperator_Tests(unittest.TestCase):
    spaces = generate_spaces()

    @expand(product(spaces, [True, False], [True, False]))
    def test_property(self, space, bare, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag)
        if D.domain[0] != space:
            raise TypeError
        if D.unitary != False:
            raise TypeError
        if D.self_adjoint != True:
            raise TypeError

    @expand(product(spaces, [True, False], [True, False]))
    def test_times_adjoint(self, space, bare, copy):
        rand1 = Field.from_random('normal', domain=space)
        rand2 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, bare=bare, copy=copy)
        tt1 = rand1.dot(D.times(rand2))
        tt2 = rand2.dot(D.times(rand1))
        assert_approx_equal(tt1, tt2)

    @expand(product(spaces, [True, False], [True, False]))
    def test_times_inverse(self, space, bare, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, bare=bare, copy=copy)
        tt1 = D.times(D.inverse_times(rand1))
        assert_allclose(rand1.val.get_full_data(), tt1.val.get_full_data())

    @expand(product(spaces, [True, False], [True, False]))
    def test_times(self, space, bare, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, bare=bare, copy=copy)
        tt = D.times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces, [True, False], [True, False]))
    def test_adjoint_times(self, space, bare, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, bare=bare, copy=copy)
        tt = D.adjoint_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces, [True, False], [True, False]))
    def test_inverse_times(self, space, bare, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, bare=bare, copy=copy)
        tt = D.inverse_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces, [True, False], [True, False]))
    def test_adjoint_inverse_times(self, space, bare, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, bare=bare, copy=copy)
        tt = D.adjoint_inverse_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces, [True, False]))
    def test_diagonal(self, space, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, copy=copy)
        diag_op = D.diagonal()
        assert_allclose(diag.val.get_full_data(), diag_op.val.get_full_data())

    @expand(product(spaces, [True, False]))
    def test_inverse(self, space, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, copy=copy)
        diag_op = D.inverse_diagonal()
        assert_allclose(1./diag.val.get_full_data(), diag_op.val.get_full_data())

    @expand(product(spaces, [True, False]))
    def test_trace(self, space, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, copy=copy)
        trace_op = D.trace()
        assert_allclose(trace_op, np.sum(diag.val.get_full_data()))

    @expand(product(spaces, [True, False]))
    def test_inverse_trace(self, space, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, copy=copy)
        trace_op = D.inverse_trace()
        assert_allclose(trace_op, np.sum(1./diag.val.get_full_data()))

    @expand(product(spaces, [True, False]))
    def test_trace_log(self, space, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, copy=copy)
        trace_log = D.trace_log()
        assert_allclose(trace_log, np.log(np.sum(diag.val.get_full_data())))

    @expand(product(spaces, [True, False]))
    def test_determinant(self, space, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, bare=bare, copy=copy)
        det = D.determinant()
        assert_allclose(det, np.prod(diag.val.get_full_data()))

    @expand(product(spaces, [True, False], [True, False]))
    def test_inverse_determinant(self, space, bare, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, bare=bare, copy=copy)
        inv_det = D.inverse_determinant()
        assert_allclose(inv_det, 1./D.determinant())

    @expand(product(spaces, [True, False], [True, False]))
    def test_log_determinant(self, space, bare, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(space, diagonal=diag, bare=bare, copy=copy)
        log_det = D.log_determinant()
        assert_allclose(log_det, np.log(D.determinant()))