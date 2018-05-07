import unittest
from test.common import expand

import nifty4 as ift
import nifty4.nonlinear.nonlinear_operator as nl
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal, assert_raises

space = ift.RGSpace(2)
a = nl.NLOp_var(space)


class OneDSpaceToOneDSpaceTests(unittest.TestCase):
    @staticmethod
    def takeOp(op, at, out):
        gradient = op.derivative.value(at)
        dom = gradient.domain
        grad1 = gradient(ift.Field(dom, np.array([1., 0.]))).val
        grad2 = gradient(ift.Field(dom, np.array([0., 1.]))).val
        grad = np.array([grad1, grad2])
        np.testing.assert_allclose(grad, out)

    def test_neg(self):
        x = ift.Field(space, val=np.array([2., 5.]))
        self.takeOp(nl.NLOp_neg(a), x, np.diagflat(-np.ones_like(x.val)))

    def test_linear(self):
        x = ift.Field(space, val=np.array([2., 5.]))
        self.takeOp(a, x, np.array([[1., 0.], [0., 1.]]))

    def test_linear2(self):
        x = ift.Field(space, val=np.array([2., 5.]))
        self.takeOp(2*a, x, np.array([[2., 0.], [0., 2.]]))

    def test_quadratic(self):
        x = ift.Field(space, val=np.array([2., 5.]))
        self.takeOp(a*a, x, np.diagflat(2*x.val))

    def test_const(self):
        x = ift.Field(space, val=np.array([2., 5.]))
        self.takeOp(nl.NLOp_const(4) + 0*a, x, np.zeros((2, 2)))

    def test_exp(self):
        x = ift.Field(space, val=np.array([2., 5.]))
        self.takeOp(nl.NLOp_Exp(a), x, np.diagflat(np.exp(x.val)))

    def test_exp_quadratic(self):
        x = ift.Field(space, val=np.array([2., 5.]))
        self.takeOp(nl.NLOp_Exp(a*a), x, np.diagflat(2*x.val*np.exp((x**2).val)))

    def test_tanh(self):
        x = ift.Field(space, val=np.array([2., 5.]))
        self.takeOp(nl.NLOp_Tanh(a), x, np.diagflat(1. - np.tanh(x.val))**2)

    def test_mul(self):
        x = ift.Field(space, val=np.array([2., 5.]))
        a1, a2 = x.val
        grad = np.zeros((2, 2))
        grad[0, 0] = 3*a1**2+a2**2
        grad[0, 1] = 2*a1*a2
        grad[1, 0] = 2*a1*a2
        grad[1, 1] = a1**2+3*a2**2
        self.takeOp(nl.NLOp_mul(a, nl.NLOp_vdot(a, a), False, True), x, grad)
