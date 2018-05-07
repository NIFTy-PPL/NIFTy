import unittest

import nifty4 as ift
import nifty4.nonlinear.nonlinear_operator as nl
import numpy as np
from numpy.testing import assert_allclose


class OneDToOneDTests(unittest.TestCase):
    def make(self):
        space = ift.RGSpace(2)
        self.x = ift.Field(space, val=np.array([2., 5.]))
        self.a = nl.NLOp_var(space)

    @staticmethod
    def takeOp(op, at, out):
        gradient = op.derivative.value(at)
        dom = gradient.domain
        grad1 = gradient(ift.Field(dom, np.array([1., 0.]))).val
        grad2 = gradient(ift.Field(dom, np.array([0., 1.]))).val
        grad = np.array([grad1, grad2])
        assert_allclose(grad, out)

    def test_neg(self):
        self.make()
        self.takeOp(nl.NLOp_neg(self.a), self.x, np.diagflat(-np.ones_like(self.x.val)))

    def test_linear(self):
        self.make()
        self.takeOp(self.a, self.x, np.array([[1., 0.], [0., 1.]]))

    def test_linear2(self):
        self.make()
        self.takeOp(2*self.a, self.x, np.array([[2., 0.], [0., 2.]]))

    def test_quadratic(self):
        self.make()
        self.takeOp(self.a*self.a, self.x, np.diagflat(2*self.x.val))

    def test_const(self):
        self.make()
        self.takeOp(nl.NLOp_const(4) + 0*self.a, self.x, np.zeros((2, 2)))

    def test_exp(self):
        self.make()
        self.takeOp(nl.NLOp_Exp(self.a), self.x, np.diagflat(np.exp(self.x.val)))

    def test_exp_quadratic(self):
        self.make()
        self.takeOp(nl.NLOp_Exp(self.a*self.a), self.x, np.diagflat(2*self.x.val*np.exp((self.x**2).val)))

    def DISABLEDtest_tanh(self):
        self.make()
        self.takeOp(nl.NLOp_Tanh(self.a), self.x, np.diagflat(1. - np.tanh(self.x.val))**2)

    def test_mul(self):
        self.make()
        a1, a2 = self.x.val
        grad = np.zeros((2, 2))
        grad[0, 0] = 3*a1**2+a2**2
        grad[0, 1] = 2*a1*a2
        grad[1, 0] = 2*a1*a2
        grad[1, 1] = a1**2+3*a2**2
        self.takeOp(nl.NLOp_mul(self.a, nl.NLOp_vdot(self.a, self.a), False, True), self.x, grad)


class OneDToZeroDTests(unittest.TestCase):
    def make(self):
        space = ift.RGSpace(2)
        self.x = ift.Field(space, val=np.array([2., 5.]))
        self.a = nl.NLOp_var(space)

    @staticmethod
    def takeOp2D1D(op, at, out):
        gradient = op.derivative.value(at)
        dom = gradient.domain
        grad1 = gradient(ift.Field(dom, np.array([1., 0.]))).val
        grad2 = gradient(ift.Field(dom, np.array([0., 1.]))).val
        grad = np.array([grad1, grad2])
        assert_allclose(grad, out)

    def test_vdot(self):
        self.make()
        self.takeOp2D1D(nl.NLOp_vdot(self.a, self.a), self.x, 2*self.x.val)

    def test_vdot_linear(self):
        self.make()
        self.takeOp2D1D(nl.NLOp_vdot(2*self.a, self.a), self.x, 4*self.x.val)

    def test_vdot_mul(self):
        self.make()
        self.takeOp2D1D(nl.NLOp_vdot(self.a, self.a) * nl.NLOp_vdot(self.a, self.a), self.x, 4*(self.x.vdot(self.x))*self.x.val)


class TwoDToTwoDTests(unittest.TestCase):
    def make(self):
        space = ift.RGSpace((2, 2))
        self.x = ift.Field(space, val=np.array([[2., 5.], [1., 3.]]))
        self.a = nl.NLOp_var(space)

    @staticmethod
    def takeOp(op, at, out):
        gradient = op.derivative.value(at)
        dom = gradient.domain
        grad11 = gradient(ift.Field(dom, np.array([[1., 0.], [0., 0.]]))).val
        grad12 = gradient(ift.Field(dom, np.array([[0., 1.], [0., 0.]]))).val
        grad21 = gradient(ift.Field(dom, np.array([[0., 0.], [1., 0.]]))).val
        grad22 = gradient(ift.Field(dom, np.array([[0., 0.], [0., 1.]]))).val
        grad = np.array([[grad11, grad12], [grad21, grad22]])
        assert_allclose(grad, out)

    @staticmethod
    def identity(diag=None):
        identity = np.zeros((2, 2, 2, 2))
        if diag is None:
            identity[0,0,0,0] = 1.
            identity[0,1,0,1] = 1.
            identity[1,0,1,0] = 1.
            identity[1,1,1,1] = 1.
        else:
            identity[0,0,0,0] = diag[0, 0]
            identity[0,1,0,1] = diag[0, 1]
            identity[1,0,1,0] = diag[1, 0]
            identity[1,1,1,1] = diag[1, 1]
        return identity

    def test_neg(self):
        self.make()
        self.takeOp(nl.NLOp_neg(self.a), self.x, -self.identity())

    def test_linear(self):
        self.make()
        self.takeOp(self.a, self.x, self.identity())

    def test_linear2(self):
        self.make()
        self.takeOp(2*self.a, self.x, 2*self.identity())

    def test_mul(self):
        self.make()
        self.takeOp(self.a*self.a, self.x, 2*self.identity()*self.x.val)

    def test_const(self):
        self.make()
        self.takeOp(nl.NLOp_const(4) + 0*self.a, self.x, np.zeros((2, 2, 2, 2)))

    def test_exp(self):
        self.make()
        self.takeOp(nl.NLOp_Exp(self.a), self.x, self.identity(np.exp(self.x.val)))

    def test_exp_squared(self):
        self.make()
        self.takeOp(nl.NLOp_Exp(self.a*self.a), self.x, self.identity(2*self.x.val*np.exp((self.x**2).val)))

    def DISABLEDtest_tanh(self):
        self.make()
        self.takeOp(nl.NLOp_Tanh(self.a), self.x, np.diagflat(1. - np.tanh(self.x.val))**2)

    def test_nonlinear(self):
        self.make()
        a1, a2 = self.x.val
        print(a1)
        a11, a12, a21, a22 = self.x.val.flatten()
        grad = np.zeros((2, 2, 2, 2))
        # TODO Compute gradient automatically
        grad = np.array([[[[47., 20.], [4., 12.]], [[20., 89.], [10., 30.]]], [[[4., 10.], [41.,  6.]], [[12., 30.], [6., 57.]]]])
        self.takeOp(nl.NLOp_mul(self.a, nl.NLOp_vdot(self.a, self.a), False, True), self.x, grad)
