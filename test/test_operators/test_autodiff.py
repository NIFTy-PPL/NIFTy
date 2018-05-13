import unittest

import nifty4 as ift
import numpy as np
from numpy.testing import assert_allclose


class NonlinearTests(unittest.TestCase):
    def make(self):
        space = ift.RGSpace(2)
        self.x = ift.Field(space, val=np.array([2., 5.]))
        self.a = ift.NLVariable(space)
        self.S = ift.DiagonalOperator(ift.Field(space, 2.))

    @staticmethod
    def takeOp1D1D(op, at, out):
        gradient = op.derivative.eval(at)
        dom = gradient.domain
        grad1 = gradient(ift.Field(dom, np.array([1., 0.]))).val
        grad2 = gradient(ift.Field(dom, np.array([0., 1.]))).val
        grad = np.array([grad1, grad2])
        assert_allclose(grad, out)

    def test_const(self):
        # E = a
        self.make()
        E = self.a
        res = E.eval(self.x).val
        assert_allclose(res, self.x.val)
        self.takeOp1D1D(E, self.x, np.diagflat(np.ones(2)))

    def test_const2(self):
        # E = (2*Id)(a)
        self.make()
        A = ift.NLConstant(ift.Tensor(self.S, 2), (-1, -1))
        E = ift.NLCABF(A, self.a)
        res = E.eval(self.x).val
        Sdiag = self.S(ift.Field.ones(self.S.domain))
        true_res = (Sdiag * self.x).val
        # Test function evaluation
        assert_allclose(res, true_res)
        # Test gradient
        self.takeOp1D1D(E, self.x, np.diagflat(Sdiag.val))

    def test_with_crossterms(self):
        # E = a * |a|**2
        self.make()
        E = ift.NLScalarMul(self.a, ift.NLVdot(self.a, self.a))
        res = E.eval(self.x).val
        res_true = (self.x * self.x.vdot(self.x)).val
        assert_allclose(res, res_true)
        x1, x2 = self.x.val
        self.takeOp1D1D(E, self.x, np.array([[3*x1**2+x2**2, 2*x1*x2], [2*x1*x2, 3*x2**2+x1**2]]))

    def test_priorEnergy(self):
        # E = a^dagger (2*Id)(a)
        self.make()
        A = ift.NLConstant(ift.Tensor(self.S, 2), (-1, -1))
        E = ift.NLApplyForm(ift.NLCABF(A, self.a), self.a)
        res = E.eval(self.x)
        assert_allclose(res, 58)
        gradient = E.derivative.eval(self.x)
        assert_allclose(gradient.val, 4 * self.x.val)
        curv = E.derivative.derivative

        curv = curv.eval(self.x)
        curv_true = 2 * self.S
        assert_allclose((curv-curv_true)(ift.Field.from_random('normal', curv.domain)).val, ift.Field.zeros(curv.domain).val)

    def test_nonlinearity(self):
        # E = exp(a)
        self.make()
        E = ift.NLExp(self.a)
        res = E.eval(self.x).val
        assert_allclose(res, np.exp(self.x.val))
        self.takeOp1D1D(E, self.x, np.diagflat(res))

    def test_nonlinearpriorEnergy(self):
        # E = 0.5 * exp(a)^dagger S exp(a)
        self.make()
        exp_a = ift.NLExp(self.a)
        A = ift.NLConstant(ift.Tensor(0.5 * self.S, 2), (-1, -1))
        E = ift.NLApplyForm(ift.NLCABF(A, exp_a), exp_a)
        res = E.eval(self.x)
        res_true = 0.5 * ift.exp(self.x).vdot(self.S(ift.exp(self.x)))
        assert_allclose(res, res_true)
        gradient = E.derivative.eval(self.x)
        gradient_true = (ift.DiagonalOperator(ift.exp(self.x)) * self.S)(ift.exp(self.x)).val
        assert_allclose(gradient.val, gradient_true)
        curv = E.derivative.derivative
        curv = curv.eval(self.x)
        curv_true = ift.DiagonalOperator(ift.exp(self.x)) * self.S * ift.DiagonalOperator(ift.exp(self.x))
        assert_allclose((curv-curv_true)(ift.Field.from_random('normal', curv.domain)).val, ift.Field.zeros(curv.domain).val)
