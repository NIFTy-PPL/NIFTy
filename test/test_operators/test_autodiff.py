import unittest

import nifty4 as ift
import numpy as np
from numpy.testing import assert_allclose


class NonlinearTests(unittest.TestCase):
    def make(self):
        space = ift.RGSpace(2)
        self.x = ift.from_global_data(space, np.array([2., 5.]))
        self.a = ift.SymbolicVariable(space)
        self.S = ift.DiagonalOperator(ift.Field(space, 2.))

    @staticmethod
    def takeOp1D1D(op, at, out):
        gradient = op.derivative.eval(at)
        dom = gradient.domain
        grad1 = gradient(ift.from_global_data(dom, np.array([1., 0.]))).to_global_data()
        grad2 = gradient(ift.from_global_data(dom, np.array([0., 1.]))).to_global_data()
        grad = np.array([grad1, grad2])
        assert_allclose(grad, out)

    def test_const(self):
        # E = a
        self.make()
        E = self.a
        res = E.eval(self.x).local_data
        assert_allclose(res, self.x.local_data)
        self.takeOp1D1D(E, self.x, np.diagflat(np.ones(2)))

    def test_const2(self):
        # E = (2*Id)(a)
        self.make()
        A = ift.SymbolicConstant(ift.Tensor(self.S, 2), (-1, -1))
        E = ift.SymbolicCABF(A, self.a)
        res = E.eval(self.x).local_data
        Sdiag = self.S(ift.Field.full(self.S.domain, 1.))
        true_res = (Sdiag * self.x).local_data
        # Test function evaluation
        assert_allclose(res, true_res)
        # Test gradient
        self.takeOp1D1D(E, self.x, np.diagflat(Sdiag.to_global_data()))

    def test_with_crossterms(self):
        # E = a * |a|**2
        self.make()
        E = ift.SymbolicScalarMul(self.a, ift.SymbolicVdot(self.a, self.a))
        res = E.eval(self.x).local_data
        res_true = (self.x * self.x.vdot(self.x)).local_data
        assert_allclose(res, res_true)
        x1, x2 = self.x.to_global_data()
        self.takeOp1D1D(E, self.x, np.array([[3*x1**2+x2**2, 2*x1*x2], [2*x1*x2, 3*x2**2+x1**2]]))

    def test_priorEnergy(self):
        # E = a^dagger (2*Id)(a)
        self.make()
        A = ift.SymbolicConstant(ift.Tensor(self.S, 2), (-1, -1))
        E = ift.SymbolicApplyForm(ift.SymbolicCABF(A, self.a), self.a)
        res = E.eval(self.x)
        assert_allclose(res, 58)
        gradient = E.derivative.eval(self.x)
        assert_allclose(gradient.local_data, 4 * self.x.local_data)
        curv = E.derivative.derivative

        curv = curv.eval(self.x)
        curv_true = 2 * self.S
        assert_allclose((curv-curv_true)(ift.Field.from_random('normal', curv.domain)).local_data, ift.Field.full(curv.domain, 0.).local_data)

    def test_nonlinearity(self):
        # E = exp(a)
        self.make()
        E = ift.SymbolicExp(self.a)
        res = E.eval(self.x).to_global_data()
        assert_allclose(res, np.exp(self.x.to_global_data()))
        self.takeOp1D1D(E, self.x, np.diagflat(res))

    def test_nonlinearpriorEnergy(self):
        # E = 0.5 * exp(a)^dagger S exp(a)
        self.make()
        exp_a = ift.SymbolicExp(self.a)
        A = ift.SymbolicConstant(ift.Tensor(0.5 * self.S, 2), (-1, -1))
        E = ift.SymbolicApplyForm(ift.SymbolicCABF(A, exp_a), exp_a)
        res = E.eval(self.x)
        res_true = 0.5 * ift.exp(self.x).vdot(self.S(ift.exp(self.x)))
        assert_allclose(res, res_true)
        gradient = E.derivative.eval(self.x)
        gradient_true = (ift.DiagonalOperator(ift.exp(self.x)) * self.S)(ift.exp(self.x)).local_data
        assert_allclose(gradient.local_data, gradient_true)
        curv = E.derivative.derivative
        curv = curv.eval(self.x)
        curv_true = ift.DiagonalOperator(ift.exp(self.x)) * self.S * ift.DiagonalOperator(ift.exp(self.x))
        assert_allclose((curv-curv_true)(ift.Field.from_random('normal', curv.domain)).local_data, ift.Field.full(curv.domain, 0.).local_data)

    def test_quad(self):
        self.make()
        S = ift.SymbolicConstant(ift.Tensor(self.S, 2), (1, -1))
        E = ift.SymbolicQuad(ift.SymbolicCABF(S, self.a))
        res = E.eval(self.x)
        assert_allclose(res, 0.5 * self.x.vdot((self.S.adjoint*self.S)(self.x)))
        gradient = E.derivative.eval(self.x)
        assert_allclose(gradient.local_data, (self.S.adjoint*self.S)(self.x).local_data)
        curv = E.curvature
        curv = curv.eval(self.x)
        curv_true = self.S.adjoint*self.S
        assert_allclose((curv-curv_true)(ift.Field.from_random('normal', curv.domain)).local_data, ift.Field.full(curv.domain, 0.).local_data)
