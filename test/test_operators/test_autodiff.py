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
        print(op.derivative)
        gradient = op.derivative.eval(at).output
        dom = gradient.domain
        grad1 = gradient(ift.Field(dom, np.array([1., 0.]))).val
        grad2 = gradient(ift.Field(dom, np.array([0., 1.]))).val
        grad = np.array([grad1, grad2])
        assert_allclose(grad, out)

    def test_const(self):
        self.make()
        E = self.a
        res = E.eval(self.x).output.val
        assert_allclose(res, self.x.val)
        self.takeOp1D1D(E, self.x, np.diagflat(np.ones(2)))

    def test_const2(self):
        self.make()
        A = ift.NLConstant(ift.Tensor((-1, -1), self.S))
        E = ift.NLContract(A, self.a, 1)
        res = E.eval(self.x).output.val
        Sdiag = self.S(ift.Field.ones(self.S.domain))
        true_res = (Sdiag * self.x).val
        # Test function evaluation
        assert_allclose(res, true_res)
        # Test gradient
        self.takeOp1D1D(E, self.x, np.diagflat(Sdiag.val))

    def test_priorEnergy(self):
        self.make()
        A = ift.NLConstant(ift.Tensor((-1, -1), self.S))
        E = ift.NLContract(ift.NLContract(A, self.a, 1), self.a, 0)
        res = E.eval(self.x).output
        assert_allclose(res, 58)
