import numpy as np
import nifty4 as ift
import nifty4.nonlinear.nonlinear_operator as nl

space = ift.RGSpace(2)
a = nl.NLOp_var(space)


def takeOp(op, at, out):
    print('E(x) = ', op.value(at))
    print('dE/dx = ', op.derivative)
    gradient = op.derivative.value(at)
    print('dE/dx|(2,5) = ', gradient)
    dom = gradient.domain
    grad1 = gradient(ift.Field(dom, np.array([1, 0]))).val
    grad2 = gradient(ift.Field(dom, np.array([0, 1]))).val
    grad = np.array([grad1, grad2])
    print('dE/dx|(2,5) (1,0) = ', grad[0])
    print('dE/dx|(2,5) (0, 1) = ', grad[1])
    print()
    np.testing.assert_allclose(grad, out)


# space -> space
x = ift.Field(space, val=np.array([2, 5]))
takeOp(a, x, np.array([[1, 0], [0, 1]]))
takeOp(2*a, x, np.array([[2, 0], [0, 2]]))
takeOp(a*a, x, np.diagflat(2*x.val))
takeOp(nl.NLOp_const(4) + 0*a, x, np.zeros((2, 2)))


def takeOp2D1D(op, at, out):
    print('E(x) = ', op.value(at))
    print('dE/dx = ', op.derivative)
    gradient = op.derivative.value(at)
    print('dE/dx|(2,5) = ', gradient)
    dom = gradient.domain
    grad = op.derivative.value(x)(ift.Field.full(dom, 1.)).val
    print('dE/dx|(2,5) (1,1) = ', grad)
    print()
    np.testing.assert_allclose(grad, out)


# space -> float
takeOp2D1D(nl.NLOp_vdot(a, a), x, 2*x.val)
takeOp2D1D(nl.NLOp_vdot(a, a) * nl.NLOp_vdot(a, a), x, 4*(x.vdot(x))*x.val)
