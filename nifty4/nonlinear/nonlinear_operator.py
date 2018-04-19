import numpy as np

class NLOp(object):
    def value(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

    @staticmethod
    def _makeOp(thing):
        return thing if isinstance(thing, NLOp) else NLOp_const(thing)

    def __neg__(self):
        return NLOp_neg(self)

    def __add__(self, other):
        return NLOp_add(self, self._makeOp(other))

    def __radd__(self, other):
        return NLOp_add(self._makeOp(other), self)

    def __sub__(self, other):
        return NLOp_add(self, NLOp_neg(self._makeOp(other)))

    def __rsub__(self, other):
        return NLOp_add(NLOp_neg(self._makeOp(other)), self)

    def __mul__(self, other):
        return NLOp_mul(self, self._makeOp(other))

    def __rmul__(self, other):
        return NLOp_mul(self._makeOp(other), self)

class NLOp_const(NLOp):
    def __init__(self, val):
        self._val = val

    def value(self, x):
        return self._val

    def derivative(self, x):
        return 0.

class NLOp_var(NLOp):
    def __init__(self):
        pass

    def value(self, x):
        return x

    def derivative(self, x):
        return 1.

class NLOp_Linop(NLOp):
    def __init__(self, lop, arg):
        self._lop = lop
        self._arg = arg

    def value(self, x):
        return self._lop(self._arg.value(x))

    def derivative(self, x):
        return self._arg.derivative(x)*self._lop.adjoint

class NLOp_add(NLOp):
    def __init__(self, a, b):
        self._a, self._b = a, b

    def value(self,x):
        return self._a.value(x) + self._b.value(x)

    def derivative(self, x):
        return self._a.derivative(x) + self._b.derivative(x)

class NLOp_mul(NLOp):
    def __init__(self, a, b):
        self._a, self._b = a, b

    def value(self,x):
        return self._a.value(x) * self._b.value(x)

    def derivative(self, x):
        return self._b.value(x)*self._a.derivative(x) + self._b.derivative(x)*self._a.value(x)
import nifty4 as ift

class NLOp_vdot(NLOp):
    def __init__(self, a, b):
        self._a, self._b = a, b

    def value(self,x):
        return self._a.value(x).vdot(self._b.value(x))

    def derivative(self, x):
        return (self._a.derivative(x)*ift.ScalarDistributor(self._b.value(x)) +
                self._b.derivative(x)*ift.ScalarDistributor(self._a.value(x)))

class NLOp_Exp(NLOp):
    def __init__(self, var):
        self._var = var

    def value(self, x):
        return ift.exp(self._var.value(x))

    def derivative(self, x):
        return self._var.derivative(x) * ift.exp(self._var.value(x))

class NLOp_Tanh(NLOp):
    def __init__(self, var):
        self._var = var

    def value(self, x):
        return ift.tanh(self._var.value(x))

    def derivative(self, x):
        return self._var.derivative(x)*(1.-ift.tanh(self._var.value(x))**2)

class NLOp_neg(NLOp):
    def __init__(self, var):
        self._var = var

    def value(self, x):
        return -self._var.value(x)

    def derivative(self, x):
        return -self._var.derivative(x)
