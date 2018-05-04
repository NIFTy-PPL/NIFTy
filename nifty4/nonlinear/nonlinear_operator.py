import numpy as np
import nifty4 as ift


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

    @property
    def derivative(self):
        return 0.


class NLOp_var(NLOp):
    def __init__(self, domain):
        self._domain = domain

    def value(self, x):
        return x

    @property
    def derivative(self):
        return NLOp_const(ift.ScalingOperator(1., self._domain))

    @property
    def derivative_field(self):
        return NLOp_const(ift.Field.ones(self._domain))


class NLOp_Linop(NLOp):
    def __init__(self, lop, arg):
        self._lop = lop
        self._arg = arg

    def value(self, x):
        return self._lop(self._arg.value(x))

    @property
    def derivative(self):
        return self._arg.derivative * self._lop.adjoint


class NLOp_add(NLOp):
    def __init__(self, a, b):
        self._a, self._b = a, b

    def value(self, x):
        return self._a.value(x) + self._b.value(x)

    @property
    def derivative(self):
        return self._a.derivative + self._b.derivative


class NLOp_mul(NLOp):
    def __init__(self, a, b, a_number=False, b_number=False):
        self._a, self._b = a, b
        self._a_number = a_number
        self._b_number = b_number

    def value(self, x):
        a = self._a.value(x)
        b = self._b.value(x)
        if isinstance(a, ift.Field) and len(a.domain) == 0:
            a = a.to_global_data()[()]
        if isinstance(b, ift.Field) and len(b.domain) == 0:
            b = b.to_global_data()[()]
        return a * b

    @property
    def derivative(self):
        if self._a_number:
            A = NLOp_outer(self._b, self._a.derivative)
        else:
            A = self._b * self._a.derivative

        if self._b_number:
            B = NLOp_outer(self._a, self._b.derivative)
        else:
            B = self._b.derivative * self._a

        return A + B


class NLOp_outer(NLOp):
    def __init__(self, a, b):
        self._a, self._b = a, b
        assert isinstance(self._b, NLOp_row)

    def value(self, x):
        return ift.OuterOperator(self._a.value(x), self._b.value(x))


class NLOp_row(NLOp):
    def __init__(self, row):
        self._row = row

    def value(self, x):
        return ift.RowOperator(self._row.value(x))


class NLOp_vdot(NLOp):
    """ Supports only variables as inputs so far.
    """
    def __init__(self, a, b):
        self._a, self._b = a, b

    def value(self, x):
        dom = ift.DomainTuple.make(())
        return ift.Field.from_global_data(dom, self._a.value(x).vdot(self._b.value(x)))

    @property
    def derivative(self):
        return NLOp_row(self._a.derivative_field * self._b + self._b.derivative_field * self._a)


class NLOp_Exp(NLOp):
    def __init__(self, var):
        self._var = var

    def value(self, x):
        return ift.exp(self._var.value(x))

    @property
    def derivative(self):
        return self._var.derivative * NLOp_Exp(self._var)


class NLOp_Tanh(NLOp):
    def __init__(self, var):
        self._var = var

    def value(self, x):
        return ift.tanh(self._var.value(x))

    def derivative(self, x):
        return self._var.derivative(x) * (1. - ift.tanh(self._var.value(x))**2)


class NLOp_PositiveTanh(NLOp):
    def __init__(self, var):
        self._var = var

    def value(self, x):
        return 0.5 * ift.tanh(0.5 * self._var.value(x))

    def derivative(self, x):
        return 0.5 * self._var.derivative(x) * \
            (1. - ift.tanh(self._var.value(x))**2)


class NLOp_neg(NLOp):
    def __init__(self, var):
        self._var = var

    def value(self, x):
        return -self._var.value(x)

    def derivative(self, x):
        return -self._var.derivative(x)
