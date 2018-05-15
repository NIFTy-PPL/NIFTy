from ..extra.operator_tests import consistency_check
from ..operators import OuterOperator, RowOperator, SandwichOperator
from .add import SymbolicAdd
from .constant import SymbolicZero
from .tensor import SymbolicTensor


class SymbolicChainLinOps(SymbolicTensor):
    def __init__(self, op1, op2):
        assert op1.rank == 2 and op2.rank == 2
        assert op1.indices[1] == - op2.indices[0]
        self._indices = (op1.indices[0], op2.indices[1])
        self._op1 = op1
        self._op2 = op2

    def __str__(self):
        if isinstance(self._op1, SymbolicZero) or isinstance(self._op2, SymbolicZero):
            return 'Zero'
        return '({} {})'.format(self._op1, self._op2)

    def eval(self, x):
        if isinstance(self._op1, SymbolicZero) or isinstance(self._op2, SymbolicZero):
            return 0.
        A = self._op1.eval(x)
        B = self._op2.eval(x)
        return A * B

    @property
    def derivative(self):
        raise NotImplementedError


class SymbolicSandwich(SymbolicTensor):
    def __init__(self, bun):
        assert bun.rank == 2
        self._indices = 2 * (bun.indices[1],)
        self._bun = bun

    def __str__(self):
        return 'Sandwich({})'.format(self._bun)

    def eval(self, x):
        return SandwichOperator(self._bun.eval(x))

    @property
    def derivative(self):
        raise NotImplementedError


class SymbolicChainLinOps11(SymbolicTensor):
    # op1.indices = (-1, -1)
    # op2.indices = (1, -1)
    # contract first index of op1 with first index of op2
    # Assuming op1 is real
    def __init__(self, op1, op2):
        assert op1.rank == 2 and op2.rank == 2
        assert op1.indices[0] == - op2.indices[0]
        self._indices = (op1.indices[1], op2.indices[1])
        self._op1 = op1
        self._op2 = op2

    def __str__(self):
        return '{} _11_ {}'.format(self._op1, self._op2)

    def eval(self, x):
        A = self._op2.eval(x)
        B = self._op1.eval(x)
        if (isinstance(A, float) and A == 0.) or (isinstance(B, float) and B == 0.):
            return 0.
        return A.adjoint * B

    @property
    def derivative(self):
        raise NotImplementedError


class SymbolicCABF(SymbolicTensor):
    # CABF = Contract All But First
    def __init__(self, nltensor, *nlvectors):
        self._indices = nltensor.indices[0:1]
        self._args = nlvectors
        self._nltensor = nltensor
        assert len(self._nltensor.indices) == len(self._args) + 1
        for ii in range(len(self._args)):
            assert self._args[ii].indices[0] == - self._nltensor.indices[ii + 1]

    def __str__(self):
        s = '{} '.format(self._nltensor)
        for vv in self._args:
            s += '{}, '.format(vv)
        s = s[:-2]
        # s += ')'
        return s

    def eval(self, x):
        if len(self._args) == 1:
            vector = self._args[0].eval(x)
            operator = self._nltensor.eval(x)
            return operator(vector)
        raise NotImplementedError

    @property
    def derivative(self):
        if isinstance(self._nltensor.derivative, SymbolicZero) and len(self._args) == 1:
            return SymbolicChainLinOps(self._nltensor, self._args[0].derivative)
        else:
            # TODO Implement more general case
            raise NotImplementedError


class SymbolicCABL(SymbolicTensor):
    # CABL = Contract All But Last
    def __init__(self, nltensor, *nlvectors):
        self._indices = nltensor.indices[-1:]
        self._args = nlvectors
        self._nltensor = nltensor
        assert len(self._nltensor.indices) == len(self._args) + 1
        for ii in range(len(self._args)):
            assert self._args[ii].indices[0] == - self._nltensor.indices[ii]

    def __str__(self):
        s = '{}CABL('.format(self._nltensor)
        for vv in self._args:
            s += '{}, '.format(vv)
        s = s[:-2]
        s += ')'
        return s

    def eval(self, x):
        if isinstance(self._nltensor, SymbolicZero) or self._nltensor.eval(x) == 0.:
            return 0.
        if len(self._args) == 1:
            nlvector = self._args[0]
            vector = nlvector.eval(x)
            operator = self._nltensor.eval(x)
            return operator.adjoint(vector).conjugate()
        raise NotImplementedError

    @property
    def derivative(self):
        # FIXME This is not the general case!!!
        assert len(self._args) == 1
        return SymbolicChainLinOps11(self._nltensor, self._args[0].derivative)


class SymbolicVdot(SymbolicTensor):
    def __init__(self, vector1, vector2):
        assert vector1.indices == vector2.indices
        assert vector1.indices in [(1,), (-1,)]
        self._vector1 = vector1
        self._vector2 = vector2
        self._indices = ()

    def __str__(self):
        return '({}).vdot({})'.format(self._vector1, self._vector2)

    def eval(self, x):
        return self._vector1.eval(x).vdot(self._vector2.eval(x))

    @property
    def derivative(self):
        A = SymbolicCABL(self._vector1.derivative, self._vector2.adjoint)
        B = SymbolicCABL(self._vector2.derivative, self._vector1.adjoint)
        return SymbolicAdd(A, B)


class SymbolicQuad(SymbolicTensor):
    def __init__(self, thing):
        assert thing.rank == 1
        self._thing = thing
        self._indices = ()

    def __str__(self):
        return '0.5 * |{}|**2'.format(self._thing)

    def eval(self, x):
        a = self._thing.eval(x)
        return 0.5 * a.vdot(a)

    @property
    def derivative(self):
        return SymbolicCABL(self._thing.derivative, self._thing.adjoint)

    @property
    def curvature(self):
        # This is Jakob's curvature
        return SymbolicSandwich(self._thing.derivative)


class SymbolicOuterProd(SymbolicTensor):
    def __init__(self, snd, fst):
        """ Computes A = fst*snd """
        assert snd.indices == (-1,)
        assert fst.indices in [(1,), (-1,)]
        self._snd = snd
        self._fst = fst
        self._indices = self._fst.indices + self._snd.indices

    def __str__(self):
        return '{} outer {}'.format(self._snd, self._fst)

    def eval(self, x):
        if isinstance(self._snd, SymbolicZero) or isinstance(self._fst, SymbolicZero):
            return 0.
        op = OuterOperator(self._fst.eval(x), RowOperator(self._snd.eval(x)))
        consistency_check(op)
        return op

    @property
    def derivative(self):
        raise NotImplementedError


class SymbolicApplyForm(SymbolicTensor):
    def __init__(self, form, vector):
        assert vector.indices == (1,)
        assert form.indices == (-1,)
        self._vector = vector
        self._form = form
        self._indices = ()

    def __str__(self):
        return '{}.applyForm({})'.format(self._form, self._vector)

    def eval(self, x):
        return self._form.eval(x).vdot(self._vector.eval(x))

    @property
    def derivative(self):
        A = SymbolicCABL(self._form.derivative, self._vector)
        B = SymbolicCABL(self._vector.derivative, self._form)
        return SymbolicAdd(A, B)
