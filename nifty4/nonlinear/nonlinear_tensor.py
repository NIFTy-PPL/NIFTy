from ..field import Field


class NLTensor(object):
    def __call__(self, x):
        return NLChain(self, x)

    @property
    def indices(self):
        return self._indices

    @property
    def rank(self):
        return len(self._indices)

    def eval(self, x):
        raise NotImplementedError

    @property
    def derivative(self):
        raise NotImplementedError

    @property
    def adjoint(self):
        if self.rank in [1, 2]:
            return NLAdjoint(self)
        else:
            print(self.rank)
            raise NotImplementedError

    @staticmethod
    def _makeOp(thing):
        raise NotImplementedError
        # return thing if isinstance(thing, NLTensor) else NLOp_const(thing)

    def __neg__(self):
        raise NotImplementedError
        # return NLOp_neg(self)

    def __add__(self, other):
        raise NotImplementedError
        # return NLOp_add(self, self._makeOp(other))

    def __radd__(self, other):
        raise NotImplementedError
        # return NLOp_add(self._makeOp(other), self)

    def __sub__(self, other):
        raise NotImplementedError
        # return NLOp_add(self, NLOp_neg(self._makeOp(other)))

    def __rsub__(self, other):
        raise NotImplementedError
        # return NLOp_add(NLOp_neg(self._makeOp(other)), self)


class NLChain(NLTensor):
    def __init__(self, outer, inner):
        assert outer.rank == 1 and inner.rank == 1
        self._outer = outer
        self._inner = inner
        self._indices = outer.indices

    def __str__(self):
        return '{}({})'.format(self._outer, self._inner)

    def eval(self, x):
        return self._outer.eval(self._inner.eval(x))

    @property
    def derivative(self):
        return self.__class__(self._outer, self._inner.derivative)


class NLChainLinOps(NLTensor):
    def __init__(self, op1, op2):
        assert op1.rank == 2 and op2.rank == 2
        assert op1.indices[-1] == - op2.indices[0]
        self._indices = (op1.indices[0], op2.indices[-1])
        self._op1 = op1
        self._op2 = op2

    def __str__(self):
        return '{} {}'.format(self._op1, self._op2)

    def eval(self, x):
        # print()
        # print('start')
        # # print(self._op1)
        # print(self._op1.eval(x).domain)
        # # print(self._op2)
        # print(self._op2.eval(x).target)
        # print('end')
        # print()
        A = self._op1.eval(x)
        B = self._op2.eval(x)
        return A * B

    @property
    def derivative(self):
        raise NotImplementedError


class NLChainLinOps11(NLTensor):
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
        return self._op2.eval(x) * self._op1.eval(x)

    @property
    def derivative(self):
        raise NotImplementedError


class NLCABF(NLTensor):
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
        # FIXME
        # f = self._lop.value(x)(self._arg.value(x))
        # df = self._arg.derivative * self._lop.adjoint
        from .constant import NLZero
        if isinstance(self._nltensor.derivative, NLZero) and len(self._args) == 1:
            return NLChainLinOps(self._nltensor, self._args[0].derivative)
        else:
            # TODO Implement more general case
            raise NotImplementedError


class NLCABL(NLTensor):
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
        if len(self._args) == 1:
            nlvector = self._args[0]
            vector = nlvector.eval(x)
            operator = self._nltensor.eval(x)
            if nlvector.indices == (1,):
                print(operator)
                print(vector)
                return operator(vector)
            elif nlvector.indices == (-1,):
                return operator.adjoint(vector).conjugate()
            else:
                raise NotImplementedError
        raise NotImplementedError

    @property
    def derivative(self):
        # FIXME This is not the general case!!!
        assert len(self._args) == 1
        return NLChainLinOps11(self._nltensor, self._args[0].derivative)


class NLVdot(NLTensor):
    def __init__(self, vector1, vector2):
        assert vector1.indices == (1,)
        assert vector2.indices == (1,)
        self._vector1 = vector1
        self._vector2 = vector2
        self._indices = ()

    def __str__(self):
        return '({}).vdot({})'.format(self._vector1, self._vector2)

    def eval(self, x):
        return self._vector1.eval(x).vdot(self._vector2.eval(x))

    @property
    def derivative(self):
        from .add import NLTensorAdd
        A = NLCABL(self._vector1.derivative, self._vector2.adjoint)
        B = NLCABL(self._vector2.derivative, self._vector1.adjoint)
        return NLTensorAdd(A, B)


class NLScalarMul(NLTensor):
    def __init__(self, nltensor, nlscalar):
        assert isinstance(nltensor, NLTensor)
        assert isinstance(nlscalar, NLTensor)
        assert nlscalar.rank == 0
        self._nltensor = nltensor
        self._nlscalar = nlscalar
        self._indices = self._nltensor.indices

    def __str__(self):
        return '({}) x ({})'.format(self._nlscalar, self._nltensor)

    def eval(self, x):
        scalar = self._nlscalar.eval(x)
        if isinstance(scalar, Field) and len(scalar.domain) == 0:
            scalar = scalar.to_global_data()[()]
        return scalar * self._nltensor.eval(x)

    @property
    def derivative(self):
        from .add import NLTensorAdd
        if self._nltensor.rank == 0 and self._nlscalar.rank == 0:
            A = NLScalarMul(self._nltensor.derivative, self._nlscalar)
            B = NLScalarMul(self._nlscalar.derivative, self._nltensor)
        elif self._nltensor.rank == 1:
            A = NLScalarMul(self._nltensor.derivative, self._nlscalar)
            B = NLOuterProd(self._nlscalar.derivative, self._nltensor)
        else:
            raise NotImplementedError
        return NLTensorAdd(A, B)


class NLOuterProd(NLTensor):
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
        from ..operators import RowOperator
        from ..operators import OuterOperator
        from .constant import NLZero
        if isinstance(self._snd, NLZero) or isinstance(self._fst, NLZero):
            return 0.
        op = OuterOperator(self._fst.eval(x), RowOperator(self._snd.eval(x)))
        from ..extra.operator_tests import consistency_check
        consistency_check(op)
        return op

    @property
    def derivative(self):
        raise NotImplementedError


class NLApplyForm(NLTensor):
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
        A = NLCABL(self._form.derivative, self._vector)
        B = NLCABL(self._vector.derivative, self._form)
        from .add import NLTensorAdd
        return NLTensorAdd(A, B)


class NLAdjoint(NLTensor):
    def __init__(self, thing, indices=None):
        assert thing.rank in [1, 2]
        if indices is None:
            if thing.rank == 1:
                self._indices = (-1 * thing.indices[0],)
            else:
                self._indices = thing.indices[::-1]
        else:
            self._indices = indices
        self._thing = thing

    def __str__(self):
        return '{}^dagger'.format(self._thing)

    def eval(self, x):
        if self.rank == 2:
            return self._thing.eval(x).adjoint
        elif self.rank == 1:
            return self._thing.eval(x).conjugate()
        else:
            raise NotImplementedError

    @property
    def derivative(self):
        if self.rank == 1:
            return self.__class__(self._thing.derivative, self._indices + (-1,))
        raise NotImplementedError
