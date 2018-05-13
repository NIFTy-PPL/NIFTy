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
        return self._op1.eval(x) * self._op2.eval(x)

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
        s = '{}F('.format(self._nltensor)
        for vv in self._args:
            s += '{}, '.format(vv)
        s = s[:-2]
        s += ')'
        return s

    def eval(self, x):
        if len(self._args) == 1:
            vector = self._args[0].eval(x)
            operator = self._nltensor.eval(x)
            return operator(vector)
        raise NotImplementedError

    @property
    def derivative(self):
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
        s = '{}L('.format(self._nltensor)
        for vv in self._args:
            s += '{}, '.format(vv)
        s = s[:-2]
        s += ')'
        return s

    def eval(self, x):
        if len(self._args) == 1:
            vector = self._args[0].eval(x)
            operator = self._nltensor.eval(x)
            return operator(vector)
        raise NotImplementedError

    @property
    def derivative(self):
        raise NotImplementedError


class NLVdot(NLTensor):
    def __init__(self, vector1, vector2):
        assert vector1.indices == (1,)
        assert vector2.indices == (1,)
        self._vector1 = vector1
        self._vector2 = vector2
        self._indices = ()

    def __str__(self):
        return '{}.vdot({})'.format(self._vector1, self._vector2)

    def eval(self, x):
        return self._vector1.eval(x).vdot(self._vector2.eval(x))

    @property
    def derivative(self):
        return NLCABL(self._vector1.derivative, self._vector2.adjoint) + NLCABL(self._vector2.derivative, self._vector1.adjoint)


class NLScalarMul(NLTensor):
    def __init__(self, nltensor, nlscalar):
        assert isinstance(nltensor, NLTensor)
        assert isinstance(nlscalar, NLTensor)
        assert nlscalar.rank == 0
        self._nltensor = nltensor
        self._nlscalar = nlscalar

    def __str__(self):
        return '{} x {}'.format(self._nlscalar, self._nltensor)

    def eval(self, x):
        return self._nlscalar.eval(x) * self._nltensor.eval(x)

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
        raise NotImplementedError
        return NLCABL(self._vector1.derivative, self._vector2.adjoint) + NLCABL(self._vector2.derivative, self._vector1.adjoint)

