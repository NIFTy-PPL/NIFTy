class NLTensor(object):
    def __call__(self, x):
        return NLChain(self, x)

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
        # FIXME Check indices
        self._outer = outer
        self._inner = inner

    def __str__(self):
        return '{}({})'.format(self._outer, self._inner)

    def __call__(self, x):
        return self.__class__(self, x)

    def eval(self, x):
        return self._outer.eval(self._inner.eval(x))

    @property
    def derivative(self):
        return self.__class__(self._outer, self._inner.derivative)
