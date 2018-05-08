class NLTensor(object):
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
