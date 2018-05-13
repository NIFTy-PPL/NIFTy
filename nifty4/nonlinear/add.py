from . import NLTensor


class NLTensorAdd(NLTensor):
    def __init__(self, fst, snd):
        assert fst.indices == snd.indices
        self._indices = fst.indices
        self._fst = fst
        self._snd = snd

    def __str__(self):
        return '{} + {})'.format(self._fst, self._snd)

    def __call__(self, x):
        return self._fst(x) + self._snd(x)

    def eval(self, x):
        return self._fst.eval(x) + self._snd.eval(x)

    @property
    def derivative(self):
        return self.__class__(self._fst.derivative,  self._snd.derivative)
