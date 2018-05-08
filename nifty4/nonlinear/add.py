from . import NLTensor


class Tensor_Add(NLTensor):
    def __init__(self, fst, snd):
        # TODO Check indices
        self._fst = fst
        self._snd = snd

    def __call__(x):
        pass

    def eval(self, x):
        return self._fst.eval(x) + self._snd.eval(x)

    @property
    def derivative(self):
        return self._fst.derivative + self._snd.derivative
