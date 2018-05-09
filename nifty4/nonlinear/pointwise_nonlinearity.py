from . import NLTensor, NLContract
from ..field import Field, exp
from ..operators import Tensor


class NLExp(NLTensor):
    def __init__(self, inner):
        self._inner = inner

    def __call__(self, x):
        raise NotImplementedError

    def __str__(self):
        return 'EXP({})'.format(self._inner)

    def eval(self, x):
        field = self._inner.eval(x)._thing
        assert isinstance(field, Field)
        return Tensor(self._inner.eval(x).indices, exp(field))

    @property
    def derivative(self):
        return NLContract(NLDiag(NLExp(self._inner), -1), self._inner.derivative, 1)


class NLDiag(NLTensor):
    def __init__(self, diag, additional_index):
        self._diag = diag
        self._additional_index = (additional_index,)

    def __call__(self, x):
        raise NotImplementedError

    def __str__(self):
        return 'DIAG({})_{}'.format(self._diag, self._additional_index)

    def eval(self, x):
        newIndex = self._diag.eval(x).indices + self._additional_index
        return Tensor(newIndex, self._diag.eval(x).output)

    @property
    def derivative(self):
        raise NotImplementedError
