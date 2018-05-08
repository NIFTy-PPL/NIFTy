from . import NLTensor
from .. import ZeroTensor
from ..operators import Tensor


class NLConstant(NLTensor):
    def __init__(self, tensor, index=None):
        assert isinstance(tensor, Tensor)
        assert index in (None, 0, 1)
        self._tensor = tensor
        self._index = index

    def __call__(self, x):
        return self

    def __str__(self):
        if self._index is not None:
            return '{}^{}'.format(self._tensor, self._index)
        return '{}'.format(self._tensor)

    def eval(self, x):
        if self._index is not None:
            return self._tensor.contract(x, index=self._index)
        return self._tensor

    @property
    def derivative(self):
        indices = self._tensor.indices + (-1,)
        self.__class__(ZeroTensor(indices))
