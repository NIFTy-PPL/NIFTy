from . import NLTensor
from .. import ZeroTensor
from ..operators import Tensor


class Constant(NLTensor):
    def __init__(self, tensor):
        assert isinstance(tensor, Tensor)
        self._tensor = tensor

    def __call__(self, x):
        return self

    def eval(self, x):
        return self._tensor

    @property
    def derivative(self):
        indices = self._tensor.indices + (-1,)
        self.__class__(ZeroTensor(indices))
