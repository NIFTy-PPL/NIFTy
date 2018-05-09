from . import NLTensor
from .. import ZeroTensor
from ..operators import Tensor


class NLConstant(NLTensor):
    def __init__(self, tensor):
        """
        Takes a tensor object and wraps it into a Nonlinear Object.
        """
        assert isinstance(tensor, Tensor)
        self._tensor = tensor

    def __call__(self, x):
        return self

    def __str__(self):
        return 'NLConst{}'.format(self._tensor)

    def eval(self, x):
        return self._tensor

    @property
    def derivative(self):
        indices = self._tensor.indices + (-1,)
        return self.__class__(ZeroTensor(indices))
