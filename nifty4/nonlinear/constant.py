from . import NLTensor
from .. import ZeroTensor


class Constant(NLTensor):
    def __init__(self, tensor):
        self._tensor = tensor

    def __call__(self, x):
        return self

    def eval(self, x):
        return self._tensor

    @property
    def derivative(self):
        # FIXME
        # self.__class__(ZeroTensor(...))
