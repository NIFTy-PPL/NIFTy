from .tensor import Tensor
from .constant import SymbolicConstant
from .symbolic_tensor import SymbolicTensor


class SymbolicVariable(SymbolicTensor):
    def __init__(self, domain):
        super(SymbolicVariable, self).__init__((1,))
        self._domain = domain

    def __call__(self, x):
        raise ValueError

    def __str__(self):
        return 'var'

    def eval(self, x):
        return x

    @property
    def derivative(self):
        return SymbolicConstant(Tensor(1., 2, self._domain), self._indices + (-1, ))
