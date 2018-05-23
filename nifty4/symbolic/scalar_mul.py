from ..field import Field
from .add import SymbolicAdd
from .contractions import SymbolicOuterProd
from .symbolic_tensor import SymbolicTensor


class SymbolicScalarMul(SymbolicTensor):
    def __init__(self, nltensor, nlscalar):
        assert isinstance(nltensor, SymbolicTensor)
        assert isinstance(nlscalar, SymbolicTensor)
        assert nlscalar.rank == 0
        super(SymbolicScalarMul, self).__init__(nltensor._indices)
        self._nltensor = nltensor
        self._nlscalar = nlscalar

    def __str__(self):
        return '({}) x ({})'.format(self._nlscalar, self._nltensor)

    def eval(self, x):
        scalar = self._nlscalar.eval(x)
        if isinstance(scalar, Field) and len(scalar.domain) == 0:
            scalar = scalar.to_global_data()[()]
        return scalar * self._nltensor.eval(x)

    @property
    def derivative(self):
        if self._nltensor.rank == 0 and self._nlscalar.rank == 0:
            A = self.__class__(self._nltensor.derivative, self._nlscalar)
            B = self.__class__(self._nlscalar.derivative, self._nltensor)
        elif self._nltensor.rank == 1:
            A = self.__class__(self._nltensor.derivative, self._nlscalar)
            B = SymbolicOuterProd(self._nlscalar.derivative, self._nltensor)
        else:
            raise NotImplementedError
        return SymbolicAdd(A, B)
