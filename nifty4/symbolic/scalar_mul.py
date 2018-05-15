from ..field import Field
from .add import NLAdd
from .contractions import NLOuterProd
from .tensor import NLTensor


class NLScalarMul(NLTensor):
    def __init__(self, nltensor, nlscalar):
        assert isinstance(nltensor, NLTensor)
        assert isinstance(nlscalar, NLTensor)
        assert nlscalar.rank == 0
        self._nltensor = nltensor
        self._nlscalar = nlscalar
        self._indices = self._nltensor.indices

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
            B = NLOuterProd(self._nlscalar.derivative, self._nltensor)
        else:
            raise NotImplementedError
        return NLAdd(A, B)
