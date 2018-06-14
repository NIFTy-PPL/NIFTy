from ..operators import MultiSkyGradientOperator
from .nonlinear_operator import NonlinearOperator


class ConstantModel(NonlinearOperator):
    def __init__(self, position, constant):
        super(ConstantModel, self).__init__(position)
        self._constant = constant

        self._value = self._constant

        self._gradient = MultiSkyGradientOperator({},
                                                  position.domain,
                                                  self.value.domain)

    def at(self, position):
        return self.__class__(position, self._constant)
