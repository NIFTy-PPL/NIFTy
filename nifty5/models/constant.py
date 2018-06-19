from ..operators.model_gradient_operator import ModelGradientOperator
from .model import Model


class Constant(Model):
    def __init__(self, position, constant):
        super(Constant, self).__init__(position)
        self._constant = constant

        self._value = self._constant

        self._gradient = ModelGradientOperator({}, position.domain,
                                               self.value.domain)

    def at(self, position):
        return self.__class__(position, self._constant)
