from ..operators import MultiSkyGradientOperator
from .model import Model


class Constant(Model):
    def __init__(self, position, constant):
        super(Constant, self).__init__(position)
        self._constant = constant

        self._value = self._constant

        self._gradient = MultiSkyGradientOperator({},
                                                  position.domain,
                                                  self.value.domain)

    def at(self, position):
        return self.__class__(position, self._constant)
