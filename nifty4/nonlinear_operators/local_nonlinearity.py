from nifty4.sugar import makeOp

from .nonlinear_operator import NonlinearOperator


class LocalModel(NonlinearOperator):
    def __init__(self, position, inp, nonlinearity):
        """
        Computes nonlinearity(inp)
        """
        super(LocalModel, self).__init__(position)

        self._inp = inp.at(self.position)
        self._nonlinearity = nonlinearity

        self._value = nonlinearity(self._inp.value)

        # Gradient
        self._gradient = makeOp(self._nonlinearity.derivative(self._inp.value))*self._inp.gradient

    def at(self, position):
        return self.__class__(position, self._inp, self._nonlinearity)
