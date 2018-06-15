from nifty4.sugar import makeOp

from .model import Model


class LocalModel(Model):
    def __init__(self, position, inp, nonlinearity):
        """
        Computes nonlinearity(inp)
        """
        super(LocalModel, self).__init__(position)
        self._inp = inp.at(self.position)
        self._nonlinearity = nonlinearity
        self._value = nonlinearity(self._inp.value)
        d_inner = self._inp.gradient
        d_outer = makeOp(self._nonlinearity.derivative(self._inp.value))
        self._gradient = d_outer * d_inner

    def at(self, position):
        return self.__class__(position, self._inp, self._nonlinearity)
