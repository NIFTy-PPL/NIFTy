from nifty4.library.nonlinearities import Exponential, PositiveTanh, Tanh
from nifty4.sugar import makeOp

from .model import Model


class LocalModel(Model):
    def __init__(self, inp, nonlinearity):
        """
        Computes nonlinearity(inp)
        """
        super(LocalModel, self).__init__(inp.position)
        self._inp = inp
        self._nonlinearity = nonlinearity
        self._value = nonlinearity(self._inp.value)
        d_inner = self._inp.gradient
        d_outer = makeOp(self._nonlinearity.derivative(self._inp.value))
        self._gradient = d_outer * d_inner

    def at(self, position):
        return self.__class__(self._inp.at(position), self._nonlinearity)


def PointwiseExponential(inp):
    return LocalModel(inp, Exponential())


def PointwiseTanh(inp):
    return LocalModel(inp, Tanh())


def PointwisePositiveTanh(inp):
    return LocalModel(inp, PositiveTanh())
