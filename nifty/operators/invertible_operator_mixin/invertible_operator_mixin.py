# -*- coding: utf-8 -*-

from nifty.minimization import ConjugateGradient

from nifty.field import Field


class InvertibleOperatorMixin(object):
    def __init__(self, inverter=None, preconditioner=None):
        self.__preconditioner = preconditioner
        if inverter is not None:
            self.__inverter = inverter
        else:
            self.__inverter = ConjugateGradient(
                                            preconditioner=self.preconditioner)

    def _times(self, x, spaces, types, x0=None):
        if x0 is None:
            x0 = Field(self.target, val=0., dtype=x.dtype)

        (result, convergence) = self.inverter(A=self.inverse_times,
                                              b=x,
                                              x0=x0)
        return result

    def _adjoint_times(self, x, spaces, types, x0=None):
        if x0 is None:
            x0 = Field(self.domain, val=0., dtype=x.dtype)

        (result, convergence) = self.inverter(A=self.adjoint_inverse_times,
                                              b=x,
                                              x0=x0)
        return result

    def _inverse_times(self, x, spaces, types, x0=None):
        if x0 is None:
            x0 = Field(self.domain, val=0., dtype=x.dtype)

        (result, convergence) = self.inverter(A=self.times,
                                              b=x,
                                              x0=x0)
        return result

    def _adjoint_inverse_times(self, x, spaces, types, x0=None):
        if x0 is None:
            x0 = Field(self.target, val=0., dtype=x.dtype)

        (result, convergence) = self.inverter(A=self.adjoint_times,
                                              b=x,
                                              x0=x0)
        return result

    def _inverse_adjoint_times(self, x, spaces, types):
        raise NotImplementedError(
            "no generic instance method 'inverse_adjoint_times'.")
