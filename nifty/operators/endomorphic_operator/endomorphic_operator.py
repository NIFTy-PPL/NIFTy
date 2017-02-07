# -*- coding: utf-8 -*-

import abc

from nifty.operators.linear_operator import LinearOperator


class EndomorphicOperator(LinearOperator):

    # ---Overwritten properties and methods---

    def inverse_times(self, x, spaces=None):
        if self.symmetric and self.unitary:
            return self.times(x, spaces)
        else:
            return super(EndomorphicOperator, self).inverse_times(
                                                              x=x,
                                                              spaces=spaces)

    def adjoint_times(self, x, spaces=None):
        if self.symmetric:
            return self.times(x, spaces)
        else:
            return super(EndomorphicOperator, self).adjoint_times(
                                                                x=x,
                                                                spaces=spaces)

    def adjoint_inverse_times(self, x, spaces=None):
        if self.symmetric:
            return self.inverse_times(x, spaces)
        else:
            return super(EndomorphicOperator, self).adjoint_inverse_times(
                                                                x=x,
                                                                spaces=spaces)

    def inverse_adjoint_times(self, x, spaces=None):
        if self.symmetric:
            return self.inverse_times(x, spaces)
        else:
            return super(EndomorphicOperator, self).inverse_adjoint_times(
                                                                x=x,
                                                                spaces=spaces)

    # ---Mandatory properties and methods---

    @property
    def target(self):
        return self.domain

    # ---Added properties and methods---

    @abc.abstractproperty
    def symmetric(self):
        raise NotImplementedError
