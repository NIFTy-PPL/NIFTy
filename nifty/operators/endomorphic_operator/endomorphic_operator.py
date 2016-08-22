# -*- coding: utf-8 -*-

import abc

from nifty.operators.linear_operator import LinearOperator


class EndomorphicOperator(LinearOperator):
    __metaclass__ = abc.ABCMeta

    # ---Overwritten properties and methods---

    def inverse_times(self, x, spaces=None, types=None):
        if self.paradict['symmetric'] and self.paradict['unitary']:
            return self.times(x, spaces, types)
        else:
            return super(EndomorphicOperator, self).inverse_times(
                                                              x=x,
                                                              spaces=spaces,
                                                              types=types)

    def adjoint_times(self, x, spaces=None, types=None):
        if self.paradict['symmetric']:
            return self.times(x, spaces, types)
        elif self.paradict['unitary']:
            return self.inverse_times(x, spaces, types)
        else:
            return super(EndomorphicOperator, self).adjoint_times(
                                                                x=x,
                                                                spaces=spaces,
                                                                types=types)

    def adjoint_inverse_times(self, x, spaces=None, types=None):
        if self.paradict['symmetric']:
            return self.inverse_times(x, spaces, types)
        elif self.paradict['unitary']:
            return self.times(x, spaces, types)
        else:
            return super(EndomorphicOperator, self).adjoint_inverse_times(
                                                                x=x,
                                                                spaces=spaces,
                                                                types=types)

    def inverse_adjoint_times(self, x, spaces=None, types=None):
        if self.paradict['symmetric']:
            return self.inverse_times(x, spaces, types)
        elif self.paradict['unitary']:
            return self.times(x, spaces, types)
        else:
            return super(EndomorphicOperator, self).inverse_adjoint_times(
                                                                x=x,
                                                                spaces=spaces,
                                                                types=types)

    # ---Mandatory properties and methods---

    @property
    def target(self):
        return self.domain

    @property
    def field_type_target(self):
        return self.field_type

    # ---Added properties and methods---

    @abc.abstractproperty
    def symmetric(self):
        raise NotImplementedError

    @abc.abstractproperty
    def unitary(self):
        raise NotImplementedError

    def trace(self):
        pass

    def inverse_trace(self):
        pass

    def diagonal(self):
        pass

    def inverse_diagonal(self):
        pass

    def determinant(self):
        pass

    def inverse_determinant(self):
        pass

    def log_determinant(self):
        pass

    def trace_log(self):
        pass
