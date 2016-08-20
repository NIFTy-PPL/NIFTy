# -*- coding: utf-8 -*-

from nifty.config import about
from nifty.operators.linear_operator import LinearOperator
from square_operator_paradict import SquareOperatorParadict


class SquareOperator(LinearOperator):

    def __init__(self, domain=None, target=None,
                 field_type=None, field_type_target=None,
                 implemented=False, symmetric=False, unitary=False):

        if target is not None:
            about.warnings.cprint(
                "WARNING: Discarding given target for SquareOperator.")
        target = domain

        if field_type_target is not None:
            about.warnings.cprint(
                "WARNING: Discarding given field_type_target for "
                "SquareOperator.")
        field_type_target = field_type

        LinearOperator.__init__(self,
                                domain=domain,
                                target=target,
                                field_type=field_type,
                                field_type_target=field_type_target,
                                implemented=implemented)

        self.paradict = SquareOperatorParadict(symmetric=symmetric,
                                               unitary=unitary)

    def inverse_times(self, x, spaces=None, types=None):
        if self.paradict['symmetric'] and self.paradict['unitary']:
            return self.times(x, spaces, types)
        else:
            return LinearOperator.inverse_times(self,
                                                x=x,
                                                spaces=spaces,
                                                types=types)

    def adjoint_times(self, x, spaces=None, types=None):
        if self.paradict['symmetric']:
            return self.times(x, spaces, types)
        elif self.paradict['unitary']:
            return self.inverse_times(x, spaces, types)
        else:
            return LinearOperator.adjoint_times(self,
                                                x=x,
                                                spaces=spaces,
                                                types=types)

    def adjoint_inverse_times(self, x, spaces=None, types=None):
        if self.paradict['symmetric']:
            return self.inverse_times(x, spaces, types)
        elif self.paradict['unitary']:
            return self.times(x, spaces, types)
        else:
            return LinearOperator.adjoint_inverse_times(self,
                                                        x=x,
                                                        spaces=spaces,
                                                        types=types)

    def inverse_adjoint_times(self, x, spaces=None, types=None):
        if self.paradict['symmetric']:
            return self.inverse_times(x, spaces, types)
        elif self.paradict['unitary']:
            return self.times(x, spaces, types)
        else:
            return LinearOperator.inverse_adjoint_times(self,
                                                        x=x,
                                                        spaces=spaces,
                                                        types=types)

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
