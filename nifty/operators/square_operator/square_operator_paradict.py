# -*- coding: utf-8 -*-

from nifty.config import about
from nifty.operators.linear_operator import LinearOperatorParadict


class SquareOperatorParadict(LinearOperatorParadict):
    def __init__(self, symmetric, unitary):
        LinearOperatorParadict.__init__(self,
                                        symmetric=symmetric,
                                        unitary=unitary)

    def __setitem__(self, key, arg):
        if key not in ['symmetric', 'unitary']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported SquareOperator parameter: " + key))
        if key == 'symmetric':
            temp = bool(arg)
        elif key == 'unitary':
            temp = bool(arg)

        self.parameters.__setitem__(key, temp)
