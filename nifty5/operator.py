from __future__ import absolute_import, division, print_function

from .compat import *
from .utilities import NiftyMetaBase


class Operator(NiftyMetaBase()):
    """Transforms values living on one domain into values living on another
    domain, and can also provide the Jacobian.
    """

    def chain(self, x):
        if not callable(x):
            raise TypeError("callable needed")
        ops1 = self._ops if isinstance(self, OpChain) else (self,)
        ops2 = x._ops if isinstance(x, OpChain) else (x,)
        return OpChain(ops1+ops2)

    def __call__(self, x):
        """Returns transformed x

        Parameters
        ----------
        x : Linearization
            input

        Returns
        -------
        Linearization
            output
        """
        raise NotImplementedError


class OpChain(Operator):
    def __init__(self, ops):
        self._ops = tuple(ops)

    def __call__(self, x):
        for op in reversed(self._ops):
            x = op(x)
        return x
