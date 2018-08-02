from __future__ import absolute_import, division, print_function
import abc

from ..compat import *
from ..utilities import NiftyMetaBase


class Operator(NiftyMetaBase()):
    """Transforms values living on one domain into values living on another
    domain, and can also provide the Jacobian.
    """

    def domain(self):
        """DomainTuple or MultiDomain : the operator's input domain

            The domain on which the Operator's input Field lives."""
        return self._domain

    def target(self):
        """DomainTuple or MultiDomain : the operator's output domain

            The domain on which the Operator's output Field lives."""
        return self._target

    def __matmul__(self, x):
        if not isinstance(x, Operator):
            return NotImplemented
        return OpChain.make((self, x))
        ops1 = self._ops if isinstance(self, OpChain) else (self,)
        ops2 = x._ops if isinstance(x, OpChain) else (x,)
        return OpChain(ops1+ops2)

    def chain(self, x):
        res = self.__matmul__(x)
        if res == NotImplemented:
            raise TypeError("operator expected")
        return res

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


class _CombinedOperator(Operator):
    def __init__(self, ops, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        self._ops = tuple(ops)

    @classmethod
    def unpack(cls, ops, res):
        for op in ops:
            if isinstance(op, cls):
                res = cls.unpack(op, res)
            else:
                res = res + [op]
        return res

    @classmethod
    def make(cls, ops):
        res = cls.unpack(ops, [])
        if len(res) == 1:
            return res[0]
        return cls(res, _callingfrommake=True)


class _OpChain(_CombinedOperator):
    def __init__(self, ops, _callingfrommake=False):
        super(_OpChain, self).__init__(ops, _callingfrommake)
        self._domain = self._ops[-1].domain
        self._target = self._ops[0].target

    def __call__(self, x):
        for op in reversed(self._ops):
            x = op(x)
        return x


class _OpProd(_CombinedOperator):
    def __init__(self, ops, _callingfrommake=False):
        super(_OpProd, self).__init__(ops, _callingfrommake)
        self._domain = self._ops[0].domain
        self._target = self._ops[0].target

    def __call__(self, x):
        return my_prod(map(lambda op: op(x) for op in self._ops))


class _OpSum(_CombinedOperator):
    def __init__(self, ops, _callingfrommake=False):
        super(_OpSum, self).__init__(ops, _callingfrommake)
        self._domain = domain_union([op.domain for op in self._ops])
        self._target = domain_union([op.target for op in self._ops])


    def __call__(self, x):
        raise NotImplementedError
