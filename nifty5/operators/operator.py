from __future__ import absolute_import, division, print_function

from ..compat import *
from ..utilities import NiftyMetaBase, my_product
from ..domain_tuple import DomainTuple


class Operator(NiftyMetaBase()):
    """Transforms values living on one domain into values living on another
    domain, and can also provide the Jacobian.
    """

    @property
    def domain(self):
        """DomainTuple or MultiDomain : the operator's input domain

            The domain on which the Operator's input Field lives."""
        return self._domain

    @property
    def target(self):
        """DomainTuple or MultiDomain : the operator's output domain

            The domain on which the Operator's output Field lives."""
        return self._target

    def scale(self, factor):
        if factor == 1:
            return self
        from .scaling_operator import ScalingOperator
        return ScalingOperator(factor, self.target)(self)

    def conjugate(self):
        from .simple_linear_operators import ConjugationOperator
        return ConjugationOperator(self.target)(self)

    @property
    def real(self):
        from .simple_linear_operators import Realizer
        return Realizer(self.target)(self)

    def __neg__(self):
        return self.scale(-1)

    def __matmul__(self, x):
        if not isinstance(x, Operator):
            return NotImplemented
        return _OpChain.make((self, x))

    def __mul__(self, x):
        if not isinstance(x, Operator):
            return NotImplemented
        return _OpProd.make((self, x))

    def apply(self, x):
        raise NotImplementedError

    def __call__(self, x):
       if isinstance(x, Operator):
           return _OpChain.make((self, x))
       return self.apply(x)


for f in ["sqrt", "exp", "log", "tanh", "positive_tanh"]:
    def func(f):
        def func2(self):
            fa = _FunctionApplier(self.target, f)
            return _OpChain.make((fa, self))
        return func2
    setattr(Operator, f, func(f))


class _FunctionApplier(Operator):
    def __init__(self, domain, funcname):
        from ..sugar import makeDomain
        self._domain = self._target = makeDomain(domain)
        self._funcname = funcname

    def apply(self, x):
        return getattr(x, self._funcname)()


class _CombinedOperator(Operator):
    def __init__(self, ops, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        self._ops = tuple(ops)

    @classmethod
    def unpack(cls, ops, res):
        for op in ops:
            if isinstance(op, cls):
                res = cls.unpack(op._ops, res)
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

    def apply(self, x):
        for op in reversed(self._ops):
            x = op(x)
        return x


class _OpProd(_CombinedOperator):
    def __init__(self, ops, _callingfrommake=False):
        super(_OpProd, self).__init__(ops, _callingfrommake)
        self._domain = self._ops[0].domain
        self._target = self._ops[0].target

    def apply(self, x):
        return my_product(map(lambda op: op(x), self._ops))


class _OpSum(_CombinedOperator):
    def __init__(self, ops, _callingfrommake=False):
        super(_OpSum, self).__init__(ops, _callingfrommake)
        self._domain = domain_union([op.domain for op in self._ops])
        self._target = domain_union([op.target for op in self._ops])

    def apply(self, x):
        raise NotImplementedError
