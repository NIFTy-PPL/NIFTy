# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from ..utilities import NiftyMetaBase, indent


class Operator(NiftyMetaBase()):
    """Transforms values defined on one domain into values defined on another
    domain, and can also provide the Jacobian.
    """

    @property
    def domain(self):
        """DomainTuple or MultiDomain : the operator's input domain

            The domain on which the Operator's input Field is defined."""
        return self._domain

    @property
    def target(self):
        """DomainTuple or MultiDomain : the operator's output domain

            The domain on which the Operator's output Field is defined."""
        return self._target

    @staticmethod
    def _check_domain_equality(dom_op, dom_field):
        if dom_op != dom_field:
            s = "The operator's and field's domains don't match."
            from ..domain_tuple import DomainTuple
            from ..multi_domain import MultiDomain
            if not isinstance(dom_op, (DomainTuple, MultiDomain,)):
                s += " Your operator's domain is neither a `DomainTuple`" \
                     " nor a `MultiDomain`."
            raise ValueError(s)

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
        if isinstance(x, Operator):
            return _OpProd(self, x)
        if np.isscalar(x):
            return self.scale(x)
        return NotImplemented

    def __rmul__(self, x):
        return self.__mul__(x)

    def __add__(self, x):
        if not isinstance(x, Operator):
            return NotImplemented
        return _OpSum(self, x)

    def __sub__(self, x):
        if not isinstance(x, Operator):
            return NotImplemented
        return _OpSum(self, -x)

    def __pow__(self, power):
        if not np.isscalar(power):
            return NotImplemented
        return _OpChain.make((_PowerOp(self.target, power), self))

    def clip(self, min=None, max=None):
        if min is None and max is None:
            return self
        return _OpChain.make((_Clipper(self.target, min, max), self))

    def apply(self, x):
        """Returns the result of applying the operator to the Field x.

        Parameters
        ----------
        x : Field/Multifield
            the operator's input
        """
        raise NotImplementedError

    def force(self, x):
        """Extract correct subset of domain of x and apply operator."""
        return self.apply(x.extract(self.domain))

    def _check_input(self, x):
        from ..linearization import Linearization
        d = x.target if isinstance(x, Linearization) else x.domain
        self._check_domain_equality(self._domain, d)

    def __call__(self, x):
        if isinstance(x, Operator):
            return _OpChain.make((self, x))
        return self.apply(x)

    def ducktape(self, name):
        from .simple_linear_operators import ducktape
        return self(ducktape(self, None, name))

    def ducktape_left(self, name):
        from .simple_linear_operators import ducktape
        return ducktape(None, self, name)(self)

    def __repr__(self):
        return self.__class__.__name__


for f in ["sqrt", "exp", "log", "tanh", "sigmoid", 'sin', 'cos', 'tan',
          'sinh', 'cosh', 'absolute', 'sinc', 'one_over']:
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
        self._check_input(x)
        return getattr(x, self._funcname)()


class _Clipper(Operator):
    def __init__(self, domain, min=None, max=None):
        from ..sugar import makeDomain
        self._domain = self._target = makeDomain(domain)
        self._min = min
        self._max = max

    def apply(self, x):
        self._check_input(x)
        return x.clip(self._min, self._max)


class _PowerOp(Operator):
    def __init__(self, domain, power):
        from ..sugar import makeDomain
        self._domain = self._target = makeDomain(domain)
        self._power = power

    def apply(self, x):
        self._check_input(x)
        return x**self._power


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
        for i in range(1, len(self._ops)):
            if self._ops[i-1].domain != self._ops[i].target:
                raise ValueError("domain mismatch")

    def apply(self, x):
        self._check_input(x)
        for op in reversed(self._ops):
            x = op(x)
        return x

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in self._ops)
        return "_OpChain:\n" + indent(subs)


class _OpProd(Operator):
    def __init__(self, op1, op2):
        from ..sugar import domain_union
        self._domain = domain_union((op1.domain, op2.domain))
        self._target = op1.target
        if op1.target != op2.target:
            raise ValueError("target mismatch")
        self._op1 = op1
        self._op2 = op2

    def apply(self, x):
        from ..linearization import Linearization
        from ..sugar import makeOp
        self._check_input(x)
        lin = isinstance(x, Linearization)
        v = x._val if lin else x
        v1 = v.extract(self._op1.domain)
        v2 = v.extract(self._op2.domain)
        if not lin:
            return self._op1(v1) * self._op2(v2)
        wm = x.want_metric
        lin1 = self._op1(Linearization.make_var(v1, wm))
        lin2 = self._op2(Linearization.make_var(v2, wm))
        op = (makeOp(lin1._val)(lin2._jac))._myadd(
            makeOp(lin2._val)(lin1._jac), False)
        return lin1.new(lin1._val*lin2._val, op(x.jac))

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in (self._op1, self._op2))
        return "_OpProd:\n"+indent(subs)


class _OpSum(Operator):
    def __init__(self, op1, op2):
        from ..sugar import domain_union
        self._domain = domain_union((op1.domain, op2.domain))
        self._target = domain_union((op1.target, op2.target))
        self._op1 = op1
        self._op2 = op2

    def apply(self, x):
        from ..linearization import Linearization
        self._check_input(x)
        lin = isinstance(x, Linearization)
        v = x._val if lin else x
        v1 = v.extract(self._op1.domain)
        v2 = v.extract(self._op2.domain)
        res = None
        if not lin:
            return self._op1(v1).unite(self._op2(v2))
        wm = x.want_metric
        lin1 = self._op1(Linearization.make_var(v1, wm))
        lin2 = self._op2(Linearization.make_var(v2, wm))
        op = lin1._jac._myadd(lin2._jac, False)
        res = lin1.new(lin1._val.unite(lin2._val), op(x.jac))
        if lin1._metric is not None and lin2._metric is not None:
            res = res.add_metric(lin1._metric + lin2._metric)
        return res

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in (self._op1, self._op2))
        return "_OpSum:\n"+indent(subs)
