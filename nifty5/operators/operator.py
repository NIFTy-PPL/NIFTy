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
    """Transforms values living on one domain into values living on another
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

    def simplify_for_constant_input(self, c_inp):
        if c_inp is None or c_inp.domain != self.domain:
            return None, self
        op = _ConstantOperator(self.domain, self(c_inp))
        return op(c_inp), op


for f in ["sqrt", "exp", "log", "tanh", "sigmoid", 'sin', 'cos', 'tan',
          'sinh', 'cosh', 'absolute', 'sinc', 'one_over']:
    def func(f):
        def func2(self):
            fa = _FunctionApplier(self.target, f)
            return _OpChain.make((fa, self))
        return func2
    setattr(Operator, f, func(f))


class _ConstCollector(object):
    def __init__(self):
        self._const = None
        self._nc = set()

    def mult(self, const, fulldom):
        if const is None:
            self._nc |= set(fulldom)
        else:
            self._nc |= set(fulldom) - set(const)
            if self._const is None:
                from ..multi_field import MultiField
                self._const = MultiField.from_dict(
                    {key: const[key] for key in const if key not in self._nc})
            else:
                from ..multi_field import MultiField
                self._const = MultiField.from_dict(
                    {key: self._const[key]*const[key]
                     for key in const if key not in self._nc})

    def add(self, const, fulldom):
        if const is None:
            self._nc |= set(fulldom.keys())
        else:
            from ..multi_field import MultiField
            self._nc |= set(fulldom.keys()) - set(const.keys())
            if self._const is None:
                self._const = MultiField.from_dict(
                    {key: const[key] for key in const.keys() if key not in self._nc})
            else:
                self._const = self._const.unite(const)
                self._const = MultiField.from_dict(
                    {key: self._const[key]
                     for key in self._const if key not in self._nc})

    @property
    def constfield(self):
        return self._const


class _ConstantOperator(Operator):
    def __init__(self, dom, output):
        from ..sugar import makeDomain
        self._domain = makeDomain(dom)
        self._target = output.domain
        self._output = output

    def apply(self, x):
        from ..linearization import Linearization
        from .simple_linear_operators import NullOperator
        self._check_input(x)
        if not isinstance(x, Linearization):
            return self._output
        return x.new(self._output, NullOperator(self._domain, self._target))

    def __repr__(self):
        return 'ConstantOperator <- {}'.format(self.domain.keys())


class _ConstantOperator2(Operator):
    def __init__(self, target, constant_output):
        from ..sugar import makeDomain
        self._target = makeDomain(target)
        dom_keys = set(target.keys())-set(constant_output.domain.keys())
        self._domain = makeDomain({key: self._target[key] for key in dom_keys})
        self._constant_output = constant_output

    def apply(self, x):
        from ..linearization import Linearization
        self._check_input(x)
        if not isinstance(x, Linearization):
            return x.unite(self._constant_output)
        from .simple_linear_operators import _PartialExtractor

        op = _PartialExtractor(self.target, x.jac.target).adjoint
        val = x.val.unite(self._constant_output)

        assert val.domain is self.target
        assert val.domain is op.target

        return x.new(val, op(x.jac))

    def __repr__(self):
        return 'ConstantOperator2: {} <- {}'.format(self.target.keys(), self.domain.keys())


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

    def simplify_for_constant_input(self, c_inp):
        if c_inp is None:
            return None, self
        if c_inp.domain == self.domain:
            op = _ConstantOperator(self.domain, self(c_inp))
            return op(c_inp), op

        from ..multi_domain import MultiDomain
        if not isinstance(self._domain, MultiDomain):
            return None, self

        newop = None
        for op in reversed(self._ops):
            c_inp, t_op = op.simplify_for_constant_input(c_inp)
            newop = t_op if newop is None else op(newop)
        return c_inp, newop

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

    def simplify_for_constant_input(self, c_inp):
        if c_inp is None:
            return None, self
        if c_inp.domain == self.domain:
            op = _ConstantOperator(self.domain, self(c_inp))
            return op(c_inp), op

        f1, o1 = self._op1.simplify_for_constant_input(
            c_inp.extract_part(self._op1.domain))
        f2, o2 = self._op2.simplify_for_constant_input(
            c_inp.extract_part(self._op2.domain))

        from ..multi_domain import MultiDomain
        if not isinstance(self._target, MultiDomain):
            return None, _OpProd(o1, o2)

        cc = _ConstCollector()
        cc.mult(f1, o1.target)
        cc.mult(f2, o2.target)
        return cc.constfield, _OpProd(o1, o2)

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

    def simplify_for_constant_input(self, c_inp):
        if c_inp is None:
            return None, self
        if c_inp.domain == self.domain:
            op = _ConstantOperator(self.domain, self(c_inp))
            return op(c_inp), op

        f1, o1 = self._op1.simplify_for_constant_input(
            c_inp.extract_part(self._op1.domain))
        f2, o2 = self._op2.simplify_for_constant_input(
            c_inp.extract_part(self._op2.domain))

        from ..multi_domain import MultiDomain
        if not isinstance(self._target, MultiDomain):
            return None, _OpSum(o1, o2)

        cc = _ConstCollector()
        cc.add(f1, o1.target)
        cc.add(f2, o2.target)
        return cc.constfield, _OpSum(o1, o2)

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in (self._op1, self._op2))
        return "_OpSum:\n"+indent(subs)
