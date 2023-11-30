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
# Copyright(C) 2013-2022 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numbers
from functools import reduce
from operator import add
from typing import Callable, Optional
from warnings import warn

import numpy as np

from .. import pointwise
from ..domain_tuple import DomainTuple
from ..logger import logger
from ..multi_domain import MultiDomain
from ..utilities import NiftyMeta, check_object_identity, indent, myassert


class Operator(metaclass=NiftyMeta):
    """Transforms values defined on one domain into values defined on another
    domain, and can also provide the Jacobian.
    """

    @property
    def domain(self):
        """The domain on which the Operator's input Field is defined.

        Returns
        -------
        domain : DomainTuple or MultiDomain
        """
        return self._domain

    @property
    def target(self):
        """The domain on which the Operator's output Field is defined.

        Returns
        -------
        target : DomainTuple or MultiDomain
        """
        return self._target

    @property
    def val(self):
        """The numerical value associated with this object
        For "pure" operators this is `None`. For Field-like objects this
        is a `numpy.ndarray` or a dictionary of `numpy.ndarray`s mathcing the
        object's `target`.

        Returns
        -------
        None or numpy.ndarray or dictionary of np.ndarrays : the numerical value
        """
        return None

    @property
    def jac(self):
        """The Jacobian associated with this object.
        For "pure" operators this is `None`. For Field-like objects this
        can be `None` (in which case the object is a constant), or it can be a
        `LinearOperator` with `domain` and `target` matching the object's.

        Returns
        -------
        None or LinearOperator : the Jacobian

        Notes
        -----
        if `value` is None, this must be `None` as well!
        """
        return None

    @property
    def want_metric(self):
        """Whether a metric should be computed for the full expression.
        This is `False` whenever `jac` is `None`. In other cases it signals
        that operators processing this object should compute the metric.

        Returns
        -------
        bool : whether the metric should be computed
        """
        return False

    @property
    def metric(self):
        """The metric associated with the object.
        This is `None`, except when all the following conditions hold:
        - `want_metric` is `True`

        - `target` is the scalar domain

        - the operator chain contained an operator which could compute the
          metric

        Returns
        -------
        None or LinearOperator : the metric
        """
        return None

    @property
    def jax_expr(self) -> Optional[Callable]:
        """Equivalent representation of the operator in JAX."""
        expr = getattr(self, "_jax_expr", None)
        # NOTE, it is incredibly useful to enable this for debugging
        # if expr is None:
        #     warn(f"no JAX expression associated with operator {self!r}")
        return expr

    def scale(self, factor):
        if not isinstance(factor, numbers.Number):
            raise TypeError(".scale() takes a number as input")
        if factor == 1:
            return self
        from .scaling_operator import ScalingOperator
        return ScalingOperator(self.target, factor)(self)

    def conjugate(self):
        from .simple_linear_operators import ConjugationOperator
        return ConjugationOperator(self.target)(self)

    def sum(self, spaces=None):
        from .contraction_operator import ContractionOperator
        return ContractionOperator(self.target, spaces)(self)

    def integrate(self, spaces=None):
        from .contraction_operator import IntegrationOperator
        return IntegrationOperator(self.target, spaces)(self)

    def vdot(self, other):
        from ..sugar import makeOp
        if not isinstance(other, Operator):
            raise TypeError
        if other.jac is None:
            res = self.conjugate()*other
        else:
            res = makeOp(other) @ self.conjugate()
        return res.sum()

    @property
    def real(self):
        from .simple_linear_operators import Realizer
        return Realizer(self.target)(self)

    @property
    def imag(self):
        from .simple_linear_operators import Imaginizer
        return Imaginizer(self.target)(self)

    def __neg__(self):
        return self.scale(-1)

    def __matmul__(self, x):
        from .energy_operators import LikelihoodEnergyOperator

        if not isinstance(x, Operator):
            return NotImplemented
        if isinstance(x, LikelihoodEnergyOperator):
            return NotImplemented
        if x.target is self.domain:
            return _OpChain.make((self, x))
        return self.partial_insert(x)

    def __rmatmul__(self, x):
        from .energy_operators import LikelihoodEnergyOperator

        if not isinstance(x, Operator):
            return NotImplemented
        if isinstance(x, LikelihoodEnergyOperator):
            return NotImplemented
        if x.domain is self.target:
            return _OpChain.make((x, self))
        return x.partial_insert(self)

    def partial_insert(self, x):
        from ..multi_domain import MultiDomain
        if not isinstance(x, Operator):
            raise TypeError
        if not isinstance(self.domain, MultiDomain):
            raise TypeError
        if not isinstance(x.target, MultiDomain):
            raise TypeError
        bigdom = MultiDomain.union([self.domain, x.target])
        k1, k2 = set(self.domain.keys()), set(x.target.keys())
        le, ri = k2 - k1, k1 - k2
        leop, riop = self, x
        if len(ri) > 0:
            riop = riop + self.identity_operator(
                MultiDomain.make({kk: bigdom[kk]
                                  for kk in ri}))
        if len(le) > 0:
            leop = leop + self.identity_operator(
                MultiDomain.make({kk: bigdom[kk]
                                  for kk in le}))
        return leop @ riop

    @staticmethod
    def identity_operator(dom):
        from ..sugar import makeDomain
        from .block_diagonal_operator import BlockDiagonalOperator
        from .scaling_operator import ScalingOperator

        dom = makeDomain(dom)
        if isinstance(dom, DomainTuple):
            return ScalingOperator(dom, 1.)
        idops = {kk: ScalingOperator(dd, 1.) for kk, dd in dom.items()}
        return BlockDiagonalOperator(dom, idops)

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

    def __abs__(self):
        return self.ptw("abs")

    def __pow__(self, power):
        if not (np.isscalar(power) or power.jac is None):
            return NotImplemented
        return self.ptw("power", power)

    def __getitem__(self, key):
        from ..sugar import is_operator
        from .simple_linear_operators import ducktape

        if not is_operator(self):
            return NotImplemented
        if not isinstance(self.target, MultiDomain):
            raise TypeError("Only Operators with a MultiDomain as target can be subscripted.")
        return ducktape(None, self, key) @ self

    def apply(self, x):
        """Applies the operator to a Field, MultiField or Linearization.

        Parameters
        ----------
        x : :class:`nifty8.field.Field`, :class:`nifty8.multi_field.MultiField`,
            or :class:`nifty8.linearization.Linearization`
            Input on which the operator shall act. Needs to be defined on
            :attr:`domain`. If `x`is a :class:`nifty8.linearization.Linearization`,
            `apply` returns a new :class:`nifty8.linearization.Linearization`
            contining the result of the operator application as well as its
            Jacobian, evaluated at `x`.
        """
        raise NotImplementedError

    def force(self, x):
        """Extract subset of domain of x according to `self.domain` and apply
        operator."""
        return self.apply(x.extract(self.domain))

    def _check_input(self, x):
        from .scaling_operator import ScalingOperator
        if not (isinstance(x, Operator) and x.val is not None):
            raise TypeError
        if x.jac is not None:
            if not isinstance(x.jac, ScalingOperator):
                raise ValueError
            if x.jac._factor != 1:
                raise ValueError
        check_object_identity(self._domain, x.domain)

    def __call__(self, x):
        if not isinstance(x, Operator):
            raise TypeError
        if x.jac is not None:
            return self.apply(x.trivial_jac()).prepend_jac(x.jac)
        elif x.val is not None:
            return self.apply(x)
        return self @ x

    def ducktape(self, name):
        from ..sugar import is_operator, makeDomain
        from .simple_linear_operators import DomainChangerAndReshaper, ducktape

        if not is_operator(self):
            raise RuntimeError("ducktape works only on operators")

        if isinstance(name, str):  # convert to MultiDomain
            return self @ ducktape(self, None, name)
        else:  # convert domain
            newdom = makeDomain(name)
            return self @ DomainChangerAndReshaper(newdom, self.domain)

    def ducktape_left(self, name):
        from ..sugar import is_fieldlike, is_operator, makeDomain
        from .simple_linear_operators import DomainChangerAndReshaper, ducktape

        if isinstance(name, str):  # convert to MultiDomain
            tgt = self.target if is_operator(self) else self.domain
            return ducktape(None, tgt, name)(self)
        else:  # convert domain
            newdom = DomainTuple.make(name)
            dom = self.domain if is_fieldlike(self) else self.target
            return DomainChangerAndReshaper(dom, newdom)(self)

    def transpose(self, indices):
        """Transposes a Field.

        Parameters
        ----------
        indices : tuple
            Must be a tuple or list which contains a permutation of
            [0,1,..,N-1] where N is the number of domains in the target of the
            Operator (or the Field).
        """
        from ..sugar import is_fieldlike
        from .transpose_operator import TransposeOperator

        dom = self.domain if is_fieldlike(self) else self.target
        return TransposeOperator(dom, indices)(self)

    def __repr__(self):
        return self.__class__.__name__

    def simplify_for_constant_input(self, c_inp):
        from ..multi_field import MultiField
        from ..sugar import makeDomain
        from .energy_operators import EnergyOperator, LikelihoodEnergyOperator
        from .simplify_for_const import (ConstantEnergyOperator,
                                         ConstantLikelihoodEnergyOperator,
                                         ConstantOperator)
        if c_inp is None or (isinstance(c_inp, MultiField) and len(c_inp.keys()) == 0):
            return None, self
        dom = c_inp.domain
        if isinstance(dom, MultiDomain) and len(dom) == 0:
            return None, self

        # Convention: If c_inp is MultiField, it needs to be defined on a
        # subdomain of self._domain
        if isinstance(self.domain, MultiDomain):
            myassert(isinstance(dom, MultiDomain))
            if not set(c_inp.keys()) <= set(self.domain.keys()):
                raise ValueError

        if dom is self.domain:
            if isinstance(self, DomainTuple):
                raise RuntimeError
            if isinstance(self, EnergyOperator):
                op = ConstantEnergyOperator(self(c_inp))
            elif isinstance(self, LikelihoodEnergyOperator):
                op = ConstantLikelihoodEnergyOperator(self(c_inp))
            else:
                op = ConstantOperator(self(c_inp))
            return None, op
        if not isinstance(dom, MultiDomain):
            raise RuntimeError
        c_out, op = self._simplify_for_constant_input_nontrivial(c_inp)
        vardom = makeDomain({kk: vv for kk, vv in self.domain.items()
                             if kk not in c_inp.keys()})
        myassert(op.domain is vardom)
        myassert(op.target is self.target)
        myassert(isinstance(op, Operator))
        if c_out is not None:
            myassert(isinstance(c_out, MultiField))
            myassert(len(set(c_out.keys()) & self.domain.keys()) == 0)
            myassert(set(c_out.keys()) <= set(c_inp.keys()))
        return c_out, op

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        from .simplify_for_const import InsertionOperator
        logger.warning('SlowPartialConstantOperator used for:')
        logger.warning(self.__repr__())
        return None, self @ InsertionOperator(self.domain, c_inp)

    def ptw(self, op, *args, **kwargs):
        return _OpChain.make((_FunctionApplier(self.target, op, *args, **kwargs), self))

    def ptw_pre(self, op, *args, **kwargs):
        return _OpChain.make((self, _FunctionApplier(self.domain, op, *args, **kwargs)))

    def apply_to_random_sample(self, **kwargs):
        """Applies the operator to a sample drawn with `ift.sugar.from_random`.
        Keyword arguments are passed through to the sample generation.

        .. warning::
            If not specified otherwise, the sample will be of dtype
            `np.float64`. If the operator requires input values of other
            dtypes, this needs to be indicated with the `dtype` keyword argument.
        """
        from ..sugar import from_random
        random_input = from_random(self.domain, **kwargs)
        return self(random_input)


for f in pointwise.ptw_dict.keys():
    def func(f):
        def func2(self, *args, **kwargs):
            return self.ptw(f, *args, **kwargs)
        return func2
    setattr(Operator, f, func(f))
    def func(f):
        def func2(self, *args, **kwargs):
            return self.ptw_pre(f, *args, **kwargs)
        return func2
    setattr(Operator, f + "_pre", func(f))


class _FunctionApplier(Operator):
    def __init__(self, domain, funcname, *args, **kwargs):
        from ..sugar import makeDomain
        self._domain = self._target = makeDomain(domain)
        self._funcname = funcname
        self._args = args
        self._kwargs = kwargs

        try:
            import jax.numpy as jnp
            from jax import nn as jax_nn

            if funcname in pointwise.ptw_nifty2jax_dict:
                jax_expr = pointwise.ptw_nifty2jax_dict[funcname]
            elif hasattr(jnp, funcname):
                jax_expr = getattr(jnp, funcname)
            elif hasattr(jax_nn, funcname):
                jax_expr = getattr(jax_nn, funcname)
            else:
                # warn(f"unable to add JAX call for {funcname!r}")
                jax_expr = None

            def jax_expr_part(x):  # Partial insert with first open argument
                return jax_expr(x, *args, **kwargs)

            if isinstance(self.domain, MultiDomain):
                from functools import partial

                from jax.tree_util import tree_map

                jax_expr_part = partial(tree_map, jax_expr_part)
            self._jax_expr = jax_expr_part
        except ImportError:
            self._jax_expr = None

    def apply(self, x):
        self._check_input(x)
        return x.ptw(self._funcname, *self._args, **self._kwargs)

    def __repr__(self):
        return f"_FunctionApplier ('{self._funcname}')"


class _CombinedOperator(Operator):
    def __init__(self, ops, jax_ops, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        self._ops = tuple(ops)

        if all(callable(jop) for jop in jax_ops):

            def joined_jax_op(x):
                for jop in reversed(jax_ops):
                    x = jop(x)
                return x

            self._jax_expr = joined_jax_op
        else:
            self._jax_expr = None

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
        jax_res = tuple(op.jax_expr for op in ops)
        return cls(res, jax_res, _callingfrommake=True)


class _OpChain(_CombinedOperator):
    def __init__(self, ops, jax_ops, _callingfrommake=False):
        super(_OpChain, self).__init__(ops, jax_ops, _callingfrommake)
        self._domain = self._ops[-1].domain
        self._target = self._ops[0].target
        for i in range(1, len(self._ops)):
            check_object_identity(self._ops[i-1].domain, self._ops[i].target)

    def apply(self, x):
        self._check_input(x)
        for op in reversed(self._ops):
            x = op(x)
        return x

    def _simplify_for_constant_input_nontrivial(self, c_inp):
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

        lhs_has_jax = callable(self._op1.jax_expr)
        rhs_has_jax = callable(self._op2.jax_expr)
        if lhs_has_jax and rhs_has_jax:

            def joined_jax_expr(x):
                return self._op1.jax_expr(x) * self._op2.jax_expr(x)

            self._jax_expr = joined_jax_expr
        else:
            self._jax_expr = None

    def apply(self, x):
        from ..linearization import Linearization
        from ..sugar import makeOp
        self._check_input(x)
        lin = x.jac is not None
        wm = x.want_metric if lin else False
        x = x.val if lin else x
        v1 = x.extract(self._op1.domain)
        v2 = x.extract(self._op2.domain)
        if not lin:
            return self._op1(v1) * self._op2(v2)
        lin1 = self._op1(Linearization.make_var(v1, wm))
        lin2 = self._op2(Linearization.make_var(v2, wm))
        jac = (makeOp(lin1._val)(lin2._jac))._myadd(makeOp(lin2._val)(lin1._jac), False)
        return lin1.new(lin1._val*lin2._val, jac)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        from ..multi_domain import MultiDomain
        from .simplify_for_const import ConstCollector
        f1, o1 = self._op1.simplify_for_constant_input(
            c_inp.extract_part(self._op1.domain))
        f2, o2 = self._op2.simplify_for_constant_input(
            c_inp.extract_part(self._op2.domain))
        if not isinstance(self._target, MultiDomain):
            return None, _OpProd(o1, o2)
        cc = ConstCollector()
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

        try:
            from ..re import unite

            def joined_jax_expr(x):
                return unite(self._op1.jax_expr(x), self._op2.jax_expr(x))

            self._jax_expr = joined_jax_expr
        except ImportError:
            self._jax_expr = None

    def apply(self, x):
        self._check_input(x)
        return self._apply_operator_sum(x, [self._op1, self._op2])

    @staticmethod
    def _apply_operator_sum(x, ops):
        from ..linearization import Linearization

        unite = lambda x, y: x.unite(y)
        if x.jac is None:
            return reduce(unite, (oo.force(x) for oo in ops))
        lin = [oo(Linearization.make_var(x.val.extract(oo.domain), x.want_metric))
                for oo in ops]
        jacs = map(lambda x: x._jac, lin)
        vals = map(lambda x: x._val, lin)
        metrics = list(map(lambda x: x._metric, lin))
        jac = reduce(lambda x, y: x._myadd(y, False), jacs)
        val = reduce(unite, vals)
        res = x.new(val, jac)
        if all(mm is not None for mm in metrics):
            res = res.add_metric(reduce(add, metrics))
        return res

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        from ..multi_domain import MultiDomain
        from .simplify_for_const import ConstCollector
        f1, o1 = self._op1.simplify_for_constant_input(
            c_inp.extract_part(self._op1.domain))
        f2, o2 = self._op2.simplify_for_constant_input(
            c_inp.extract_part(self._op2.domain))
        if not isinstance(self._target, MultiDomain):
            return None, _OpSum(o1, o2)
        cc = ConstCollector()
        cc.add(f1, o1.target)
        cc.add(f2, o2.target)
        return cc.constfield, _OpSum(o1, o2)

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in (self._op1, self._op2))
        return "_OpSum:\n"+indent(subs)
