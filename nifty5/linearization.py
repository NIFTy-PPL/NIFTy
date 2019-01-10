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

from .field import Field
from .multi_field import MultiField
from .sugar import makeOp
from .operators.scaling_operator import ScalingOperator


class Linearization(object):
    """Let `A` be an operator and `x` a field. `Linearization` stores the value
    of the operator application (i.e. `A(x)`), the local Jacobian
    (i.e. `dA(x)/dx`) and, optionally, the local metric.

    Parameters
    ----------
    val : Field/MultiField
        the value of the operator application
    jac : LinearOperator
        the Jacobian
    metric : LinearOperator or None (default: None)
        the metric
    want_metric : bool (default: False)
        if True, the metric will be computed for other Linearizations derived
        from this one.
    """
    def __init__(self, val, jac, metric=None, want_metric=False):
        self._val = val
        self._jac = jac
        if self._val.domain != self._jac.target:
            raise ValueError("domain mismatch")
        self._want_metric = want_metric
        self._metric = metric

    def new(self, val, jac, metric=None):
        """Create a new Linearization, taking the `want_metric` property from
           this one.

        Parameters
        ----------
        val : Field/MultiField
            the value of the operator application
        jac : LinearOperator
            the Jacobian
        metric : LinearOperator or None (default: None)
            the metric
        """
        return Linearization(val, jac, metric, self._want_metric)

    @property
    def domain(self):
        """DomainTuple/MultiDomain : the Jacobian's domain"""
        return self._jac.domain

    @property
    def target(self):
        """DomainTuple/MultiDomain : the Jacobian's target (i.e. the value's domain)"""
        return self._jac.target

    @property
    def val(self):
        """Field/MultiField : the value"""
        return self._val

    @property
    def jac(self):
        """LinearOperator : the Jacobian"""
        return self._jac

    @property
    def gradient(self):
        """Field/MultiField : the gradient

        Notes
        -----
        Only available if target is a scalar
        """
        return self._jac.adjoint_times(Field.scalar(1.))

    @property
    def want_metric(self):
        """bool : the value of `want_metric`"""
        return self._want_metric

    @property
    def metric(self):
        """LinearOperator : the metric

        Notes
        -----
        Only available if target is a scalar
        """
        return self._metric

    def __getitem__(self, name):
        from .operators.simple_linear_operators import ducktape
        return self.new(self._val[name], ducktape(None, self.domain, name))

    def __neg__(self):
        return self.new(-self._val, -self._jac,
                        None if self._metric is None else -self._metric)

    def conjugate(self):
        return self.new(
            self._val.conjugate(), self._jac.conjugate(),
            None if self._metric is None else self._metric.conjugate())

    @property
    def real(self):
        return self.new(self._val.real, self._jac.real)

    def _myadd(self, other, neg):
        if isinstance(other, Linearization):
            met = None
            if self._metric is not None and other._metric is not None:
                met = self._metric._myadd(other._metric, neg)
            return self.new(
                self._val.flexible_addsub(other._val, neg),
                self._jac._myadd(other._jac, neg), met)
        if isinstance(other, (int, float, complex, Field, MultiField)):
            if neg:
                return self.new(self._val-other, self._jac, self._metric)
            else:
                return self.new(self._val+other, self._jac, self._metric)

    def __add__(self, other):
        return self._myadd(other, False)

    def __radd__(self, other):
        return self._myadd(other, False)

    def __sub__(self, other):
        return self._myadd(other, True)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __truediv__(self, other):
        if isinstance(other, Linearization):
            return self.__mul__(other.one_over())
        return self.__mul__(1./other)

    def __rtruediv__(self, other):
        return self.one_over().__mul__(other)

    def __pow__(self, power):
        if not np.isscalar(power):
            return NotImplemented
        return self.new(self._val**power,
                        makeOp(self._val**(power-1)).scale(power)(self._jac))

    def __mul__(self, other):
        from .sugar import makeOp
        if isinstance(other, Linearization):
            if self.target != other.target:
                raise ValueError("domain mismatch")
            return self.new(
                self._val*other._val,
                (makeOp(other._val)(self._jac))._myadd(
                 makeOp(self._val)(other._jac), False))
        if np.isscalar(other):
            if other == 1:
                return self
            met = None if self._metric is None else self._metric.scale(other)
            return self.new(self._val*other, self._jac.scale(other), met)
        if isinstance(other, (Field, MultiField)):
            if self.target != other.domain:
                raise ValueError("domain mismatch")
            return self.new(self._val*other, makeOp(other)(self._jac))

    def __rmul__(self, other):
        return self.__mul__(other)

    def outer(self, other):
        from .operators.outer_product_operator import OuterProduct
        if isinstance(other, Linearization):
            return self.new(
                OuterProduct(self._val, other.target)(other._val),
                OuterProduct(self._jac(self._val), other.target)._myadd(
                    OuterProduct(self._val, other.target)(other._jac), False))
        if np.isscalar(other):
            return self.__mul__(other)
        if isinstance(other, (Field, MultiField)):
            return self.new(OuterProduct(self._val, other.domain)(other),
                            OuterProduct(self._jac(self._val), other.domain))

    def vdot(self, other):
        from .operators.simple_linear_operators import VdotOperator
        if isinstance(other, (Field, MultiField)):
            return self.new(
                Field.scalar(self._val.vdot(other)),
                VdotOperator(other)(self._jac))
        return self.new(
            Field.scalar(self._val.vdot(other._val)),
            VdotOperator(self._val)(other._jac) +
            VdotOperator(other._val)(self._jac))

    def sum(self, spaces=None):
        from .operators.contraction_operator import ContractionOperator
        if spaces is None:
            return self.new(
                Field.scalar(self._val.sum()),
                ContractionOperator(self._jac.target, None)(self._jac))
        else:
            return self.new(
                self._val.sum(spaces),
                ContractionOperator(self._jac.target, spaces)(self._jac))

    def integrate(self, spaces=None):
        from .operators.contraction_operator import ContractionOperator
        if spaces is None:
            return self.new(
                Field.scalar(self._val.integrate()),
                ContractionOperator(self._jac.target, None, 1)(self._jac))
        else:
            return self.new(
                self._val.integrate(spaces),
                ContractionOperator(self._jac.target, spaces, 1)(self._jac))

    def exp(self):
        tmp = self._val.exp()
        return self.new(tmp, makeOp(tmp)(self._jac))

    def clip(self, min=None, max=None):
        tmp = self._val.clip(min, max)
        if (min is None) and (max is None):
            return self
        elif max is None:
            tmp2 = makeOp(1. - (tmp == min))
        elif min is None:
            tmp2 = makeOp(1. - (tmp == max))
        else:
            tmp2 = makeOp(1. - (tmp == min) - (tmp == max))
        return self.new(tmp, tmp2(self._jac))

    def sin(self):
        tmp = self._val.sin()
        tmp2 = self._val.cos()
        return self.new(tmp, makeOp(tmp2)(self._jac))

    def cos(self):
        tmp = self._val.cos()
        tmp2 = - self._val.sin()
        return self.new(tmp, makeOp(tmp2)(self._jac))

    def tan(self):
        tmp = self._val.tan()
        tmp2 = 1./(self._val.cos()**2)
        return self.new(tmp, makeOp(tmp2)(self._jac))

    def sinc(self):
        tmp = self._val.sinc()
        tmp2 = (self._val.cos()-tmp)/self._val
        return self.new(tmp, makeOp(tmp2)(self._jac))

    def log(self):
        tmp = self._val.log()
        return self.new(tmp, makeOp(1./self._val)(self._jac))

    def sinh(self):
        tmp = self._val.sinh()
        tmp2 = self._val.cosh()
        return self.new(tmp, makeOp(tmp2)(self._jac))

    def cosh(self):
        tmp = self._val.cosh()
        tmp2 = self._val.sinh()
        return self.new(tmp, makeOp(tmp2)(self._jac))

    def tanh(self):
        tmp = self._val.tanh()
        return self.new(tmp, makeOp(1.-tmp**2)(self._jac))

    def sigmoid(self):
        tmp = self._val.tanh()
        tmp2 = 0.5*(1.+tmp)
        return self.new(tmp2, makeOp(0.5*(1.-tmp**2))(self._jac))

    def absolute(self):
        tmp = self._val.absolute()
        tmp2 = self._val.sign()
        return self.new(tmp, makeOp(tmp2)(self._jac))

    def one_over(self):
        tmp = 1./self._val
        tmp2 = - tmp/self._val
        return self.new(tmp, makeOp(tmp2)(self._jac))

    def add_metric(self, metric):
        return self.new(self._val, self._jac, metric)

    def with_want_metric(self):
        return Linearization(self._val, self._jac, self._metric, True)

    @staticmethod
    def make_var(field, want_metric=False):
        from .operators.scaling_operator import ScalingOperator
        return Linearization(field, ScalingOperator(1., field.domain),
                             want_metric=want_metric)

    @staticmethod
    def make_const(field, want_metric=False):
        from .operators.simple_linear_operators import NullOperator
        return Linearization(field, NullOperator(field.domain, field.domain),
                             want_metric=want_metric)

    @staticmethod
    def make_const_empty_input(field, want_metric=False):
        from .operators.simple_linear_operators import NullOperator
        from .multi_domain import MultiDomain
        return Linearization(
            field, NullOperator(MultiDomain.make({}), field.domain),
            want_metric=want_metric)

    @staticmethod
    def make_partial_var(field, constants, want_metric=False):
        from .operators.scaling_operator import ScalingOperator
        from .operators.block_diagonal_operator import BlockDiagonalOperator
        if len(constants) == 0:
            return Linearization.make_var(field, want_metric)
        else:
            ops = [ScalingOperator(0. if key in constants else 1., dom)
                   for key, dom in field.domain.items()]
            bdop = BlockDiagonalOperator(field.domain, tuple(ops))
            return Linearization(field, bdop, want_metric=want_metric)
