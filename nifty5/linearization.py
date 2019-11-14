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
from . import utilities


class Linearization(object):
    """Let `A` be an operator and `x` a field. `Linearization` stores the value
    of the operator application (i.e. `A(x)`), the local Jacobian
    (i.e. `dA(x)/dx`) and, optionally, the local metric.

    Parameters
    ----------
    val : Field or MultiField
        The value of the operator application.
    jac : LinearOperator
        The Jacobian.
    metric : LinearOperator or None
        The metric. Default: None.
    want_metric : bool
        If True, the metric will be computed for other Linearizations derived
        from this one. Default: False.
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
        val : Field or MultiField
            the value of the operator application
        jac : LinearOperator
            the Jacobian
        metric : LinearOperator or None
            The metric. Default: None.
        """
        return Linearization(val, jac, metric, self._want_metric)

    @property
    def domain(self):
        """DomainTuple or MultiDomain : the Jacobian's domain"""
        return self._jac.domain

    @property
    def target(self):
        """DomainTuple or MultiDomain : the Jacobian's target (i.e. the value's domain)"""
        return self._jac.target

    @property
    def val(self):
        """Field or MultiField : the value"""
        return self._val

    @property
    def jac(self):
        """LinearOperator : the Jacobian"""
        return self._jac

    @property
    def gradient(self):
        """Field or MultiField : the gradient

        Notes
        -----
        Only available if target is a scalar
        """
        return self._jac.adjoint_times(Field.scalar(1.))

    @property
    def want_metric(self):
        """bool : True iff the metric was requested in the constructor"""
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
        return self.new(self._val[name], self._jac.ducktape_left(name))

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
        """Computes the outer product of this Linearization with a Field or
        another Linearization

        Parameters
        ----------
        other : Field or MultiField or Linearization

        Returns
        -------
        Linearization
            the outer product of self and other
        """
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
        """Computes the inner product of this Linearization with a Field or
        another Linearization

        Parameters
        ----------
        other : Field or MultiField or Linearization

        Returns
        -------
        Linearization
            the inner product of self and other
        """
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
        """Computes the (partial) sum over self

        Parameters
        ----------
        spaces : None, int or list of int
            - if None, sum over the entire domain
            - else sum over the specified subspaces

        Returns
        -------
        Linearization
            the (partial) sum
        """
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
        """Computes the (partial) integral over self

        Parameters
        ----------
        spaces : None, int or list of int
            - if None, integrate over the entire domain
            - else integrate over the specified subspaces

        Returns
        -------
        Linearization
            the (partial) integral
        """
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

    def sqrt(self):
        tmp = self._val.sqrt()
        return self.new(tmp, makeOp(0.5/tmp)(self._jac))

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
        tmp2 = ((np.pi*self._val).cos()-tmp)/self._val
        ind = self._val.local_data == 0
        loc = tmp2.local_data.copy()
        loc[ind] = 0
        tmp2 = Field.from_local_data(tmp.domain, loc)
        return self.new(tmp, makeOp(tmp2)(self._jac))

    def log(self):
        tmp = self._val.log()
        return self.new(tmp, makeOp(1./self._val)(self._jac))

    def log10(self):
        tmp = self._val.log10()
        tmp2 = 1. / (self._val * np.log(10))
        return self.new(tmp, makeOp(tmp2)(self._jac))

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
        if utilities.iscomplextype(self._val.dtype):
            raise TypeError("Argument must not be complex")
        tmp = self._val.absolute()
        tmp2 = self._val.sign()

        ind = self._val.local_data == 0
        loc = tmp2.local_data.copy().astype(float)
        loc[ind] = np.nan
        tmp2 = Field.from_local_data(tmp.domain, loc)

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
        """Converts a Field to a Linearization, with a unity Jacobian

        Parameters
        ----------
        field : Field or Multifield
            the field to be converted
        want_metric : bool
            If True, the metric will be computed for other Linearizations
            derived from this one. Default: False.

        Returns
        -------
        Linearization
            the requested Linearization
        """
        from .operators.scaling_operator import ScalingOperator
        return Linearization(field, ScalingOperator(1., field.domain),
                             want_metric=want_metric)

    @staticmethod
    def make_const(field, want_metric=False):
        """Converts a Field to a Linearization, with a zero Jacobian

        Parameters
        ----------
        field : Field or Multifield
            the field to be converted
        want_metric : bool
            If True, the metric will be computed for other Linearizations
            derived from this one. Default: False.

        Returns
        -------
        Linearization
            the requested Linearization

        Notes
        -----
        The Jacobian is square and contains only zeroes.
        """
        from .operators.simple_linear_operators import NullOperator
        return Linearization(field, NullOperator(field.domain, field.domain),
                             want_metric=want_metric)

    @staticmethod
    def make_const_empty_input(field, want_metric=False):
        """Converts a Field to a Linearization, with a zero Jacobian

        Parameters
        ----------
        field : Field or Multifield
            the field to be converted
        want_metric : bool
            If True, the metric will be computed for other Linearizations
            derived from this one. Default: False.

        Returns
        -------
        Linearization
            the requested Linearization

        Notes
        -----
        The Jacobian has an empty input domain, i.e. its matrix representation
        has 0 columns.
        """
        from .operators.simple_linear_operators import NullOperator
        from .multi_domain import MultiDomain
        return Linearization(
            field, NullOperator(MultiDomain.make({}), field.domain),
            want_metric=want_metric)

    @staticmethod
    def make_partial_var(field, constants, want_metric=False):
        """Converts a MultiField to a Linearization, with a Jacobian that is
        unity for some MultiField components and a zero matrix for others.

        Parameters
        ----------
        field : Multifield
            the field to be converted
        constants : list of string
            the MultiField components for which the Jacobian should be
            a zero matrix.
        want_metric : bool
            If True, the metric will be computed for other Linearizations
            derived from this one. Default: False.

        Returns
        -------
        Linearization
            the requested Linearization

        Notes
        -----
        The Jacobian is square.
        """
        from .operators.scaling_operator import ScalingOperator
        from .operators.block_diagonal_operator import BlockDiagonalOperator
        if len(constants) == 0:
            return Linearization.make_var(field, want_metric)
        else:
            ops = {key: ScalingOperator(0. if key in constants else 1., dom)
                   for key, dom in field.domain.items()}
            bdop = BlockDiagonalOperator(field.domain, ops)
            return Linearization(field, bdop, want_metric=want_metric)
