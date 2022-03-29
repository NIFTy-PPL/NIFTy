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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .operators.operator import Operator
from .sugar import makeOp
from .utilities import check_object_identity


class Linearization(Operator):
    """Let `A` be an operator and `x` a field. `Linearization` stores the value
    of the operator application (i.e. `A(x)`), the local Jacobian
    (i.e. `dA(x)/dx`) and, optionally, the local metric.

    Parameters
    ----------
    val : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField`
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
        check_object_identity(self._val.domain, self._jac.target)
        self._want_metric = want_metric
        self._metric = metric

    def new(self, val, jac, metric=None):
        """Create a new Linearization, taking the `want_metric` property from
           this one.

        Parameters
        ----------
        val : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField`
            the value of the operator application
        jac : LinearOperator
            the Jacobian
        metric : LinearOperator or None
            The metric. Default: None.
        """
        return Linearization(val, jac, metric, self._want_metric)

    def trivial_jac(self):
        return self.make_var(self._val, self._want_metric)

    def prepend_jac(self, jac):
        if self._metric is None:
            return self.new(self._val, self._jac @ jac)
        from .operators.sandwich_operator import SandwichOperator
        metric = SandwichOperator.make(jac, self._metric)
        return self.new(self._val, self._jac @ jac, metric)

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
        """:class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField` : the value"""
        return self._val

    @property
    def jac(self):
        """LinearOperator : the Jacobian"""
        return self._jac

    @property
    def gradient(self):
        """:class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField` : the gradient

        Notes
        -----
        Only available if target is a scalar
        """
        from .field import Field
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
        if self._metric is not None:
            raise RuntimeError("Cannot negate operators with metric")
        return self.new(-self._val, -self._jac, metric=None)

    def conjugate(self):
        return self.new(
            self._val.conjugate(), self._jac.conjugate(),
            None if self._metric is None else self._metric.conjugate())

    @property
    def real(self):
        return self.new(self._val.real, self._jac.real)

    @property
    def imag(self):
        return self.new(self._val.imag, self._jac.imag)

    def _myadd(self, other, neg):
        if np.isscalar(other) or other.jac is None:
            return self.new(self._val-other if neg else self._val+other,
                            self._jac, self._metric)
        met = None
        if self._metric is not None and other._metric is not None:
            met = self._metric._myadd(other._metric, neg)
        return self.new(
            self.val.flexible_addsub(other.val, neg),
            self.jac._myadd(other.jac, neg), met)

    def __add__(self, other):
        return self._myadd(other, False)

    def __radd__(self, other):
        return self._myadd(other, False)

    def __sub__(self, other):
        return self._myadd(other, True)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __truediv__(self, other):
        if np.isscalar(other):
            return self.__mul__(1/other)
        return self.__mul__(other.ptw("reciprocal"))

    def __rtruediv__(self, other):
        return self.ptw("reciprocal").__mul__(other)

    def __pow__(self, power):
        if not (np.isscalar(power) or power.jac is None):
            return NotImplemented
        return self.ptw("power", power)

    def __mul__(self, other):
        if np.isscalar(other):
            if other == 1:
                return self
            met = None if self._metric is None else self._metric.scale(other)
            return self.new(self._val*other, self._jac.scale(other), met)
        if other.jac is None:
            check_object_identity(self.target, other.domain)
            return self.new(self._val*other, makeOp(other)(self._jac))
        check_object_identity(self.target, other.target)
        return self.new(
            self.val*other.val,
            (makeOp(other.val)(self.jac))._myadd(
             makeOp(self.val)(other.jac), False))

    def __rmul__(self, other):
        return self.__mul__(other)

    def outer(self, other):
        """Computes the outer product of this Linearization with a Field or
        another Linearization

        Parameters
        ----------
        other : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField` or Linearization

        Returns
        -------
        Linearization
            the outer product of self and other
        """
        if np.isscalar(other):
            return self.__mul__(other)
        from .operators.outer_product_operator import OuterProduct
        if other.jac is None:
            return self.new(OuterProduct(other.domain, self._val)(other),
                            OuterProduct(other.domain, self._jac(self._val)))
        tmp_op = OuterProduct(other.target, self._val)
        return self.new(
            tmp_op(other._val),
            OuterProduct(other.target, self._jac(self._val))._myadd(
                tmp_op(other._jac), False))

    def vdot(self, other):
        """Computes the inner product of this Linearization with a Field or
        another Linearization

        Parameters
        ----------
        other : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField` or Linearization

        Returns
        -------
        Linearization
            the inner product of self and other
        """
        from .operators.simple_linear_operators import VdotOperator
        if other.jac is None:
            return self.new(
                self._val.vdot(other),
                VdotOperator(other)(self._jac))
        return self.new(
            self._val.vdot(other._val),
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
        from .operators.contraction_operator import IntegrationOperator
        return IntegrationOperator(self.target, spaces)(self)

    def ptw(self, op, *args, **kwargs):
        t1, t2 = self._val.ptw_with_deriv(op, *args, **kwargs)
        return self.new(t1, makeOp(t2)(self._jac))

    def add_metric(self, metric):
        return self.new(self._val, self._jac, metric)

    def with_want_metric(self):
        return Linearization(self._val, self._jac, self._metric, True)

    @staticmethod
    def make_var(field, want_metric=False):
        """Converts a Field to a Linearization, with a unity Jacobian

        Parameters
        ----------
        field : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField`
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
        return Linearization(field, ScalingOperator(field.domain, 1.),
                             want_metric=want_metric)

    @staticmethod
    def make_const(field, want_metric=False):
        """Converts a Field to a Linearization, with a zero Jacobian

        Parameters
        ----------
        field : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField`
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
        field : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField`
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
        from .multi_domain import MultiDomain
        from .operators.simple_linear_operators import NullOperator
        return Linearization(
            field, NullOperator(MultiDomain.make({}), field.domain),
            want_metric=want_metric)

    @staticmethod
    def make_partial_var(field, constants, want_metric=False):
        """Converts a MultiField to a Linearization, with a Jacobian that is
        unity for some MultiField components and a zero matrix for others.

        Parameters
        ----------
        field ::class:`nifty8.multi_field.MultiField`
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
        from .operators.block_diagonal_operator import BlockDiagonalOperator
        from .operators.scaling_operator import ScalingOperator
        if len(constants) == 0:
            return Linearization.make_var(field, want_metric)
        else:
            ops = {key: ScalingOperator(dom, 0. if key in constants else 1.)
                   for key, dom in field.domain.items()}
            bdop = BlockDiagonalOperator(field.domain, ops)
            return Linearization(field, bdop, want_metric=want_metric)
