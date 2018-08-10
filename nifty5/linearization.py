from __future__ import absolute_import, division, print_function

import numpy as np

from .compat import *
from .field import Field
from .multi_field import MultiField
from .sugar import makeOp
from .domain_tuple import DomainTuple


class Linearization(object):
    def __init__(self, val, jac, metric=None):
        self._val = val
        self._jac = jac
        if self._val.domain != self._jac.target:
            raise ValueError("domain mismatch")
        self._metric = metric

    @property
    def domain(self):
        return self._jac.domain

    @property
    def target(self):
        return self._jac.target

    @property
    def val(self):
        return self._val

    @property
    def jac(self):
        return self._jac

    @property
    def gradient(self):
        """Only available if target is a scalar"""
        return self._jac.adjoint_times(Field.scalar(1.))

    @property
    def metric(self):
        """Only available if target is a scalar"""
        return self._metric

    def __getitem__(self, name):
        from .operators.simple_linear_operators import FieldAdapter
        return Linearization(self._val[name], FieldAdapter(self.domain, name))

    def __neg__(self):
        return Linearization(
            -self._val, -self._jac,
            None if self._metric is None else -self._metric)

    def conjugate(self):
        return Linearization(
            self._val.conjugate(), self._jac.conjugate(),
            None if self._metric is None else self._metric.conjugate())

    @property
    def real(self):
        return Linearization(self._val.real, self._jac.real)

    def __add__(self, other):
        if isinstance(other, Linearization):
            met = None
            if self._metric is not None and other._metric is not None:
                met = self._metric._myadd(other._metric, False)
            return Linearization(
                self._val.unite(other._val),
                self._jac._myadd(other._jac, False), met)
        if isinstance(other, (int, float, complex, Field, MultiField)):
            return Linearization(self._val+other, self._jac, self._metric)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        from .sugar import makeOp
        if isinstance(other, Linearization):
            if self.target != other.target:
                raise ValueError("domain mismatch")
            return Linearization(
                self._val*other._val,
                (makeOp(other._val)(self._jac))._myadd(
                 makeOp(self._val)(other._jac), False))
        if np.isscalar(other):
            if other == 1:
                return self
            met = None if self._metric is None else self._metric.scale(other)
            return Linearization(self._val*other, self._jac.scale(other), met)
        if isinstance(other, (Field, MultiField)):
            if self.target != other.domain:
                raise ValueError("domain mismatch")
            return Linearization(self._val*other, makeOp(other)(self._jac))

    def __rmul__(self, other):
        return self.__mul__(other)

    def vdot(self, other):
        from .domain_tuple import DomainTuple
        from .operators.simple_linear_operators import VdotOperator
        if isinstance(other, (Field, MultiField)):
            return Linearization(
                Field.scalar(self._val.vdot(other)),
                VdotOperator(other)(self._jac))
        return Linearization(
            Field.scalar(self._val.vdot(other._val)),
            VdotOperator(self._val)(other._jac) +
            VdotOperator(other._val)(self._jac))

    def sum(self):
        from .operators.simple_linear_operators import SumReductionOperator
        from .sugar import full
        return Linearization(
            Field.scalar(self._val.sum()),
            SumReductionOperator(self._jac.target)(self._jac))

    def exp(self):
        tmp = self._val.exp()
        return Linearization(tmp, makeOp(tmp)(self._jac))

    def log(self):
        tmp = self._val.log()
        return Linearization(tmp, makeOp(1./self._val)(self._jac))

    def tanh(self):
        tmp = self._val.tanh()
        return Linearization(tmp, makeOp(1.-tmp**2)(self._jac))

    def positive_tanh(self):
        tmp = self._val.tanh()
        tmp2 = 0.5*(1.+tmp)
        return Linearization(tmp2, makeOp(0.5*(1.-tmp**2))(self._jac))

    def add_metric(self, metric):
        return Linearization(self._val, self._jac, metric)

    @staticmethod
    def make_var(field):
        from .operators.scaling_operator import ScalingOperator
        return Linearization(field, ScalingOperator(1., field.domain))

    @staticmethod
    def make_const(field):
        from .operators.simple_linear_operators import NullOperator
        return Linearization(field, NullOperator(field.domain, field.domain))
