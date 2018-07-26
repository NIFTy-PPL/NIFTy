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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function

import numpy as np

from ..compat import *
from ..field import Field
from .multi_domain import MultiDomain
from .. import utilities


class MultiField(object):
    def __init__(self, domain, val):
        """
        Parameters
        ----------
        domain: MultiDomain
        val: tuple containing Field entries
        """
        if not isinstance(domain, MultiDomain):
            raise TypeError("domain must be of type MultiDomain")
        if not isinstance(val, tuple):
            raise TypeError("val must be a tuple")
        if len(val) != len(domain):
            raise ValueError("length mismatch")
        for d, v in zip(domain._domains, val):
            if isinstance(v, Field):
                if v._domain is not d:
                    raise ValueError("domain mismatch")
            else:
                raise TypeError("bad entry in val (must be Field)")
        self._domain = domain
        self._val = val

    @staticmethod
    def from_dict(dict, domain=None):
        if domain is None:
            domain = MultiDomain.make({key: v._domain
                                       for key, v in dict.items()})
        return MultiField(domain, tuple(dict[key] for key in domain.keys()))

    def to_dict(self):
        return {key: val for key, val in zip(self._domain.keys(), self._val)}

    def __getitem__(self, key):
        return self._val[self._domain.idx[key]]

    def keys(self):
        return self._domain.keys()

    def items(self):
        return zip(self._domain.keys(), self._val)

    def values(self):
        return self._val

    @property
    def domain(self):
        return self._domain

#    @property
#    def dtype(self):
#        return {key: val.dtype for key, val in self._val.items()}

    def _transform(self, op):
        return MultiField(self._domain, tuple(op(v) for v in self._val))

    @property
    def real(self):
        """MultiField : The real part of the multi field"""
        return self._transform(lambda x: x.real)

    @property
    def imag(self):
        """MultiField : The imaginary part of the multi field"""
        return self._transform(lambda x: x.imag)

    @staticmethod
    def from_random(random_type, domain, dtype=np.float64, **kwargs):
        domain = MultiDomain.make(domain)
#        dtype = MultiField.build_dtype(dtype, domain)
        return MultiField(
            domain, tuple(Field.from_random(random_type, dom, dtype, **kwargs)
                          for dom in domain._domains))

    def _check_domain(self, other):
        if other._domain is not self._domain:
            raise ValueError("domains are incompatible.")

    def vdot(self, x):
        result = 0.
        self._check_domain(x)
        for v1, v2 in zip(self._val, x._val):
            result += v1.vdot(v2)
        return result

#    @staticmethod
#    def build_dtype(dtype, domain):
#        if isinstance(dtype, dict):
#            return dtype
#        if dtype is None:
#            dtype = np.float64
#        return {key: dtype for key in domain.keys()}

    @staticmethod
    def full(domain, val):
        domain = MultiDomain.make(domain)
        return MultiField(domain, tuple(Field.full(dom, val)
                          for dom in domain._domains))

    def to_global_data(self):
        return {key: val.to_global_data()
                for key, val in zip(self._domain.keys(), self._val)}

    @staticmethod
    def from_global_data(domain, arr, sum_up=False):
        return MultiField(
            domain, tuple(Field.from_global_data(domain[key], arr[key], sum_up)
                          for key in domain.keys()))

    def norm(self):
        """ Computes the L2-norm of the field values.

        Returns
        -------
        norm : float
            The L2-norm of the field values.
        """
        return np.sqrt(np.abs(self.vdot(x=self)))

    def sum(self):
        """ Computes the sum all field values.

        Returns
        -------
        norm : float
            The sum of the field values.
        """
        return utilities.my_sum(map(lambda v: v.sum(), self._val))

    @property
    def size(self):
        """ Computes the overall degrees of freedom.

        Returns
        -------
        size : int
            The sum of the size of the individual fields
        """
        return utilities.my_sum(map(lambda d: d.size, self._domain.domains()))

    def squared_norm(self):
        """ Computes the square of the L2-norm of the field values.

        Returns
        -------
        float
            The square of the L2-norm of the field values.
        """
        return abs(self.vdot(x=self))

    def __neg__(self):
        return self._transform(lambda x: -x)

    def __abs__(self):
        return self._transform(lambda x: abs(x))

    def conjugate(self):
        return self._transform(lambda x: x.conjugate())

    def all(self):
        for v in self._val:
            if not v.all():
                return False
        return True

    def any(self):
        for v in self._val:
            if v.any():
                return True
        return False

    def isEquivalentTo(self, other):
        """Determines (as quickly as possible) whether `self`'s content is
        identical to `other`'s content."""
        if self is other:
            return True
        if not isinstance(other, MultiField):
            return False
        if self._domain is not other._domain:
            return False
        for v1, v2 in zip(self._val, other._val):
            if not v1.isEquivalentTo(v2):
                return False
        return True

    def extract(self, subset):
        if isinstance(subset, MultiDomain):
            return MultiField(subset,
                              tuple(self[key] for key in subset.keys()))
        else:
            return MultiField.from_dict({key: self[key] for key in subset})

    def unite(self, other):
        return self.combine((self, other))

    @staticmethod
    def combine(fields):
        res = {}
        for f in fields:
            for key in f.keys():
                if key in res:
                    res[key] = res[key]+f[key]
                else:
                    res[key] = f[key]
        return MultiField.from_dict(res)


for op in ["__add__", "__radd__",
           "__sub__", "__rsub__",
           "__mul__", "__rmul__",
           "__div__", "__rdiv__",
           "__truediv__", "__rtruediv__",
           "__floordiv__", "__rfloordiv__",
           "__pow__", "__rpow__",
           "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"]:
    def func(op):
        def func2(self, other):
            if isinstance(other, MultiField):
                if self._domain is not other._domain:
                    raise ValueError("domain mismatch")
                val = tuple(getattr(v1, op)(v2)
                            for v1, v2 in zip(self._val, other._val))
            else:
                val = tuple(getattr(v1, op)(other) for v1 in self._val)
            return MultiField(self._domain, val)
        return func2
    setattr(MultiField, op, func(op))


for op in ["__iadd__", "__isub__", "__imul__", "__idiv__",
           "__itruediv__", "__ifloordiv__", "__ipow__"]:
    def func(op):
        def func2(self, other):
            raise TypeError(
                "In-place operations are deliberately not supported")
        return func2
    setattr(MultiField, op, func(op))


for f in ["sqrt", "exp", "log", "tanh"]:
    def func(f):
        def func2(self):
            fu = getattr(dobj, f)
            return MultiField(self.domain,
                              tuple(func2(val) for val in self.values()))
        return func2
    setattr(MultiField, f, func(f))
