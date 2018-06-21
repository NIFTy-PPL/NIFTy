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

from ..field import Field
import numpy as np
from .multi_domain import MultiDomain


class MultiField(object):
    def __init__(self, val):
        """
        Parameters
        ----------
        val : dict
        """
        self._val = val
        self._domain = MultiDomain.make(
            {key: val.domain for key, val in self._val.items()})

    def __getitem__(self, key):
        return self._val[key]

    def keys(self):
        return self._val.keys()

    def items(self):
        return self._val.items()

    def values(self):
        return self._val.values()

    @property
    def domain(self):
        return self._domain

    @property
    def dtype(self):
        return {key: val.dtype for key, val in self._val.items()}

    @property
    def real(self):
        """MultiField : The real part of the multi field"""
        return MultiField({key: field.real for key, field in self.items()})

    @property
    def imag(self):
        """MultiField : The imaginary part of the multi field"""
        return MultiField({key: field.imag for key, field in self.items()})

    @staticmethod
    def from_random(random_type, domain, dtype=np.float64, **kwargs):
        dtype = MultiField.build_dtype(dtype, domain)
        return MultiField({key: Field.from_random(random_type, domain[key],
                                                  dtype[key], **kwargs)
                           for key in domain.keys()})

    def fill(self, fill_value):
        """Fill `self` uniformly with `fill_value`

        Parameters
        ----------
        fill_value: float or complex or int
            The value to fill the field with.
        """
        for val in self._val.values():
            val.fill(fill_value)
        return self

    def _check_domain(self, other):
        if other._domain != self._domain:
            raise ValueError("domains are incompatible.")

    def vdot(self, x):
        result = 0.
        self._check_domain(x)
        for key, sub_field in self.items():
            result += sub_field.vdot(x[key])
        return result

    def lock(self):
        for v in self.values():
            v.lock()
        return self

    @property
    def locked(self):
        return all(v.locked for v in self.values())

    def copy(self):
        return MultiField({key: val.copy() for key, val in self.items()})

    def locked_copy(self):
        if self.locked:
            return self
        return MultiField({key: val.locked_copy()
                          for key, val in self.items()})

    def empty_copy(self):
        return MultiField({key: val.empty_copy() for key, val in self.items()})

    @staticmethod
    def build_dtype(dtype, domain):
        if isinstance(dtype, dict):
            return dtype
        if dtype is None:
            dtype = np.float64
        return {key: dtype for key in domain.keys()}

    @staticmethod
    def empty(domain, dtype=None):
        dtype = MultiField.build_dtype(dtype, domain)
        return MultiField({key: Field.empty(dom, dtype=dtype[key])
                           for key, dom in domain.items()})

    @staticmethod
    def full(domain, val):
        return MultiField({key: Field.full(dom, val)
                           for key, dom in domain.items()})

    def to_global_data(self):
        return {key: val.to_global_data() for key, val in self._val.items()}

    @staticmethod
    def from_global_data(domain, arr, sum_up=False):
        return MultiField({key: Field.from_global_data(domain[key],
                                                       val, sum_up)
                           for key, val in arr.items()})

    def norm(self):
        """ Computes the L2-norm of the field values.

        Returns
        -------
        norm : float
            The L2-norm of the field values.
        """
        return np.sqrt(np.abs(self.vdot(x=self)))

    def __neg__(self):
        return MultiField({key: -val for key, val in self.items()})

    def conjugate(self):
        return MultiField({key: sub_field.conjugate()
                           for key, sub_field in self.items()})

    def isEquivalentTo(self, other):
        """Determines (as quickly as possible) whether `self`'s content is
        identical to `other`'s content."""
        if self is other:
            return True
        if not isinstance(other, MultiField):
            return False
        if self._domain != other._domain:
            return False
        for key, val in self._val.items():
            if not val.isEquivalentTo(other[key]):
                return False
        return True

    def isSubsetOf(self, other):
        """Determines (as quickly as possible) whether `self`'s content is
        a subset of `other`'s content."""
        if self is other:
            return True
        if not isinstance(other, MultiField):
            return False
        for key, val in self._domain.items():
            if key not in other._domain or other._domain[key] != val:
                return False
        for key, val in self._val.items():
            if not val.isSubsetOf(other[key]):
                return False
        return True


for op in ["__add__", "__radd__", "__iadd__",
           "__sub__", "__rsub__", "__isub__",
           "__mul__", "__rmul__", "__imul__",
           "__div__", "__rdiv__", "__idiv__",
           "__truediv__", "__rtruediv__", "__itruediv__",
           "__floordiv__", "__rfloordiv__", "__ifloordiv__",
           "__pow__", "__rpow__", "__ipow__",
           "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"]:
    def func(op):
        def func2(self, other):
            if isinstance(other, MultiField):
                self._check_domain(other)
                result_val = {key: getattr(sub_field, op)(other[key])
                              for key, sub_field in self.items()}
            else:
                result_val = {key: getattr(val, op)(other)
                              for key, val in self.items()}
            return MultiField(result_val)
        return func2
    setattr(MultiField, op, func(op))