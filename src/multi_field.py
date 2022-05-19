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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from . import utilities
from .domain_tuple import DomainTuple
from .field import Field
from .multi_domain import MultiDomain
from .operators.operator import Operator


class MultiField(Operator):
    def __init__(self, domain, val):
        """The discrete representation of a continuous field over a sum space.

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
                utilities.check_object_identity(v._domain, d)
            else:
                raise TypeError("bad entry in val (must be Field)")
        self._domain = domain
        self._val = val

    @staticmethod
    def from_dict(dct, domain=None):
        if domain is None:
            for dd in dct.values():
                if not isinstance(dd.domain, DomainTuple):
                    raise TypeError('Values of dictionary need to be Fields '
                                    'defined on DomainTuples.')
            domain = MultiDomain.make({key: v._domain
                                       for key, v in dct.items()})
        res = tuple(dct[key] if key in dct else Field(dom, 0.)
                    for key, dom in zip(domain.keys(), domain.domains()))
        return MultiField(domain, res)

    def to_dict(self):
        return {key: val for key, val in zip(self._domain.keys(), self._val)}

    def __getitem__(self, key):
        return self._val[self._domain.idx[key]]

    def __contains__(self, key):
        return key in self._domain.idx

    def keys(self):
        return self._domain.keys()

    def items(self):
        return zip(self._domain.keys(), self._val)

    def values(self):
        return self._val

    @property
    def domain(self):
        return self._domain

    @property
    def dtype(self):
        return {key: val.dtype for key, val in self.items()}

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
    def from_random(domain, random_type='normal', dtype=np.float64, **kwargs):
        """Draws a random multi-field with the given parameters.

        Parameters
        ----------
        random_type : 'pm1', 'normal', or 'uniform'
            The random distribution to use.
        domain : DomainTuple
            The domain of the output random Field.
        dtype : type
            The datatype of the output random Field.
            If the datatype is complex, each real an imaginary part have
            variance 1.

        Returns
        -------
        MultiField
            The newly created :class:`MultiField`.

        Notes
        -----
        The individual fields within this multi-field will be drawn in alphabetical
        order of the multi-field's domain keys. As a consequence, renaming these
        keys may cause the multi-field to be filled with different random numbers,
        even for the same initial RNG state.
        """
        domain = MultiDomain.make(domain)
        if isinstance(dtype, dict):
            dtype = {kk: np.dtype(dt) for kk, dt in dtype.items()}
        else:
            dtype = np.dtype(dtype)
            dtype = {kk: dtype for kk in domain.keys()}
        dct = {kk: Field.from_random(domain[kk], random_type, dtype[kk], **kwargs)
               for kk in domain.keys()}
        return MultiField.from_dict(dct)

    def s_vdot(self, x):
        result = 0.
        utilities.check_object_identity(x._domain, self._domain)
        for v1, v2 in zip(self._val, x._val):
            result += v1.s_vdot(v2)
        return result

    def vdot(self, x):
        return Field.scalar(self.s_vdot(x))

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
        return MultiField(domain, tuple(Field(dom, val)
                          for dom in domain._domains))

    @property
    def val(self):
        return {key: val.val
                for key, val in zip(self._domain.keys(), self._val)}

    def val_rw(self):
        return {key: val.val_rw()
                for key, val in zip(self._domain.keys(), self._val)}

    @staticmethod
    def from_raw(domain, arr):
        return MultiField(
            domain, tuple(Field(domain[key], arr[key])
                          for key in domain.keys()))

    def norm(self, ord=2):
        """Computes the norm of the field values.

        Parameters
        ----------
        ord : int, default=2
            accepted values: 1, 2, ..., np.inf

        Returns
        -------
        norm : float
            The norm of the field values.
        """
        nrm = np.asarray([f.norm(ord) for f in self._val])
        if ord == np.inf:
            return nrm.max()
        return (nrm ** ord).sum() ** (1./ord)
#        return np.sqrt(np.abs(self.vdot(x=self)))

    def s_sum(self):
        """Computes the sum all field values.

        Returns
        -------
        norm : float
            The sum of the field values.
        """
        return utilities.my_sum(map(lambda v: v.s_sum(), self._val))

    @property
    def size(self):
        """Computes the overall degrees of freedom.

        Returns
        -------
        size : int
            The sum of the size of the individual fields
        """
        return utilities.my_sum(map(lambda d: d.size, self._domain.domains()))

    def __neg__(self):
        return self._transform(lambda x: -x)

    def __abs__(self):
        return self._transform(lambda x: abs(x))

    def conjugate(self):
        return self._transform(lambda x: x.conjugate())

    def clip(self, a_min=None, a_max=None):
        return self.ptw("clip", a_min, a_max)

    def s_all(self):
        for v in self._val:
            if not v.s_all():
                return False
        return True

    def s_any(self):
        for v in self._val:
            if v.s_any():
                return True
        return False

    def extract(self, subset):
        if subset is self._domain:
            return self
        return MultiField(subset,
                          tuple(self[key] for key in subset.keys()))

    def extract_by_keys(self, keys):
        dom = MultiDomain.make({kk: vv for kk, vv in self.domain.items() if kk in keys})
        return self.extract(dom)

    def extract_part(self, subset):
        if subset is self._domain:
            return self
        dct = {key: self[key] for key in subset.keys() if key in self}
        if len(dct) == 0:
            return None
        return MultiField.from_dict(dct)

    def unite(self, other):
        """Merges two MultiFields on potentially different MultiDomains.

        Parameters
        ----------
        other : :class:`nifty8.multi_field.MultiField`
            the partner Field

        Returns
        -------
        MultiField
            This MultiField's domain is the union of the input fields'
            domains. The values are the sum of the fields in self and other.
            If a field is not present, it is assumed to have an uniform value
            of zero.
        """
        return self.flexible_addsub(other, False)

    @staticmethod
    def union(fields, domain=None):
        """Returns the union of its input fields.

        Parameters
        ----------
        fields : iterable of :class:`nifty8.multi_field.MultiField`
            The set of input fields. Their domains need not be identical.
        domain : MultiDomain or None
            If supplied, this will be the domain of the resulting field.
            Providing this domain will accelerate the function.

        Returns
        -------
        MultiField
            The union of the input fields

        Notes
        -----
        If the same key occurs more than once in the input fields, the value
        associated with the last occurrence will be put into the output.
        No summation is performed!
        """
        res = {}
        for field in fields:
            res.update(field.to_dict())
        return MultiField.from_dict(res, domain)

    def flexible_addsub(self, other, neg):
        """Merges two MultiFields on potentially different MultiDomains.

        Parameters
        ----------
        other : :class:`nifty8.multi_field.MultiField`
            the partner Field
        neg : bool or dict
            if True, the partner field is subtracted, otherwise added

        Returns
        -------
        MultiField
            This MultiField's domain is the union of the input fields'
            domains. The values are the sum (or difference, if neg==True) of
            the fields in self and other. If a field is not present, it is
            assumed to have an uniform value of zero.
        """
        if self._domain is other._domain and not isinstance(neg, dict):
            return self-other if neg else self+other

        if isinstance(neg, dict):
            fct = lambda k: neg[k]
        else:
            fct = lambda k: neg
        res = self.to_dict()
        for key, val in other.items():
            if key in res:
                res[key] = res[key]-val if fct(key) else res[key]+val
            else:
                res[key] = -val if fct(key) else val
        return MultiField.from_dict(res)

    def _prep_args(self, args, kwargs, i):
        for arg in args + tuple(kwargs.values()):
            if not (arg is None or np.isscalar(arg) or arg.jac is None):
                raise TypeError("bad argument")
        argstmp = tuple(arg if arg is None or np.isscalar(arg) else arg._val[i]
                        for arg in args)
        kwargstmp = {key: val if val is None or np.isscalar(val) else val._val[i]
                     for key, val in kwargs.items()}
        return argstmp, kwargstmp

    def ptw(self, op, *args, **kwargs):
        tmp = []
        for i in range(len(self._val)):
            argstmp, kwargstmp = self._prep_args(args, kwargs, i)
            tmp.append(self._val[i].ptw(op, *argstmp, **kwargstmp))
        return MultiField(self.domain, tuple(tmp))

    def ptw_with_deriv(self, op, *args, **kwargs):
        tmp = []
        for i in range(len(self._val)):
            argstmp, kwargstmp = self._prep_args(args, kwargs, i)
            tmp.append(self._val[i].ptw_with_deriv(op, *argstmp, **kwargstmp))
        return (MultiField(self.domain, tuple(v[0] for v in tmp)),
                MultiField(self.domain, tuple(v[1] for v in tmp)))

    def _binary_op(self, other, op):
        f = getattr(Field, op)
        if isinstance(other, MultiField):
            utilities.check_object_identity(self._domain, other._domain)
            val = tuple(f(v1, v2)
                        for v1, v2 in zip(self._val, other._val))
        else:
            val = tuple(f(v1, other) for v1 in self._val)
        return MultiField(self._domain, val)


for op in ["__add__", "__radd__",
           "__sub__", "__rsub__",
           "__mul__", "__rmul__",
           "__truediv__", "__rtruediv__",
           "__floordiv__", "__rfloordiv__",
           "__pow__", "__rpow__",
           "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"]:
    def func(op):
        def func2(self, other):
            return self._binary_op(other, op)
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
