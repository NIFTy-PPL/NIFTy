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
from ..compat import *
import numpy as np
from .random import Random
import sys

ntask = 1
rank = 0
master = True


def is_numpy():
    return False


def local_shape(shape, distaxis=0):
    return shape


class data_object(object):
    def __init__(self, shape, data):
        self._data = data

    def copy(self):
        return data_object(self._data.shape, self._data.copy())

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def real(self):
        return data_object(self._data.shape, self._data.real)

    @property
    def imag(self):
        return data_object(self._data.shape, self._data.imag)

    def conj(self):
        return data_object(self._data.shape, self._data.conj())

    def conjugate(self):
        return data_object(self._data.shape, self._data.conjugate())

    def _contraction_helper(self, op, axis):
        if axis is not None:
            if len(axis) == len(self._data.shape):
                axis = None
        if axis is None:
            return getattr(self._data, op)()

        res = getattr(self._data, op)(axis=axis)
        return data_object(res.shape, res)

    def sum(self, axis=None):
        return self._contraction_helper("sum", axis)

    def prod(self, axis=None):
        return self._contraction_helper("prod", axis)

    def min(self, axis=None):
        return self._contraction_helper("min", axis)

    def max(self, axis=None):
        return self._contraction_helper("max", axis)

    def mean(self, axis=None):
        if axis is None:
            sz = self.size
        else:
            sz = reduce(lambda x, y: x*y, [self.shape[i] for i in axis])
        return self.sum(axis)/sz

    def std(self, axis=None):
        return np.sqrt(self.var(axis))

    # FIXME: to be improved!
    def var(self, axis=None):
        if axis is not None and len(axis) != len(self.shape):
            raise ValueError("functionality not yet supported")
        return (abs(self-self.mean())**2).mean()

    def _binary_helper(self, other, op):
        a = self
        if isinstance(other, data_object):
            b = other
            if a._data.shape != b._data.shape:
                raise ValueError("shapes are incompatible.")
            a = a._data
            b = b._data
        elif np.isscalar(other):
            a = a._data
            b = other
        elif isinstance(other, np.ndarray):
            a = a._data
            b = other
        else:
            return NotImplemented

        tval = getattr(a, op)(b)
        if tval is a:
            return self
        else:
            return data_object(self._data.shape, tval)

    def __neg__(self):
        return data_object(self._data.shape, -self._data)

    def __abs__(self):
        return data_object(self._data.shape, abs(self._data))

    def all(self):
        return self.sum() == self.size

    def any(self):
        return self.sum() != 0

    def fill(self, value):
        self._data.fill(value)


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
            return self._binary_helper(other, op=op)
        return func2
    setattr(data_object, op, func(op))


def full(shape, fill_value, dtype=None):
    return data_object(shape, np.full(shape, fill_value, dtype))


def empty(shape, dtype=None):
    return data_object(shape, np.empty(shape, dtype))


def zeros(shape, dtype=None, distaxis=0):
    return data_object(shape, np.zeros(shape, dtype))


def ones(shape, dtype=None, distaxis=0):
    return data_object(shape, np.ones(shape, dtype))


def empty_like(a, dtype=None):
    return data_object(np.empty_like(a._data, dtype))


def vdot(a, b):
    return np.vdot(a._data, b._data)


def _math_helper(x, function, out):
    function = getattr(np, function)
    if out is not None:
        function(x._data, out=out._data)
        return out
    else:
        return data_object(x.shape, function(x._data))


_current_module = sys.modules[__name__]

for f in ["sqrt", "exp", "log", "tanh", "conjugate"]:
    def func(f):
        def func2(x, out=None):
            return _math_helper(x, f, out)
        return func2
    setattr(_current_module, f, func(f))


def from_object(object, dtype, copy, set_locked):
    if dtype is None:
        dtype = object.dtype
    dtypes_equal = dtype == object.dtype
    if set_locked and dtypes_equal and locked(object):
        return object
    if not dtypes_equal and not copy:
        raise ValueError("cannot change data type without copying")
    if set_locked and not copy:
        raise ValueError("cannot lock object without copying")
    data = np.array(object._data, dtype=dtype, copy=copy)
    if set_locked:
        data.flags.writeable = False
    return data_object(object._shape, data, distaxis=object._distaxis)


# This function draws all random numbers on all tasks, to produce the same
# array independent on the number of tasks
# MR FIXME: depending on what is really wanted/needed (i.e. same result
# independent of number of tasks, performance etc.) we need to adjust the
# algorithm.
def from_random(random_type, shape, dtype=np.float64, **kwargs):
    generator_function = getattr(Random, random_type)
    ldat = generator_function(dtype=dtype, shape=shape, **kwargs)
    return from_local_data(shape, ldat)


def local_data(arr):
    return arr._data


def ibegin_from_shape(glob_shape, distaxis=0):
    return (0,) * len(glob_shape)


def ibegin(arr):
    return (0,) * arr._data.ndim


def np_allreduce_sum(arr):
    return arr.copy()


def np_allreduce_min(arr):
    return arr.copy()


def distaxis(arr):
    return -1


def from_local_data(shape, arr, distaxis=-1):
    return data_object(shape, arr)


def from_global_data(arr, sum_up=False):
    return data_object(arr.shape, arr)


def to_global_data(arr):
    return arr._data


def redistribute(arr, dist=None, nodist=None):
    return arr.copy()


def default_distaxis():
    return -1


def lock(arr):
    arr._data.flags.writeable = False


def locked(arr):
    return not arr._data.flags.writeable
