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

# Data object module for NIFTy that uses simple numpy ndarrays.

import numpy as np
from numpy import ndarray as data_object
from numpy import full, empty, empty_like, sqrt, ones, zeros, vdot, \
                  exp, log, tanh
from .random import Random

ntask = 1
rank = 0
master = True


def is_numpy():
    return True


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
    res = np.array(object, dtype=dtype, copy=copy)
    if set_locked:
        lock(res)
    return res


def from_random(random_type, shape, dtype=np.float64, **kwargs):
    generator_function = getattr(Random, random_type)
    return generator_function(dtype=dtype, shape=shape, **kwargs)


def local_data(arr):
    return arr


def ibegin_from_shape(glob_shape, distaxis=-1):
    return (0,)*len(glob_shape)


def ibegin(arr):
    return (0,)*arr.ndim


def np_allreduce_sum(arr):
    return arr


def distaxis(arr):
    return -1


def from_local_data(shape, arr, distaxis=-1):
    if shape != arr.shape:
        raise ValueError
    return arr


def from_global_data(arr, sum_up=False, distaxis=-1):
    return arr


def to_global_data(arr):
    return arr


def redistribute(arr, dist=None, nodist=None):
    return arr


def default_distaxis():
    return -1


def local_shape(glob_shape, distaxis=-1):
    return glob_shape


def lock(arr):
    arr.flags.writeable = False


def locked(arr):
    return not arr.flags.writeable
