# Data object module for NIFTy that uses simple numpy ndarrays.

import numpy as np
from numpy import ndarray as data_object
from numpy import full, empty, sqrt, ones, zeros, vdot, abs, bincount, exp, log
from .random import Random


def from_object(object, dtype=None, copy=True):
    return np.array(object, dtype=dtype, copy=copy)


def from_random(random_type, shape, dtype=np.float64, **kwargs):
    generator_function = getattr(Random, random_type)
    return generator_function(dtype=dtype, shape=shape, **kwargs)


def to_ndarray(arr):
    return arr


def from_ndarray(arr):
    return np.asarray(arr)


def local_data(arr):
    return arr


def ibegin(arr):
    return (0,)*arr.ndim


def create_from_template(tmpl, local_data, dtype):
    res = np.ndarray(tmpl.shape, dtype=dtype)
    res[()] = local_data
    return res


def np_allreduce_sum(arr):
    return arr


def dist_axis(arr):
    return -1
