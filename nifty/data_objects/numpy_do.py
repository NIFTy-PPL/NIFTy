# Data object module for NIFTy that uses simple numpy ndarrays.

import numpy as np
from numpy import ndarray as data_object
from numpy import full, empty, empty_like, sqrt, ones, zeros, vdot, abs, \
                  exp, log
from .random import Random

__all__ = ["ntask", "rank", "master", "local_shape", "data_object", "full",
           "empty", "zeros", "ones", "empty_like", "vdot", "abs", "exp",
           "log", "sqrt", "from_object", "from_random",
           "local_data", "ibegin", "np_allreduce_sum", "distaxis",
           "from_local_data", "from_global_data", "to_global_data",
           "redistribute", "default_distaxis"]

ntask = 1
rank = 0
master = True


def from_object(object, dtype=None, copy=True):
    return np.array(object, dtype=dtype, copy=copy)


def from_random(random_type, shape, dtype=np.float64, **kwargs):
    generator_function = getattr(Random, random_type)
    return generator_function(dtype=dtype, shape=shape, **kwargs)


def local_data(arr):
    return arr


def ibegin(arr):
    return (0,)*arr.ndim


def np_allreduce_sum(arr):
    return arr


def distaxis(arr):
    return -1


def from_local_data(shape, arr, distaxis):
    if shape != arr.shape:
        raise ValueError
    return arr


def from_global_data(arr, distaxis=-1):
    return arr


def to_global_data(arr):
    return arr


def redistribute(arr, dist=None, nodist=None):
    return arr


def default_distaxis():
    return -1


def local_shape(glob_shape, distaxis):
    return glob_shape
