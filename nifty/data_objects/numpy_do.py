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


def np_allreduce_sum(arr):
    return arr


def distaxis(arr):
    return -1


def from_local_data (shape, arr, distaxis):
    if distaxis!=-1:
        raise NotImplementedError
    if shape!=arr.shape:
        raise ValueError
    return arr


def from_global_data (arr, distaxis):
    if distaxis!=-1:
        raise NotImplementedError
    return arr


def redistribute (arr, dist=None, nodist=None):
    if dist is not None and dist!=-1:
        raise NotImplementedError
    return arr


def default_distaxis():
    return -1


def local_shape(glob_shape, distaxis):
    if distaxis!=-1:
        raise NotImplementedError
    return glob_shape
