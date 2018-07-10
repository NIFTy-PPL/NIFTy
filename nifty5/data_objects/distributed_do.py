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

import sys

import numpy as np
from mpi4py import MPI

from ..compat import *
from .random import Random

_comm = MPI.COMM_WORLD
ntask = _comm.Get_size()
rank = _comm.Get_rank()
master = (rank == 0)


def is_numpy():
    return False


def _shareSize(nwork, nshares, myshare):
    return (nwork//nshares) + int(myshare < nwork % nshares)


def _shareRange(nwork, nshares, myshare):
    nbase = nwork//nshares
    additional = nwork % nshares
    lo = myshare*nbase + min(myshare, additional)
    hi = lo + nbase + int(myshare < additional)
    return lo, hi


def local_shape(shape, distaxis=0):
    if len(shape) == 0 or distaxis == -1:
        return shape
    shape2 = list(shape)
    shape2[distaxis] = _shareSize(shape[distaxis], ntask, rank)
    return tuple(shape2)


class data_object(object):
    def __init__(self, shape, data, distaxis):
        self._shape = tuple(shape)
        if len(self._shape) == 0:
            distaxis = -1
        self._distaxis = distaxis
        self._data = data
        if local_shape(self._shape, self._distaxis) != self._data.shape:
            raise ValueError("shape mismatch")

    def copy(self):
        return data_object(self._shape, self._data.copy(), self._distaxis)

#     def _sanity_checks(self):
#         # check whether the distaxis is consistent
#         if self._distaxis < -1 or self._distaxis >= len(self._shape):
#             raise ValueError
#         itmp = np.array(self._distaxis)
#         otmp = np.empty(ntask, dtype=np.int)
#         _comm.Allgather(itmp, otmp)
#         if np.any(otmp != self._distaxis):
#             raise ValueError
#         # check whether the global shape is consistent
#         itmp = np.array(self._shape)
#         otmp = np.empty((ntask, len(self._shape)), dtype=np.int)
#         _comm.Allgather(itmp, otmp)
#         for i in range(ntask):
#             if np.any(otmp[i, :] != self._shape):
#                 raise ValueError
#         # check shape of local data
#         if self._distaxis < 0:
#             if self._data.shape != self._shape:
#                 raise ValueError
#         else:
#             itmp = np.array(self._shape)
#             itmp[self._distaxis] = _shareSize(self._shape[self._distaxis],
#                                               ntask, rank)
#             if np.any(self._data.shape != itmp):
#                 raise ValueError

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return np.prod(self._shape)

    @property
    def real(self):
        return data_object(self._shape, self._data.real, self._distaxis)

    @property
    def imag(self):
        return data_object(self._shape, self._data.imag, self._distaxis)

    def conj(self):
        return data_object(self._shape, self._data.conj(), self._distaxis)

    def conjugate(self):
        return data_object(self._shape, self._data.conjugate(), self._distaxis)

    def _contraction_helper(self, op, mpiop, axis):
        if axis is not None:
            if len(axis) == len(self._data.shape):
                axis = None
        if axis is None:
            res = np.array(getattr(self._data, op)())
            if (self._distaxis == -1):
                return res[()]
            res2 = np.empty((), dtype=res.dtype)
            _comm.Allreduce(res, res2, mpiop)
            return res2[()]

        if self._distaxis in axis:
            res = getattr(self._data, op)(axis=axis)
            res2 = np.empty_like(res)
            _comm.Allreduce(res, res2, mpiop)
            return from_global_data(res2, distaxis=0)
        else:
            # perform the contraction on the local data
            res = getattr(self._data, op)(axis=axis)
            if self._distaxis == -1:
                return from_global_data(res, distaxis=0)
            shp = list(res.shape)
            shift = 0
            for ax in axis:
                if ax < self._distaxis:
                    shift += 1
            shp[self._distaxis-shift] = self.shape[self._distaxis]
            return from_local_data(shp, res, self._distaxis-shift)

    def sum(self, axis=None):
        return self._contraction_helper("sum", MPI.SUM, axis)

    def prod(self, axis=None):
        return self._contraction_helper("prod", MPI.PROD, axis)

    def min(self, axis=None):
        return self._contraction_helper("min", MPI.MIN, axis)

    def max(self, axis=None):
        return self._contraction_helper("max", MPI.MAX, axis)

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
            if a._shape != b._shape:
                raise ValueError("shapes are incompatible.")
            if a._distaxis != b._distaxis:
                raise ValueError("distributions are incompatible.")
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
            return data_object(self._shape, tval, self._distaxis)

    def __neg__(self):
        return data_object(self._shape, -self._data, self._distaxis)

    def __abs__(self):
        return data_object(self._shape, abs(self._data), self._distaxis)

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


def full(shape, fill_value, dtype=None, distaxis=0):
    return data_object(shape, np.full(local_shape(shape, distaxis),
                                      fill_value, dtype), distaxis)


def empty(shape, dtype=None, distaxis=0):
    return data_object(shape, np.empty(local_shape(shape, distaxis),
                                       dtype), distaxis)


def zeros(shape, dtype=None, distaxis=0):
    return data_object(shape, np.zeros(local_shape(shape, distaxis), dtype),
                       distaxis)


def ones(shape, dtype=None, distaxis=0):
    return data_object(shape, np.ones(local_shape(shape, distaxis), dtype),
                       distaxis)


def empty_like(a, dtype=None):
    return data_object(np.empty_like(a._data, dtype))


def vdot(a, b):
    tmp = np.array(np.vdot(a._data, b._data))
    res = np.empty((), dtype=tmp.dtype)
    _comm.Allreduce(tmp, res, MPI.SUM)
    return res[()]


def _math_helper(x, function, out):
    function = getattr(np, function)
    if out is not None:
        function(x._data, out=out._data)
        return out
    else:
        return data_object(x.shape, function(x._data), x._distaxis)


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
    for i in range(ntask):
        lshape = list(shape)
        lshape[0] = _shareSize(shape[0], ntask, i)
        ldat = generator_function(dtype=dtype, shape=lshape, **kwargs)
        if i == rank:
            outdat = ldat
    return from_local_data(shape, outdat, distaxis=0)


def local_data(arr):
    return arr._data


def ibegin_from_shape(glob_shape, distaxis=0):
    res = [0] * len(glob_shape)
    if distaxis < 0:
        return res
    res[distaxis] = _shareRange(glob_shape[distaxis], ntask, rank)[0]
    return tuple(res)


def ibegin(arr):
    res = [0] * arr._data.ndim
    res[arr._distaxis] = _shareRange(arr._shape[arr._distaxis], ntask, rank)[0]
    return tuple(res)


def np_allreduce_sum(arr):
    res = np.empty_like(arr)
    _comm.Allreduce(arr, res, MPI.SUM)
    return res


def np_allreduce_min(arr):
    res = np.empty_like(arr)
    _comm.Allreduce(arr, res, MPI.MIN)
    return res


def distaxis(arr):
    return arr._distaxis


def from_local_data(shape, arr, distaxis=0):
    return data_object(shape, arr, distaxis)


def from_global_data(arr, sum_up=False, distaxis=0):
    if sum_up:
        arr = np_allreduce_sum(arr)
    if distaxis == -1:
        return data_object(arr.shape, arr, distaxis)
    lo, hi = _shareRange(arr.shape[distaxis], ntask, rank)
    sl = [slice(None)]*len(arr.shape)
    sl[distaxis] = slice(lo, hi)
    return data_object(arr.shape, arr[sl], distaxis)


def to_global_data(arr):
    if arr._distaxis == -1:
        return arr._data
    tmp = redistribute(arr, dist=-1)
    return tmp._data


def redistribute(arr, dist=None, nodist=None):
    if dist is not None:
        if nodist is not None:
            raise ValueError
        if dist == arr._distaxis:
            return arr
    else:
        if nodist is None:
            raise ValueError
        if arr._distaxis not in nodist:
            return arr
        dist = -1
        for i in range(len(arr.shape)):
            if i not in nodist:
                dist = i
                break

    if arr._distaxis == -1:  # all data available, just pick the proper subset
        return from_global_data(arr._data, distaxis=dist)
    if dist == -1:  # gather all data on all tasks
        tmp = np.moveaxis(arr._data, arr._distaxis, 0)
        slabsize = np.prod(tmp.shape[1:])*tmp.itemsize
        sz = np.empty(ntask, dtype=np.int)
        for i in range(ntask):
            sz[i] = slabsize*_shareSize(arr.shape[arr._distaxis], ntask, i)
        disp = np.empty(ntask, dtype=np.int)
        disp[0] = 0
        disp[1:] = np.cumsum(sz[:-1])
        tmp = np.require(tmp, requirements="C")
        out = np.empty(arr.size, dtype=arr.dtype)
        _comm.Allgatherv(tmp, [out, sz, disp, MPI.BYTE])
        shp = np.array(arr._shape)
        shp[1:arr._distaxis+1] = shp[0:arr._distaxis]
        shp[0] = arr.shape[arr._distaxis]
        out = out.reshape(shp)
        out = np.moveaxis(out, 0, arr._distaxis)
        return from_global_data(out, distaxis=-1)

    # real redistribution via Alltoallv
    ssz0 = arr._data.size//arr.shape[dist]
    ssz = np.empty(ntask, dtype=np.int)
    rszall = arr.size//arr.shape[dist]*_shareSize(arr.shape[dist], ntask, rank)
    rbuf = np.empty(rszall, dtype=arr.dtype)
    rsz0 = rszall//arr.shape[arr._distaxis]
    rsz = np.empty(ntask, dtype=np.int)
    if dist == 0:  # shortcut possible
        sbuf = np.ascontiguousarray(arr._data)
        for i in range(ntask):
            lo, hi = _shareRange(arr.shape[dist], ntask, i)
            ssz[i] = ssz0*(hi-lo)
            rsz[i] = rsz0*_shareSize(arr.shape[arr._distaxis], ntask, i)
    else:
        sbuf = np.empty(arr._data.size, dtype=arr.dtype)
        sslice = [slice(None)]*arr._data.ndim
        ofs = 0
        for i in range(ntask):
            lo, hi = _shareRange(arr.shape[dist], ntask, i)
            sslice[dist] = slice(lo, hi)
            ssz[i] = ssz0*(hi-lo)
            sbuf[ofs:ofs+ssz[i]] = arr._data[sslice].flat
            ofs += ssz[i]
            rsz[i] = rsz0*_shareSize(arr.shape[arr._distaxis], ntask, i)
    ssz *= arr._data.itemsize
    rsz *= arr._data.itemsize
    sdisp = np.append(0, np.cumsum(ssz[:-1]))
    rdisp = np.append(0, np.cumsum(rsz[:-1]))
    s_msg = [sbuf, (ssz, sdisp), MPI.BYTE]
    r_msg = [rbuf, (rsz, rdisp), MPI.BYTE]
    _comm.Alltoallv(s_msg, r_msg)
    del sbuf  # free memory
    if arr._distaxis == 0:
        rbuf = rbuf.reshape(local_shape(arr.shape, dist))
        arrnew = from_local_data(arr.shape, rbuf, distaxis=dist)
    else:
        arrnew = empty(arr.shape, dtype=arr.dtype, distaxis=dist)
        rslice = [slice(None)]*arr._data.ndim
        ofs = 0
        for i in range(ntask):
            lo, hi = _shareRange(arr.shape[arr._distaxis], ntask, i)
            rslice[arr._distaxis] = slice(lo, hi)
            sz = rsz[i]//arr._data.itemsize
            arrnew._data[rslice].flat = rbuf[ofs:ofs+sz]
            ofs += sz
    return arrnew


def transpose(arr):
    if len(arr.shape) != 2 or arr._distaxis != 0:
        raise ValueError("bad input")
    ssz0 = arr._data.size//arr.shape[1]
    ssz = np.empty(ntask, dtype=np.int)
    rszall = arr.size//arr.shape[1]*_shareSize(arr.shape[1], ntask, rank)
    rbuf = np.empty(rszall, dtype=arr.dtype)
    rsz0 = rszall//arr.shape[0]
    rsz = np.empty(ntask, dtype=np.int)
    sbuf = np.empty(arr._data.size, dtype=arr.dtype)
    ofs = 0
    for i in range(ntask):
        lo, hi = _shareRange(arr.shape[1], ntask, i)
        ssz[i] = ssz0*(hi-lo)
        sbuf[ofs:ofs+ssz[i]] = arr._data[:, lo:hi].flat
        ofs += ssz[i]
        rsz[i] = rsz0*_shareSize(arr.shape[0], ntask, i)
    ssz *= arr._data.itemsize
    rsz *= arr._data.itemsize
    sdisp = np.append(0, np.cumsum(ssz[:-1]))
    rdisp = np.append(0, np.cumsum(rsz[:-1]))
    s_msg = [sbuf, (ssz, sdisp), MPI.BYTE]
    r_msg = [rbuf, (rsz, rdisp), MPI.BYTE]
    _comm.Alltoallv(s_msg, r_msg)
    del sbuf  # free memory
    arrnew = empty((arr.shape[1], arr.shape[0]), dtype=arr.dtype, distaxis=0)
    ofs = 0
    sz2 = _shareSize(arr.shape[1], ntask, rank)
    for i in range(ntask):
        lo, hi = _shareRange(arr.shape[0], ntask, i)
        sz = rsz[i]//arr._data.itemsize
        arrnew._data[:, lo:hi] = rbuf[ofs:ofs+sz].reshape(hi-lo, sz2).T
        ofs += sz
    return arrnew


def default_distaxis():
    return 0


def lock(arr):
    arr._data.flags.writeable = False


def locked(arr):
    return not arr._data.flags.writeable
