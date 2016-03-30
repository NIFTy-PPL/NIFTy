# -*- coding: utf-8 -*-

import numpy as np

from nifty.keepers import global_configuration as gc,\
                          global_dependency_injector as gdi

MPI = gdi[gc['mpi_module']]


class _dtype_converter(object):
    """
        NIFTY class for dtype conversion between python/numpy dtypes and MPI
        dtypes.
    """

    def __init__(self):
        pre_dict = [
            # [, MPI_CHAR],
            # [, MPI_SIGNED_CHAR],
            # [, MPI_UNSIGNED_CHAR],
            [np.dtype('bool'), MPI.BYTE],
            [np.dtype('int16'), MPI.SHORT],
            [np.dtype('uint16'), MPI.UNSIGNED_SHORT],
            [np.dtype('uint32'), MPI.UNSIGNED_INT],
            [np.dtype('int32'), MPI.INT],
            [np.dtype('int'), MPI.LONG],
            [np.dtype(np.long), MPI.LONG],
            [np.dtype('int64'), MPI.LONG_LONG],
            [np.dtype('longlong'), MPI.LONG],
            [np.dtype('uint'), MPI.UNSIGNED_LONG],
            [np.dtype('uint64'), MPI.UNSIGNED_LONG_LONG],
            [np.dtype('ulonglong'), MPI.UNSIGNED_LONG_LONG],
            [np.dtype('float32'), MPI.FLOAT],
            [np.dtype('float64'), MPI.DOUBLE],
            [np.dtype('float128'), MPI.LONG_DOUBLE],
            [np.dtype('complex64'), MPI.COMPLEX],
            [np.dtype('complex128'), MPI.DOUBLE_COMPLEX]]

        to_mpi_pre_dict = np.array(pre_dict)
        to_mpi_pre_dict[:, 0] = map(self.dictionize_np, to_mpi_pre_dict[:, 0])
        self._to_mpi_dict = dict(to_mpi_pre_dict)

        to_np_pre_dict = np.array(pre_dict)[:, ::-1]
        to_np_pre_dict[:, 0] = map(self.dictionize_mpi, to_np_pre_dict[:, 0])
        self._to_np_dict = dict(to_np_pre_dict)

    def dictionize_np(self, x):
        dic = x.type.__dict__.items()
        if x.type is np.float:
            dic[24] = 0
            dic[29] = 0
            dic[37] = 0
        return frozenset(dic)

    def dictionize_mpi(self, x):
        return x.name

    def to_mpi(self, dtype):
        return self._to_mpi_dict[self.dictionize_np(dtype)]

    def to_np(self, dtype):
        return self._to_np_dict[self.dictionize_mpi(dtype)]

    def known_mpi_Q(self, dtype):
        return (self.dictionize_mpi(dtype) in self._to_np_dict)

    def known_np_Q(self, dtype):
        return (self.dictionize_np(np.dtype(dtype)) in self._to_mpi_dict)

dtype_converter = _dtype_converter()
