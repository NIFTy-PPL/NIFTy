# -*- coding: utf-8 -*-

import numpy as np

#def MIN():
#    return np.min
#
#def MAX():
#    return np.max
#
#def SUM():
#    return np.sum

MIN = np.min
MAX = np.max
SUM = np.sum


class Comm(object):
    pass


class Intracomm(Comm):
    def __init__(self, name):
        if not running_single_threadedQ():
            raise RuntimeError("ERROR: MPI_dummy module is running in a " +
                               "mpirun with n>1.")
        self.name = name
        self.rank = 0
        self.size = 1

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def _scattergather_helper(self, sendbuf, recvbuf=None, **kwargs):
        sendbuf = self._unwrapper(sendbuf)
        recvbuf = self._unwrapper(recvbuf)
        if recvbuf is not None:
            recvbuf[:] = sendbuf
            return recvbuf
        else:
            recvbuf = np.copy(sendbuf)
            return recvbuf

    def bcast(self, sendbuf, *args, **kwargs):
        return sendbuf

    def Bcast(self, sendbuf, *args, **kwargs):
        return sendbuf

    def scatter(self, sendbuf, *args, **kwargs):
        return sendbuf[0]

    def Scatter(self, *args, **kwargs):
        return self._scattergather_helper(*args, **kwargs)

    def Scatterv(self, *args, **kwargs):
        return self._scattergather_helper(*args, **kwargs)

    def gather(self, sendbuf, *args, **kwargs):
        return [sendbuf]

    def Gather(self, *args, **kwargs):
        return self._scattergather_helper(*args, **kwargs)

    def Gatherv(self, *args, **kwargs):
        return self._scattergather_helper(*args, **kwargs)

    def allgather(self, sendbuf, *args, **kwargs):
        return [sendbuf]

    def Allgather(self, *args, **kwargs):
        return self._scattergather_helper(*args, **kwargs)

    def Allgatherv(self, *args, **kwargs):
        return self._scattergather_helper(*args, **kwargs)

    def Allreduce(self, sendbuf, recvbuf, op, **kwargs):
        recvbuf[:] = op(sendbuf)
        return recvbuf

    def allreduce(self, sendbuf, recvbuf, op, **kwargs):
        recvbuf[:] = op(sendbuf)
        return recvbuf

    def sendrecv(self, sendobj, **kwargs):
        return sendobj

    def _unwrapper(self, x):
        if isinstance(x, list):
            return x[0]
        else:
            return x

    def Barrier(self):
        pass


class _datatype():
    def __init__(self, name):
        self.name = str(name)


def running_single_threadedQ():
    try:
        from mpi4py import MPI
    except ImportError:
        return True
    else:
        if MPI.COMM_WORLD.size != 1:
            return False
        else:
            return True


BYTE = _datatype('MPI_BYTE')
SHORT = _datatype('MPI_SHORT')
UNSIGNED_SHORT = _datatype("MPI_UNSIGNED_SHORT")
UNSIGNED_INT = _datatype("MPI_UNSIGNED_INT")
INT = _datatype("MPI_INT")
LONG = _datatype("MPI_LONG")
UNSIGNED_LONG = _datatype("MPI_UNSIGNED_LONG")
LONG_LONG = _datatype("MPI_LONG_LONG")
UNSIGNED_LONG_LONG = _datatype("MPI_UNSIGNED_LONG_LONG")
FLOAT = _datatype("MPI_FLOAT")
DOUBLE = _datatype("MPI_DOUBLE")
LONG_DOUBLE = _datatype("MPI_LONG_DOUBLE")
COMPLEX = _datatype("MPI_COMPLEX")
DOUBLE_COMPLEX = _datatype("MPI_DOUBLE_COMPLEX")


class _comm_wrapper(object):
    def __init__(self, name):
        self.cache = None
        self.name = name

    @property
    def comm(self):
        if self.cache is None:
            self.cache = Intracomm(self.name)
        return self.cache

    def __getattr__(self, x):
        return self.comm.__getattribute__(x)


COMM_WORLD = _comm_wrapper('MPI_dummy_COMM_WORLD')







