import numpy as np
from .random import Random
from mpi4py import MPI

comm = MPI.COMM_WORLD
ntask = comm.Get_size()
rank = comm.Get_rank()


def shareSize(nwork, nshares, myshare):
    nbase = nwork//nshares
    return nbase if myshare>=nwork%nshares else nbase+1

def get_locshape(shape, distaxis):
    if distaxis==-1:
        return shape
    shape2=list(shape)
    shape2[distaxis]=shareSize(shape[distaxis],ntask,rank)
    return tuple(shape2)

class data_object(object):
    def __init__(self, shape, data, distaxis):
        """Must not be called directly by users"""
        self._shape = shape
        self._distaxis = distaxis
        lshape = get_locshape(self._shape, self._distaxis)
        self._data = data

    def sanity_checks(self):
        # check whether the distaxis is consistent
        if self._distaxis<-1 or self._distaxis>=len(self._shape):
            raise ValueError
        itmp=np.array(self._distaxis)
        otmp=np.empty(ntask,dtype=np.int)
        comm.Allgather(itmp,otmp)
        if np.any(otmp!=self._distaxis):
            raise ValueError
        # check whether the global shape is consistent
        itmp=np.array(self._shape)
        otmp=np.empty((len(self._shape),ntask),dtype=np.int)
        comm.Allgather(itmp,otmp)
        for i in range(ntask):
            if (otmp[i,:]!=self._shape).any():
                raise ValueError
        # check shape of local data
        if self._distaxis<0:
            if self._data.shape!=self._shape:
                raise ValueError
        else:
            itmp=np.array(self._shape)
            itmp[self._distaxis] = get_local_length(self._shape[self._distaxis],ntask,rank)
            if self._data.shape!=itmp:
                raise ValueError

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
        return data_object(self._shape, self._data.real, self._dist_axis)

    @property
    def imag(self):
        return data_object(self._shape, self._data.imag, self._dist_axis)

    def _contraction_helper(self, op, axis):
        if axis is not None:
            if len(axis)==len(self._data.shape):
                axis = None
        if axis is None:
            res = getattr(self._data, op)()
            MPI.COMM_WORLD.Allreduce(res,res2,mpiop)

        if self._distaxis in axis:
            pass# reduce globally, redistribute the result along axis 0(?)
        else:
            pass# reduce locally

        # perform the contraction on the data
        data = getattr(self._data, op)(axis=axis)

        # check if the result is scalar or if a result_field must be constr.
        if np.isscalar(data):
            return data
        else:
            return data_object(data)

    def sum(self, axis=None):
        return self._contraction_helper("sum", MPI.SUM, axis)

    def _binary_helper(self, other, op):
        a = self._data
        if isinstance(other, data_object):
            b = other._data
            if a._shape != b._shape:
                raise ValueError("shapes are incompatible.")
            if a._distaxis != b._distaxis:
                raise ValueError("distributions are incompatible.")
        else:
            b = other

        tval = getattr(a, op)(b)
        return self if tval is a else data_object(self._shape, tval, self._distaxis)

    def __add__(self, other):
        return self._binary_helper(other, op='__add__')

    def __radd__(self, other):
        return self._binary_helper(other, op='__radd__')

    def __iadd__(self, other):
        return self._binary_helper(other, op='__iadd__')

    def __sub__(self, other):
        return self._binary_helper(other, op='__sub__')

    def __rsub__(self, other):
        return self._binary_helper(other, op='__rsub__')

    def __isub__(self, other):
        return self._binary_helper(other, op='__isub__')

    def __mul__(self, other):
        return self._binary_helper(other, op='__mul__')

    def __rmul__(self, other):
        return self._binary_helper(other, op='__rmul__')

    def __imul__(self, other):
        return self._binary_helper(other, op='__imul__')

    def __div__(self, other):
        return self._binary_helper(other, op='__div__')

    def __rdiv__(self, other):
        return self._binary_helper(other, op='__rdiv__')

    def __truediv__(self, other):
        return self._binary_helper(other, op='__truediv__')

    def __rtruediv__(self, other):
        return self._binary_helper(other, op='__rtruediv__')

    def __pow__(self, other):
        return self._binary_helper(other, op='__pow__')

    def __rpow__(self, other):
        return self._binary_helper(other, op='__rpow__')

    def __ipow__(self, other):
        return self._binary_helper(other, op='__ipow__')

    def __eq__(self, other):
        return self._binary_helper(other, op='__eq__')

    def __ne__(self, other):
        return self._binary_helper(other, op='__ne__')

    def __neg__(self):
        return data_object(-self._data)

    def __abs__(self):
        return data_object(np.abs(self._data))

    def ravel(self):
        return data_object(self._data.ravel())

    def reshape(self, shape):
        return data_object(self._data.reshape(shape))

    def all(self):
        return self._data.all()

    def any(self):
        return self._data.any()


def full(shape, fill_value, dtype=None, dist_axis=0):
    return data_object(shape, np.full(shape, local_shape(shape, dist_axis), fill_value, dtype))


def empty(shape, dtype=np.float):
    return data_object(np.empty(shape, dtype))


def zeros(shape, dtype=np.float):
    return data_object(np.zeros(shape, dtype))


def ones(shape, dtype=np.float):
    return data_object(np.ones(shape, dtype))


def empty_like(a, dtype=None):
    return data_object(np.empty_like(a._data, dtype))


def vdot(a, b):
    return np.vdot(a._data, b._data)


def _math_helper(x, function, out):
    if out is not None:
        function(x._data, out=out._data)
        return out
    else:
        return data_object(function(x._data))


def abs(a, out=None):
    return _math_helper(a, np.abs, out)


def exp(a, out=None):
    return _math_helper(a, np.exp, out)


def log(a, out=None):
    return _math_helper(a, np.log, out)


def sqrt(a, out=None):
    return _math_helper(a, np.sqrt, out)


def bincount(x, weights=None, minlength=None):
    if weights is not None:
        weights = weights._data
    res = np.bincount(x._data, weights, minlength)
    return data_object(res)


def from_object(object, dtype=None, copy=True):
    return data_object(np.array(object._data, dtype=dtype, copy=copy))


def from_random(random_type, shape, dtype=np.float64, **kwargs):
    generator_function = getattr(Random, random_type)
    return data_object(generator_function(dtype=dtype, shape=shape, **kwargs))


def to_ndarray(arr):
    return arr._data


def from_ndarray(arr):
    return data_object(arr.shape,arr,-1)
