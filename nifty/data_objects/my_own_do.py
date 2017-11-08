import numpy as np
from .random import Random


class data_object(object):
    def __init__(self, npdata):
        self._data = np.asarray(npdata)

    # FIXME: subscripting support will most likely go away
    def __getitem__(self, key):
        res = self._data[key]
        return res if np.isscalar(res) else data_object(res)

    def __setitem__(self, key, value):
        if isinstance(value, data_object):
            self._data[key] = value._data
        else:
            self._data[key] = value

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
        return data_object(self._data.real)

    @property
    def imag(self):
        return data_object(self._data.imag)

    def _contraction_helper(self, op, axis):
        if axis is not None:
            if len(axis)==len(self._data.shape):
                axis = None
        if axis is None:
            return getattr(self._data, op)()

        # perform the contraction on the data
        data = getattr(self._data, op)(axis=axis)

        # check if the result is scalar or if a result_field must be constr.
        if np.isscalar(data):
            return data
        else:
            return data_object(data)

    def fill(self, fill_value):
        self._data.fill(fill_value)

    def sum(self, axis=None):
        return self._contraction_helper("sum", axis)

    def _binary_helper(self, other, op):
        a = self._data
        if isinstance(other, data_object):
            b = other._data
            if a.shape != b.shape:
                raise ValueError("shapes are incompatible.")
        else:
            b = other

        tval = getattr(a, op)(b)
        return self if tval is a else data_object(tval)

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


def full(shape, fill_value, dtype=None):
    return data_object(np.full(shape, fill_value, dtype))


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
    return data_object(arr)


def local_data(arr):
    return arr._data


def ibegin(arr):
    return (0,)*arr._data.ndim


def np_allreduce_sum(arr):
    return arr


def distaxis(arr):
    return -1


def from_local_data (shape, arr, distaxis):
    if distaxis!=-1:
        raise NotImplementedError
    if shape!=arr.shape:
        raise ValueError
    return data_object(arr)


def from_global_data (arr, distaxis):
    if distaxis!=-1:
        raise NotImplementedError
    return data_object(arr)


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
