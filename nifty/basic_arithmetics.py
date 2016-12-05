
import numpy as np
from d2o import distributed_data_object
from nifty.field import Field


__all__ = ['cos', 'sin', 'cosh', 'sinh', 'tan', 'tanh', 'arccos', 'arcsin',
           'arccosh', 'arcsinh', 'arctan', 'arctanh', 'sqrt', 'exp', 'log',
           'conjugate']


def _math_helper(x, function, inplace=False):
    if isinstance(x, Field):
        if inplace:
            x.val.apply_scalar_function(function, inplace=True)
            result = x
        else:
            result_val = x.val.apply_scalar_function(function)
            result = x.copy_empty(dtype=result_val.dtype)
            result.val = result_val

    elif isinstance(x, distributed_data_object):
        result = x.apply_scalar_function(function, inplace=inplace)

    else:
        result = function(np.asarray(x))
        if inplace:
            x[:] = result
            result = x

    return result


def cos(x):
    return _math_helper(x, np.cos)


def sin(x):
    return _math_helper(x, np.sin)


def cosh(x):
    return _math_helper(x, np.cosh)


def sinh(x):
    return _math_helper(x, np.sinh)


def tan(x):
    return _math_helper(x, np.tan)


def tanh(x):
    return _math_helper(x, np.tanh)


def arccos(x):
    return _math_helper(x, np.arccos)


def arcsin(x):
    return _math_helper(x, np.arcsin)


def arccosh(x):
    return _math_helper(x, np.arccosh)


def arcsinh(x):
    return _math_helper(x, np.arcsinh)


def arctan(x):
    return _math_helper(x, np.arctan)


def arctanh(x):
    return _math_helper(x, np.arctanh)


def sqrt(x):
    return _math_helper(x, np.sqrt)


def exp(x):
    return _math_helper(x, np.exp)


def log(x, base=None):
    result = _math_helper(x, np.log)
    if base is not None:
        result = result/log(base)

    return result


def conjugate(x):
    return _math_helper(x, np.conjugate)


def conj(x):
    return _math_helper(x, np.conjugate)
