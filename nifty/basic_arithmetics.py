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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division
import numpy as np
from d2o import distributed_data_object
from .field import Field


__all__ = ['cos', 'sin', 'cosh', 'sinh', 'tan', 'tanh', 'arccos', 'arcsin',
           'arccosh', 'arcsinh', 'arctan', 'arctanh', 'sqrt', 'exp', 'log',
           'conjugate']


def _math_helper(x, function):
    if isinstance(x, Field):
        result_val = x.val.apply_scalar_function(function)
        result = x.copy_empty(dtype=result_val.dtype)
        result.val = result_val
    elif isinstance(x, distributed_data_object):
        result = x.apply_scalar_function(function, inplace=False)
    else:
        result = function(np.asarray(x))

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
