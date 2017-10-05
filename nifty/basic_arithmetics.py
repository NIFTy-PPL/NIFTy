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
           'conjugate', 'clipped_exp', 'limited_exp', 'limited_exp_deriv']


def _math_helper(x, function, out):
    if not isinstance(x, Field):
        raise TypeError("This function only accepts Field objects.")
    if out is not None:
        if not isinstance(out, Field) or x.domain!=out.domain:
            raise ValueError("Bad 'out' argument")
        function(x.val, out=out.val)
        return out
    else:
        return Field(domain=x.domain, val=function(x.val))


def cos(x, out=None):
    return _math_helper(x, np.cos, out)


def sin(x, out=None):
    return _math_helper(x, np.sin, out)


def cosh(x, out=None):
    return _math_helper(x, np.cosh, out)


def sinh(x, out=None):
    return _math_helper(x, np.sinh, out)


def tan(x, out=None):
    return _math_helper(x, np.tan, out)


def tanh(x, out=None):
    return _math_helper(x, np.tanh, out)


def arccos(x, out=None):
    return _math_helper(x, np.arccos, out)


def arcsin(x, out=None):
    return _math_helper(x, np.arcsin, out)


def arccosh(x, out=None):
    return _math_helper(x, np.arccosh, out)


def arcsinh(x, out=None):
    return _math_helper(x, np.arcsinh, out)


def arctan(x, out=None):
    return _math_helper(x, np.arctan, out)


def arctanh(x, out=None):
    return _math_helper(x, np.arctanh, out)


def sqrt(x, out=None):
    return _math_helper(x, np.sqrt, out)


def exp(x, out=None):
    return _math_helper(x, np.exp, out)


def log(x, out=None):
    return _math_helper(x, np.log, out)


def conjugate(x, out=None):
    return _math_helper(x, np.conjugate, out)


def conj(x, out=None):
    return _math_helper(x, np.conj, out)


def clipped_exp(x, out=None):
    return _math_helper(x, lambda z: np.exp(np.minimum(200, z)), out)


def limited_exp(x, out=None):
    return _math_helper(x, _limited_exp_helper, out)


def _limited_exp_helper(x):
    thr = 200.
    mask = x > thr
    if np.count_nonzero(mask) == 0:
        return np.exp(x)
    result = ((1.-thr) + x)*np.exp(thr)
    result[~mask] = np.exp(x[~mask])
    return result


def limited_exp_deriv(x, out=None):
    return _math_helper(x, _limited_exp_deriv_helper, out)


def _limited_exp_deriv_helper(x):
    thr = 200.
    mask = x > thr
    if np.count_nonzero(mask) == 0:
        return np.exp(x)
    result = np.empty_like(x)
    result[mask] = np.exp(thr)
    result[~mask] = np.exp(x[~mask])
    return result
