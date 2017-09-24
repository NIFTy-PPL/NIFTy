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
from .field import Field
from . import dobj


__all__ = ['cos', 'sin', 'cosh', 'sinh', 'tan', 'tanh', 'arccos', 'arcsin',
           'arccosh', 'arcsinh', 'arctan', 'arctanh', 'sqrt', 'exp', 'log',
           'conj', 'conjugate']


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
    return _math_helper(x, dobj.cos, out)


def sin(x, out=None):
    return _math_helper(x, dobj.sin, out)


def cosh(x, out=None):
    return _math_helper(x, dobj.cosh, out)


def sinh(x, out=None):
    return _math_helper(x, dobj.sinh, out)


def tan(x, out=None):
    return _math_helper(x, dobj.tan, out)


def tanh(x, out=None):
    return _math_helper(x, dobj.tanh, out)


def arccos(x, out=None):
    return _math_helper(x, dobj.arccos, out)


def arcsin(x, out=None):
    return _math_helper(x, dobj.arcsin, out)


def arccosh(x, out=None):
    return _math_helper(x, dobj.arccosh, out)


def arcsinh(x, out=None):
    return _math_helper(x, dobj.arcsinh, out)


def arctan(x, out=None):
    return _math_helper(x, dobj.arctan, out)


def arctanh(x, out=None):
    return _math_helper(x, dobj.arctanh, out)


def sqrt(x, out=None):
    return _math_helper(x, dobj.sqrt, out)


def exp(x, out=None):
    return _math_helper(x, dobj.exp, out)


def log(x, out=None):
    return _math_helper(x, dobj.log, out)


def conjugate(x, out=None):
    return _math_helper(x, dobj.conjugate, out)


def conj(x, out=None):
    return _math_helper(x, dobj.conj, out)
