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
# Copyright(C) 2020 Max-Planck-Society
# Author: Martin Reinecke
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np


def _sqrt_helper(v):
    tmp = np.sqrt(v)
    return (tmp, 0.5/tmp)


def _sinc_helper(v):
    tmp = np.sinc(v)
    tmp2 = (np.cos(np.pi*v)-tmp)/v
    return (tmp, np.where(v==0., 0, tmp2))


def _expm1_helper(v):
    tmp = np.expm1(v)
    return (tmp, tmp+1.)


def _tanh_helper(v):
    tmp = np.tanh(v)
    return (tmp, 1.-tmp**2)


def _sigmoid_helper(v):
    tmp = np.tanh(v)
    tmp2 = 0.5+(0.5*tmp)
    return (tmp2, 0.5-0.5*tmp**2)


def _reciprocal_helper(v):
    tmp = 1./v
    return (tmp, -tmp**2)


def _abs_helper(v):
    if np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError("Argument must not be complex")
    return (np.abs(v), np.where(v==0, np.nan, np.sign(v)))


def _sign_helper(v):
    if np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError("Argument must not be complex")
    return (np.sign(v), np.where(v==0, np.nan, 0))


def _power_helper(v, expo):
    return (np.power(v, expo), expo*np.power(v, expo-1))


def _clip_helper(v, a_min, a_max):
    if np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError("Argument must not be complex")
    tmp = np.clip(v, a_min, a_max)
    tmp2 = np.ones(v.shape)
    if a_min is not None:
        tmp2 = np.where(tmp==a_min, 0., tmp2)
    if a_max is not None:
        tmp2 = np.where(tmp==a_max, 0., tmp2)
    return (tmp, tmp2)


ptw_dict = {
    "sqrt": (np.sqrt, _sqrt_helper),
    "sin" : (np.sin, lambda v: (np.sin(v), np.cos(v))),
    "cos" : (np.cos, lambda v: (np.cos(v), -np.sin(v))),
    "tan" : (np.tan, lambda v: (np.tan(v), 1./np.cos(v)**2)),
    "sinc": (np.sinc, _sinc_helper),
    "exp" : (np.exp, lambda v: (2*(np.exp(v),))),
    "expm1" : (np.expm1, _expm1_helper),
    "log" : (np.log, lambda v: (np.log(v), 1./v)),
    "log10": (np.log10, lambda v: (np.log10(v), (1./np.log(10.))/v)),
    "log1p": (np.log1p, lambda v: (np.log1p(v), 1./(1.+v))),
    "sinh": (np.sinh, lambda v: (np.sinh(v), np.cosh(v))),
    "cosh": (np.cosh, lambda v: (np.cosh(v), np.sinh(v))),
    "tanh": (np.tanh, _tanh_helper),
    "sigmoid": (lambda v: 0.5+(0.5*np.tanh(v)), _sigmoid_helper),
    "reciprocal": (lambda v: 1./v, _reciprocal_helper),
    "abs": (np.abs, _abs_helper),
    "absolute": (np.abs, _abs_helper),
    "sign": (np.sign, _sign_helper),
    "power": (np.power, _power_helper),
    "clip": (np.clip, _clip_helper),
    }
