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
    fv = np.sinc(v)
    df = np.empty(v.shape, dtype=v.dtype)
    sel = v != 0.
    v = v[sel]
    df[sel] = (np.cos(np.pi*v)-fv[sel])/v
    df[~sel] = 0
    return (fv, df)


def _expm1_helper(v, maxorder):
    tmp = np.expm1(v)
    return (tmp, ) + (tmp+1.,)*maxorder


def _tanh_helper(v):
    tmp = np.tanh(v)
    return (tmp, 1.-tmp**2)


def _sigmoid_helper(v, n):
    t = np.tanh(v)
    tp = 1.-t**2
    res = (0.5*(1.+t), 0.5*tp)
    if n == 1:
        return res
    tpp = -2.*t*tp
    res = res + (0.5*tpp,)
    if n == 2:
        return res
    tppp = -2.*(t*tpp+tp**2)
    res = res + (0.5*tppp,)
    if n == 3:
        return res
    tpppp = -2.*(t*tppp + 3.*tp*tpp)
    res = res + (0.5*tpppp,)
    if n == 4:
        return res
    tp5 = -2.*(t*tpppp + 4.*tp*tppp + 3.*tpp**2)
    res = res + (0.5*tp5,)
    if n == 5:
        return res
    tp6 = -2.*(t*tp5 + 5.*tp*tpppp + 10.*tpp*tppp)
    res = res + (0.5*tp6,)
    if n == 6:
        return res
    raise NotImplementedError


def _reciprocal_helper(v, maxorder):
    tmp = 1./v
    return (tmp, ) + tuple(tmp**(m+1)*np.math.factorial(m)*(1.-2.*(m%2)) for m in range(1,maxorder+1))

def _log_helper(v, maxorder):
    return (np.log(v), ) + _reciprocal_helper(v, maxorder-1)

def _abs_helper(v):
    if np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError("Argument must not be complex because abs(z) is not holomorphic")
    return (np.abs(v), np.where(v == 0, np.nan, np.sign(v)))


def _sign_helper(v):
    if np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError("Argument must not be complex")
    return (np.sign(v), np.where(v == 0, np.nan, 0))


def _power_helper(v, maxorder, expo):
    res = ()
    fac = 1.
    for i in range(maxorder+1):
        res += (fac*np.power(v,expo-i),)
        fac *= expo-i
    return res

def _clip_helper(v, a_min, a_max):
    if np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError("Argument must not be complex")
    tmp = np.clip(v, a_min, a_max)
    tmp2 = np.ones(v.shape)
    if a_min is not None:
        tmp2 = np.where(tmp == a_min, 0., tmp2)
    if a_max is not None:
        tmp2 = np.where(tmp == a_max, 0., tmp2)
    return (tmp, tmp2)

def _sc_helper(v, maxorder, sin):
    s,c = np.sin(v), np.cos(v)
    return tuple(s if i%2 ^ sin else c for i in range(maxorder+1))
        

def softplus(v):
    fv = np.empty(v.shape, dtype=np.float64 if np.isrealobj(v) else np.complex128)
    selp = v > 33
    selm = v < -33
    sel0 = ~np.logical_or(selp, selm)
    fv[selp] = v[selp]
    fv[sel0] = np.log(1+np.exp(v[sel0]))
    fv[selm] = 0
    return fv


def _softplus_helper(v):
    dtype = np.float64 if np.isrealobj(v) else np.complex128
    fv = np.empty(v.shape, dtype=dtype)
    dfv = np.empty(v.shape, dtype=dtype)
    selp = 33 < v
    selm = v < -33
    sel0 = ~np.logical_or(selp, selm)
    fv[selp] = v[selp]
    fv[sel0] = np.log(1+np.exp(v[sel0]))
    fv[selm] = 0
    dfv[selp] = 1
    dfv[sel0] = 1/(1+np.exp(-v[sel0]))
    dfv[selm] = 0
    return fv, dfv


ptw_dict = {
    "sqrt": (np.sqrt, _sqrt_helper, lambda v,maxorder: _power_helper(v,maxorder,0.5)),
    "sin": (np.sin, lambda v: (np.sin(v), np.cos(v)), lambda v,m: _sc_helper(v,m,True)),
    "cos": (np.cos, lambda v: (np.cos(v), -np.sin(v)), lambda v,m: _sc_helper(v,m,False)),
    "tan": (np.tan, lambda v: (np.tan(v), 1./np.cos(v)**2), None),
    "sinc": (np.sinc, _sinc_helper, None),
    "exp": (np.exp, lambda v: (2*(np.exp(v),)), lambda v,m: (m+1)*(np.exp(v), )),
    "expm1": (np.expm1, lambda v: _expm1_helper(v,1), _expm1_helper),
    "log": (np.log, lambda v: (np.log(v), 1./v), _log_helper),
    "log10": (np.log10, lambda v: (np.log10(v), (1./np.log(10.))/v)),
    "log1p": (np.log1p, lambda v: (np.log1p(v), 1./(1.+v))),
    "sinh": (np.sinh, lambda v: (np.sinh(v), np.cosh(v))),
    "cosh": (np.cosh, lambda v: (np.cosh(v), np.sinh(v))),
    "tanh": (np.tanh, _tanh_helper),
    "sigmoid": (lambda v: 0.5+(0.5*np.tanh(v)), lambda v: _sigmoid_helper(v,1),
                lambda v,m: _sigmoid_helper(v,m)),
    "reciprocal": (lambda v: 1./v, lambda v:_reciprocal_helper(v,1), _reciprocal_helper),
    "abs": (np.abs, _abs_helper),
    "absolute": (np.abs, _abs_helper),
    "sign": (np.sign, _sign_helper),
    "power": (np.power, lambda v,expo: _power_helper(v,1,expo), _power_helper),
    "clip": (np.clip, _clip_helper),
    "softplus": (softplus, _softplus_helper)
    }
