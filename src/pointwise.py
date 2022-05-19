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
# Copyright(C) 2020-2021 Max-Planck-Society
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
        raise TypeError("Argument must not be complex because abs(z) is not holomorphic")
    return (np.abs(v), np.where(v == 0, np.nan, np.sign(v)))


def _sign_helper(v):
    if np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError("Argument must not be complex")
    return (np.sign(v), np.where(v == 0, np.nan, 0))


def _power_helper(v, expo):
    return (np.power(v, expo), expo*np.power(v, expo-1))


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

def _step_helper(v, grad):
    if np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError("Argument must not be complex")
    r = np.zeros(v.shape)
    r[v>=0.] = 1.
    if grad:
        return (r, np.zeros(v.shape))
    return r

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


def exponentiate(v, base):
    return np.power(base, v)


def _exponentiate_helper(v, base):
    tmp = np.power(base, v)
    return (tmp, np.log(base) * tmp)


ptw_dict = {
    "sqrt": (np.sqrt, _sqrt_helper),
    "sin": (np.sin, lambda v: (np.sin(v), np.cos(v))),
    "cos": (np.cos, lambda v: (np.cos(v), -np.sin(v))),
    "tan": (np.tan, lambda v: (np.tan(v), 1./np.cos(v)**2)),
    "sinc": (np.sinc, _sinc_helper),
    "exp": (np.exp, lambda v: (2*(np.exp(v),))),
    "expm1": (np.expm1, _expm1_helper),
    "log": (np.log, lambda v: (np.log(v), 1./v)),
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
    "softplus": (softplus, _softplus_helper),
    "exponentiate": (exponentiate, _exponentiate_helper),
    "arctan": (np.arctan, lambda v: (np.arctan(v), 1./(1.+v**2))),
    "unitstep": (lambda v: _step_helper(v, False), lambda v: _step_helper(v, True))
    }


def sigmoid_j(v):
    from jax import numpy as jnp

    # NOTE, the sigmoid used in NIFTy is different to the one commonly referred
    # to as sigmoid in most of the literature.
    return 0.5 + (0.5 * jnp.tanh(v))


def exponentiate_j(v, base):
    from jax import numpy as jnp

    return jnp.power(base, v)


ptw_nifty2jax_dict = {
    "sigmoid": sigmoid_j,
    "exponentiate": exponentiate_j,
}
