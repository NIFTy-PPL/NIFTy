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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from ..operators.linear_operator import LinearOperator
from ..operators.operator import Operator


def _field_from_function(domain, func, absolute=False):
    domain = DomainTuple.make(domain)
    k_array = _make_coords(domain, absolute=absolute)
    return Field(domain, func(k_array))


def _make_coords(domain, absolute=False):
    domain = DomainTuple.make(domain)
    dim = len(domain.shape)
    dist = domain[0].distances
    shape = domain.shape
    k_array = np.zeros((dim,) + shape)
    for i in range(dim):
        ks = np.minimum(shape[i] - np.arange(shape[i]), np.arange(
            shape[i]))*dist[i]
        if not absolute:
            ks[int(shape[i]/2) + 1:] *= -1
        fst_dims = (1,)*i
        lst_dims = (1,)*(dim - i - 1)
        k_array[i] += ks.reshape(fst_dims + (shape[i],) + lst_dims)
    return k_array


class _LightConeDerivative(LinearOperator):
    def __init__(self, domain, target, derivatives):
        super(_LightConeDerivative, self).__init__()
        self._domain = domain
        self._target = target
        self._derivatives = derivatives
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        res = np.zeros(self._tgt(mode).shape, dtype=self._derivatives.dtype)
        for i in range(self.domain.shape[0]):
            if mode == self.TIMES:
                res += self._derivatives[i]*x[i]
            else:
                res[i] = np.sum(self._derivatives[i]*x.real)
        return Field(self._tgt(mode), res)


def _cone_arrays(c, domain, sigx, want_gradient):
    x = _make_coords(domain)
    a = np.zeros(domain.shape, dtype=np.complex128)
    if want_gradient:
        derivs = np.zeros((c.size,) + domain.shape, dtype=np.complex128)
    else:
        derivs = None
    a -= (x[0]/(sigx*domain[0].distances[0]))**2
    for i in range(c.size):
        res = (x[i + 1]/(sigx*domain[0].distances[i + 1]))**2
        a += c[i]*res
        if want_gradient:
            derivs[i] = res
    a = np.sqrt(a)
    if want_gradient:
        derivs *= -0.5
        for i in range(c.size):
            derivs[i][a == 0] = 0.
            derivs[i][a != 0] /= a[a != 0]
    a = a.real
    if want_gradient:
        derivs *= a
    a = np.exp(-0.5*a**2)
    if want_gradient:
        derivs = a*derivs.real
    return a, derivs


class LightConeOperator(Operator):
    '''Constructs a Light cone from a set of lightspeed parameters.

    The resulting cone is defined as follows

    .. math::
        \\exp \\left(- \\frac{1}{2} \\Re \\left( \\Delta \\right)^2 \\right)

    with

    .. math::
        \\Delta = \\sqrt{- \\left(t^2 - \\frac{x^\\dagger C^{-1} x}
        {\\sigma_x^2} \\right)}

    where t and x are the coordinates of the target space. Note that axis zero
    of the space is interpreted as the time axis. C denotes the input
    paramters of the operator and parametrizes the shape of the cone.
    sigx is the width of the asymptotic Gaussian in x necessary for
    discretization.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain of the input parameters of the light cone, the values of the
        lightspeed tensor.
    target : Domain, tuple of Domain or DomainTuple
        Output space on which the lightcone should be defined. The zeroth axis
        of this space is interpreted as the time axis.
    sigx : float
        Width of the Gaussian for the discretized representation of the cone.
    '''
    def __init__(self, domain, target, sigx):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)
        self._sigx = sigx

    def apply(self, x):
        lin = x.jac is not None
        a, derivs = _cone_arrays(x.val.val if lin else x.val, self.target, self._sigx, lin)
        res = Field(self.target, a)
        if not lin:
            return res
        jac = _LightConeDerivative(self._domain, self._target, derivs)
        return x.new(res, jac)
