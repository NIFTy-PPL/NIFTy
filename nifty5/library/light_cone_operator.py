from __future__ import absolute_import, division, print_function

from ..field import Field
from ..domain_tuple import DomainTuple
from ..linearization import Linearization
from ..operators.linear_operator import LinearOperator
from ..operators.operator import Operator

import numpy as np

def make_coords(domain, absolute=False):
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

def field_from_function(domain, func, absolute=False):
    domain = DomainTuple.make(domain)
    k_array = make_coords(domain, absolute=absolute)
    return Field.from_global_data(domain, func(k_array))

class LightConeDerivative(LinearOperator):
    def __init__(self, domain, target, derivatives):
        super(LightConeDerivative, self).__init__()
        self._domain = domain
        self._target = target
        self._derivatives = derivatives
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        res = np.zeros(self._tgt(mode).shape, dtype=self._derivatives.dtype)
        for i in range(self.domain.shape[0]):
            if mode == self.TIMES:
                res += self._derivatives[i]*x[i]
            else:
                res[i] = np.sum(self._derivatives[i]*x)
        return Field.from_global_data(self._tgt(mode), res)

    
def cone_arrays(c, domain, sigx,want_gradient):
    x = make_coords(domain)
    a = np.zeros(domain.shape, dtype=np.complex)
    if want_gradient:
        derivs = np.zeros((c.size,) + domain.shape, dtype=np.complex)
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
    def __init__(self, domain, target, sigx):
        self._domain = domain
        self._target = target
        self._sigx = sigx

    def apply(self, x):
        islin = isinstance(x, Linearization)
        val = x.val.to_global_data() if islin else x.to_global_data()
        a, derivs = cone_arrays(val, self.target, self._sigx, islin)
        res = Field.from_global_data(self.target, a)
        if not islin:
            return res
        jac = LightConeDerivative(x.jac.target, self.target, derivs)(x.jac)
        return Linearization(res, jac, want_metric=x.want_metric)
