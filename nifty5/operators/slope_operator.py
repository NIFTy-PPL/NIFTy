import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class SlopeOperator(LinearOperator):
    def __init__(self, domain, target, sigmas):
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)

        if self.domain[0].shape != (len(self.target[0].shape) + 1,):
            raise AssertionError("Shape mismatch!")

        self._sigmas = sigmas
        self.ndim = len(self.target[0].shape)
        self.pos = np.zeros((self.ndim,) + self.target[0].shape)

        if self.ndim == 1:
            self.pos[0] = self.target[0].get_k_length_array().to_global_data()
        else:
            shape = self.target[0].shape
            for i in range(self.ndim):
                rng = np.arange(target.shape[i])
                tmp = np.minimum(
                    rng, target.shape[i] + 1 - rng) * target.bindistances[i]
                fst_dims = (1,) * i
                lst_dims = (1,) * (self.ndim - i - 1)
                self.pos[i] += tmp.reshape(fst_dims + (shape[i],) + lst_dims)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    def apply(self, x, mode):
        self._check_input(x, mode)

        # Times
        if mode == self.TIMES:
            inp = x.to_global_data()
            res = self._sigmas[-1] * inp[-1]
            for i in range(self.ndim):
                res += self._sigmas[i] * inp[i] * self.pos[i]
            return Field.from_global_data(self.target, res)

        # Adjoint times
        res = np.zeros(self.domain[0].shape)
        xglob = x.to_global_data()
        res[-1] = np.sum(xglob) * self._sigmas[-1]
        for i in range(self.ndim):
            res[i] = np.sum(self.pos[i] * xglob) * self._sigmas[i]
        return Field.from_global_data(self.domain, res)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
