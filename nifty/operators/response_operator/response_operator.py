from nifty import Field

from nifty.field_types.field_array import FieldArray

from nifty.operators.linear_operator import LinearOperator
from nifty.operators.smoothing_operator import SmoothingOperator

import numpy as np

class ResponseOperator(LinearOperator):

    def __init__(self, domain,
                 sigma=1., exposure=1., implemented=True,
                 unitary=False):

        self._domain = self._parse_domain(domain)
        self._target = self._parse_domain(FieldArray(self._domain[0].shape,
                                      dtype=np.float64))
        self._sigma = sigma
        self._implemented = implemented
        self._unitary = unitary

        self._kernel = SmoothingOperator(self._domain,
                                        sigma=self._sigma)

        self._exposure = exposure

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def implemented(self):
        return self._implemented

    @property
    def unitary(self):
        return self._unitary

    def _times(self, x, spaces):
        res = self._kernel.times(x)
        res = self._exposure * res
        res = res.weight(power=1)
        # removing geometric information
        return Field(self._target, val=res.val)

    def _adjoint_times(self, x, spaces):
        # setting correct spaces
        res = x*self._exposure
        res = Field(self.domain, val=res.val)
        res = self._kernel.adjoint_times(res)
        return res
