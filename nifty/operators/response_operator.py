from builtins import range
from .. import Field, FieldArray, DomainTuple
from .linear_operator import LinearOperator
from .fft_smoothing_operator import FFTSmoothingOperator
from .diagonal_operator import DiagonalOperator
from .scaling_operator import ScalingOperator
import numpy as np


class ResponseOperator(LinearOperator):
    """ NIFTy ResponseOperator (example)

    This NIFTy ResponseOperator provides the user with an example how a
    ResponseOperator can look like. It smoothes and exposes a field. The
    outcome of the Operator is geometrically not ordered as typical data
    set are.

    Parameters
    ----------
    domain : tuple of DomainObjects
        The domains on which the operator lives. Either one space or a list
        of spaces
    sigma : list(np.float)
        Defines the smoothing length of the operator for each space it lives on
    exposure : list(np.float)
        Defines the exposure of the operator for each space it lives on
    spaces : tuple of ints *optional*
        Defines on which subdomain the individual components act
        (default: None)
    """

    def __init__(self, domain, sigma, exposure, spaces=None):
        super(ResponseOperator, self).__init__()

        self._domain = DomainTuple.make(domain)

        if spaces is None:
            spaces = tuple(range(len(self._domain)))
        if len(sigma) != len(exposure) or len(sigma) != len(spaces):
            raise ValueError("length mismatch between sigma, exposure, "
                             "and spaces ")
        ncomp = len(sigma)
        if ncomp == 0:
            raise ValueError("Empty response operator not allowed")

        self._kernel = None
        for i in range(ncomp):
            op = FFTSmoothingOperator(self._domain, sigma[i], space=spaces[i])
            self._kernel = op if self._kernel is None else op*self._kernel

        self._exposure = None
        for i in range(ncomp):
            if np.isscalar(exposure[i]):
                op = ScalingOperator(exposure[i], self._domain)
            elif isinstance(exposure[i], Field):
                op = DiagonalOperator(exposure[i], domain=self._domain,
                                      spaces=(spaces[i],))
            self._exposure = op if self._exposure is None\
                else op*self._exposure

        target_list = [FieldArray(self._domain[i].shape) for i in spaces]
        self._target = DomainTuple.make(target_list)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def _times(self, x):
        res = self._exposure(self._kernel(x))
        # removing geometric information
        return Field(self._target, val=res.val)

    def _adjoint_times(self, x):
        # setting correct spaces
        res = Field(self.domain, val=x.val)
        res = self._exposure.adjoint_times(res)
        res = res.weight(power=-1)
        return self._kernel.adjoint_times(res)

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._times(x) if mode == self.TIMES else self._adjoint_times(x)
