from .. import Field, FieldArray, DomainTuple
from .linear_operator import LinearOperator
from .fft_smoothing_operator import FFTSmoothingOperator
from .diagonal_operator import DiagonalOperator
from .scaling_operator import ScalingOperator
import numpy as np


class GeometryRemover(LinearOperator):
    def __init__(self, domain):
        super(GeometryRemover, self).__init__()
        self._domain = DomainTuple.make(domain)
        target_list = [FieldArray(dom.shape) for dom in self._domain]
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

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return Field(self._target, val=x.weight(1).val)
        return Field(self._domain, val=x.val).weight(1)


def ResponseOperator(domain, sigma, sensitivity):
    # sensitivity has units 1/field/volume and gives a measure of how much
    # the instrument will excited when it is exposed to a certain field
    # volume amplitude
    domain = DomainTuple.make(domain)
    ncomp = len(sensitivity)
    if len(sigma) != ncomp or len(domain) != ncomp:
        raise ValueError("length mismatch between sigma, sensitivity "
                         "and domain")
    ncomp = len(sigma)
    if ncomp == 0:
        raise ValueError("Empty response operator not allowed")

    kernel = None
    sensi = None
    for i in range(ncomp):
        if sigma[i] > 0:
            op = FFTSmoothingOperator(domain, sigma[i], space=i)
            kernel = op if kernel is None else op*kernel
        if np.isscalar(sensitivity[i]):
            if sensitivity[i] != 1.:
                op = ScalingOperator(sensitivity[i], domain)
                sensi = op if sensi is None else op*sensi
        elif isinstance(sensitivity[i], Field):
            op = DiagonalOperator(sensitivity[i], domain=domain, spaces=i)
            sensi = op if sensi is None else op*sensi

    res = GeometryRemover(domain)
    if sensi is not None:
        res = res * sensi
    if kernel is not None:
        res = res * kernel
    return res
