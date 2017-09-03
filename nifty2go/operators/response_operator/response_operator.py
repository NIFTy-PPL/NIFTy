from builtins import range
import numpy as np
from ... import Field,\
                FieldArray
from ..linear_operator import LinearOperator
from ..smoothing_operator import FFTSmoothingOperator
from ..composed_operator import ComposedOperator
from ..diagonal_operator import DiagonalOperator


class ResponseOperator(LinearOperator):
    """ NIFTy ResponseOperator (example)

    This NIFTy ResponseOperator provides the user with an example how a
    ResponseOperator can look like. It smoothes and exposes a field. The
    outcome of the Operator is geometrically not ordered as typical data
    set are.

    Parameters
    ----------
    domain : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domains on which the operator lives. Either one space or a list
        of spaces
    sigma : list(np.float)
        Defines the smoothing length of the operator for each space it lives on
    exposure : list(np.float)
        Defines the exposure of the operator for each space it lives on
    default_spaces : tuple of ints *optional*
        Defines on which space(s) of a given field the Operator acts by
        default (default: None)

    Attributes
    ----------
    domain : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domain on which the Operator's input Field lives.
    target : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domain in which the outcome of the operator lives.
    unitary : boolean
        Indicates whether the Operator is unitary or not.

    Raises
    ------
    ValueError:
        raised if:
            * len of sigma-list and exposure-list are not equal

    """

    def __init__(self, domain, sigma=[1.], exposure=[1.],
                 default_spaces=None):
        super(ResponseOperator, self).__init__(default_spaces)

        if len(sigma) != len(exposure):
            raise ValueError("Length of smoothing kernel and length of"
                             "exposure do not match")
        nsigma = len(sigma)

        self._domain = self._parse_domain(domain)

        kernel_smoothing = [FFTSmoothingOperator(self._domain[x], sigma[x])
                            for x in range(nsigma)]
        kernel_exposure = [DiagonalOperator(self._domain[x],
                           diagonal=exposure[x]) for x in range(nsigma)]

        self._composed_kernel = ComposedOperator(kernel_smoothing)
        self._composed_exposure = ComposedOperator(kernel_exposure)

        target_list = [FieldArray(x.shape) for x in self.domain]
        self._target = self._parse_domain(target_list)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return False

    def _times(self, x, spaces):
        res = self._composed_kernel.times(x, spaces)
        res = self._composed_exposure.times(res, spaces)
        # removing geometric information
        return Field(self._target, val=res.val)

    def _adjoint_times(self, x, spaces):
        # setting correct spaces
        res = Field(self.domain, val=x.val)
        res = self._composed_exposure.adjoint_times(res, spaces)
        res = res.weight(power=-1)
        return self._composed_kernel.adjoint_times(res, spaces)
