from builtins import range
from .. import Field, FieldArray, DomainTuple
from .linear_operator import LinearOperator
from .fft_smoothing_operator import FFTSmoothingOperator
from .composed_operator import ComposedOperator
from .diagonal_operator import DiagonalOperator


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
    default_spaces : tuple of ints *optional*
        Defines on which space(s) of a given field the Operator acts by
        default (default: None)

    Attributes
    ----------
    domain : DomainTuple
        The domain on which the Operator's input Field lives.
    target : DomainTuple
        The domain in which the outcome of the operator lives.
    unitary : boolean
        Indicates whether the Operator is unitary or not.

    Raises
    ------
    ValueError:
        raised if:
            * len of sigma-list and exposure-list are not equal
    """

    def __init__(self, domain, sigma=[1.], exposure=[1.], spaces=None):
        super(ResponseOperator, self).__init__()

        if len(sigma) != len(exposure):
            raise ValueError("Length of smoothing kernel and length of"
                             "exposure do not match")
        nsigma = len(sigma)

        self._domain = DomainTuple.make(domain)

        if spaces is None:
            spaces = range(len(self._domain))

        kernel_smoothing = [FFTSmoothingOperator(self._domain, sigma[x],
                                                 space=spaces[x])
                            for x in range(nsigma)]
        kernel_exposure = [DiagonalOperator(Field(self._domain[spaces[x]],
                                                  exposure[x]),
                                            domain=self._domain,
                                            spaces=(spaces[x],))
                           for x in range(nsigma)]

        self._composed_kernel = ComposedOperator(kernel_smoothing)
        self._composed_exposure = ComposedOperator(kernel_exposure)

        target_list = [FieldArray(self._domain[i].shape) for i in spaces]
        self._target = DomainTuple.make(target_list)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return False

    def _times(self, x):
        res = self._composed_kernel.times(x)
        res = self._composed_exposure.times(res)
        # removing geometric information
        return Field(self._target, val=res.val)

    def _adjoint_times(self, x):
        # setting correct spaces
        res = Field(self.domain, val=x.val)
        res = self._composed_exposure.adjoint_times(res)
        res = res.weight(power=-1)
        return self._composed_kernel.adjoint_times(res)
