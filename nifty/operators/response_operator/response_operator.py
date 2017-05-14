import numpy as np
from nifty import Field,\
                  FieldArray
from nifty.operators.linear_operator import LinearOperator
from nifty.operators.smoothing_operator import SmoothingOperator
from nifty.operators.composed_operator import ComposedOperator
from nifty.operators.diagonal_operator import DiagonalOperator

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
        The domain in which the outcome of the operator lives. As the Operator
        is endomorphic this is the same as its domain.
    unitary : boolean
        Indicates whether the Operator is unitary or not.

    Raises
    ------
    ValueError:
        raised if:
            * len of sigma-list and exposure-list are not equal

    Notes
    -----

    Examples
    --------
    >>> x1 = RGSpace(5)
    >>> x2 = RGSpace(10)
    >>> R = ResponseOperator(domain=(x1,x2), sigma=[.5, .25],
                             exposure=[2.,3.])
    >>> f = Field((x1,x2), val=4.)
    >>> R.times(f)
    <distributed_data_object>
    array([[ 24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.],
           [ 24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.],
           [ 24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.],
           [ 24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.],
           [ 24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.,  24.]])

    See Also
    --------

    """

    def __init__(self, domain,
                 sigma=[1.], exposure=[1.],
                 default_spaces=None):
        super(ResponseOperator, self).__init__(default_spaces)

        self._domain = self._parse_domain(domain)

        shapes = len(self._domain)*[None]
        shape_target = []
        for ii in xrange(len(shapes)):
            shapes[ii] = self._domain[ii].shape
            shape_target = np.append(shape_target, self._domain[ii].shape)

        self._target = self._parse_domain(FieldArray(shape_target))
        self._sigma = sigma
        self._exposure = exposure

        self._kernel = len(self._domain)*[None]

        for ii in xrange(len(self._kernel)):
            self._kernel[ii] = SmoothingOperator(self._domain[ii],
                                                 sigma=self._sigma[ii])

        self._composed_kernel = ComposedOperator(self._kernel)

        self._exposure_op = len(self._domain)*[None]
        if len(self._exposure_op) != len(self._kernel):
            raise ValueError("Definition of kernel and exposure do not suit "
                             "each other")
        else:
            for ii in xrange(len(self._exposure_op)):
                self._exposure_op[ii] = DiagonalOperator(
                                                self._domain[ii],
                                                diagonal=self._exposure[ii])
            self._composed_exposure = ComposedOperator(self._exposure_op)

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
        # res = res.weight(power=1)
        # removing geometric information
        return Field(self._target, val=res.val)

    def _adjoint_times(self, x, spaces):
        # setting correct spaces
        res = Field(self.domain, val=x.val)
        res = self._composed_exposure.adjoint_times(res, spaces)
        res = res.weight(power=-1)
        res = self._composed_kernel.adjoint_times(res, spaces)
        return res
