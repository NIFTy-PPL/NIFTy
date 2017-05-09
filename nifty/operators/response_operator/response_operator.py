import numpy as np
from nifty import Field,\
                  FieldArray
from nifty.operators.linear_operator import LinearOperator
from nifty.operators.smoothing_operator import SmoothingOperator
from nifty.operators.composed_operator import ComposedOperator
from nifty.operators.diagonal_operator import DiagonalOperator

class ResponseOperator(LinearOperator):

    def __init__(self, domain,
                 sigma=[1.], exposure=[1.],
                 unitary=False):

        self._domain = self._parse_domain(domain)

        shapes = len(self._domain)*[None]
        shape_target = []
        for ii in xrange(len(shapes)):
            shapes[ii] = self._domain[ii].shape
            shape_target = np.append(shape_target, self._domain[ii].shape)

        self._target = self._parse_domain(FieldArray(shape_target))
        self._sigma = sigma
        self._exposure = exposure
        self._unitary = unitary


        self._kernel = len(self._domain)*[None]

        for ii in xrange(len(self._kernel)):
            self._kernel[ii] = SmoothingOperator(self._domain[ii],
                                        sigma=self._sigma[ii])

        self._composed_kernel = ComposedOperator(self._kernel)


        self._exposure_op = len(self._domain)*[None]
        if len(self._exposure_op)!= len(self._kernel):
            raise ValueError("Definition of kernel and exposure do not suit each other")
        else:
            for ii in xrange(len(self._exposure_op)):
                self._exposure_op[ii] = DiagonalOperator(self._domain[ii],
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
        return self._unitary

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
