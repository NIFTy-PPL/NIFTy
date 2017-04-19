from nifty import Field,\
    FieldArray,\
    LinearOperator,\
    SmoothingOperator,\
    ComposedOperator
import numpy as np

class ResponseOperator(LinearOperator):

    def __init__(self, domain,
                 sigma=[1.], exposure=1., implemented=True,
                 unitary=False):

        self._domain = self._parse_domain(domain)

        shapes = len(self._domain)*[None]
        shape_target = []
        for ii in xrange(len(shapes)):
            shapes[ii] = self._domain[ii].shape
            shape_target = np.append(shape_target, self._domain[ii].shape)

        self._target = self._parse_domain(FieldArray(shape_target,
                                      dtype=np.float64))
        self._sigma = sigma
        self._implemented = implemented
        self._unitary = unitary


        self._kernel = len(self._domain)*[None]

        for ii in xrange(len(self._kernel)):
            self._kernel[ii] = SmoothingOperator(self._domain[ii],
                                        sigma=self._sigma[ii])

        self._composed_kernel = ComposedOperator(self._kernel)

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
        res = self._composed_kernel.times(x)
        res = self._exposure * res
        # res = res.weight(power=1)
        # removing geometric information
        return Field(self._target, val=res.val)

    def _adjoint_times(self, x, spaces):
        # setting correct spaces
        res = x*self._exposure
        res = Field(self.domain, val=res.val)
        res = res.weight(power=-1)
        res = self._composed_kernel.adjoint_times(res, spaces)
        return res
