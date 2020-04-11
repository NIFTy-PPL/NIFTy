import nifty6 as ift
from differential_tensor import DiagonalTensor, ComposedTensor

class MultiLinearization:
    def __init__(self, res, jacs):
        self._res = res
        self._jacs = jacs

    @property
    def val(self):
        return self._res
    @property
    def jacs(self):
        return self._jacs
    @property
    def maxderiv(self):
        return len(self._jacs)
    
    @staticmethod
    def make_var(val, maxorder):
        jacs = [ift.makeOp(ift.full(val.domain,1.)), ]
        for i in range(maxorder-1):
            jacs.append(DiagonalTensor(val.domain, val.domain,
                                       ift.full(val.domain,0.), i+2))
        return MultiLinearization(val, jacs)

    def new(self, val, jacs):
        assert len(jacs) == len(self.jacs)
        gn = [jacs[0]@self.jacs[0], ]
        for i in range(len(self.jacs))[1:]:
            gn.append(ComposedTensor(self.jacs[:(i+1)], jacs[:(i+1)]))
        return MultiLinearization(val, gn)
