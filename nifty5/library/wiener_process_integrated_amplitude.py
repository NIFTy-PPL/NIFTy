import nifty5 as ift
import numpy as np


class WienerProcessIntegratedAmplitude(ift.LinearOperator):
    def __init__(self, target):
        # target is PowerSpace
        self._target = ift.makeDomain(target)
        self._domain = ift.makeDomain(
            ift.UnstructuredDomain(self.target.shape[0] - 2))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            x = x.to_global_data()
            res = np.empty(self._target.shape)
            res[0] = 0
            res[1] = 0
            res[2:] = np.cumsum(x)
            res[1:] = np.cumsum(res[1:])
            return ift.from_global_data(self._target, res)
        else:
            x = x.to_global_data()
            res = np.empty(self._target.shape)
            res[1:] = np.cumsum(x[1:][::-1])[::-1]
            res[2:] = np.cumsum(res[2:][::-1])[::-1]
            return ift.from_global_data(self._domain, res[2:])


if __name__ == '__main__':
    np.random.seed(42)
    ndim = 1
    sspace = ift.RGSpace(
        np.linspace(16, 20, num=ndim).astype(np.int),
        np.linspace(2.3, 7.99, num=ndim))
    hspace = sspace.get_default_codomain()
    target = ift.PowerSpace(hspace)
    op = WienerProcessIntegratedAmplitude(target)
    ift.extra.consistency_check(op)
    fld = ift.from_random('normal', op.domain)
    ift.single_plot(op(fld) + 100, name='debug.png')
