import nifty5 as ift
import numpy as np


class _TwoLogIntegrations(ift.LinearOperator):
    def __init__(self, target):
        self._target = ift.makeDomain(target)
        self._domain = ift.makeDomain(
            ift.UnstructuredDomain(self.target.shape[0] - 2))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        if not isinstance(self._target[0], ift.PowerSpace):
            raise TypeError
        logk_lengths = np.log(self._target[0].k_lengths[1:])
        self._logvol = logk_lengths[1:] - logk_lengths[:-1]

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            x = x.to_global_data()
            res = np.empty(self._target.shape)
            res[0] = 0
            res[1] = 0
            res[2:] = np.cumsum(x*self._logvol)
            res[2:] = np.cumsum(res[2:]*self._logvol)
            return ift.from_global_data(self._target, res)
        else:
            x = x.to_global_data()
            res = np.empty(self._target.shape)
            res[2:] = np.cumsum(x[2:][::-1])[::-1]*self._logvol
            res[2:] = np.cumsum(res[2:][::-1])[::-1]*self._logvol
            return ift.from_global_data(self._domain, res[2:])


class _Rest(ift.LinearOperator):
    def __init__(self, target):
        self._target = ift.makeDomain(target)
        self._domain = ift.makeDomain(ift.UnstructuredDomain(3))
        self._logk_lengths = np.log(self._target[0].k_lengths[1:])
        self._logk_lengths -= self._logk_lengths[0]
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        res = np.empty(self._tgt(mode).shape)
        if mode == self.TIMES:
            res[0] = x[0]
            res[1:] = x[1]*self._logk_lengths + x[2]
        else:
            res[0] = x[0]
            res[1] = np.vdot(self._logk_lengths, x[1:])
            res[2] = np.sum(x[1:])
        return ift.from_global_data(self._tgt(mode), res)


def LogIntegratedWienerProcess(target, means, stddevs, keys):
    # means and stddevs: zm, slope, yintercept, wienersigma
    # keys: rest smooth wienersigma
    if not (len(means) == 4 and len(stddevs) == 4 and len(keys) == 3):
        raise ValueError
    means = np.array(means)
    stddevs = np.array(stddevs)
    # FIXME More checks
    rest = _Rest(target)
    restmeans = ift.from_global_data(rest.domain, means[:-1])
    reststddevs = ift.from_global_data(rest.domain, stddevs[:-1])
    rest = rest @ ift.Adder(restmeans) @ ift.makeOp(reststddevs)

    expander = ift.VdotOperator(ift.full(target, 1.)).adjoint
    sigma = ift.Adder(ift.full(expander.domain, means[3])) @ (
        stddevs[3]*ift.ducktape(expander.domain, None, keys[2]))
    sigma = expander @ sigma.exp()
    smooth = _TwoLogIntegrations(target).ducktape(keys[1])*sigma
    return rest.ducktape(keys[0]) + smooth


if __name__ == '__main__':
    np.random.seed(42)
    ndim = 2
    sspace = ift.RGSpace((512, 512))
    hspace = sspace.get_default_codomain()
    target = ift.PowerSpace(hspace,
                            ift.PowerSpace.useful_binbounds(hspace, True))
    test0 = _Rest(target)
    test1 = _TwoLogIntegrations(target)
    ift.extra.consistency_check(test0)
    ift.extra.consistency_check(test1)
    op = LogIntegratedWienerProcess(target,
                                    [0, -4, 1, 0],
                                    [1, 1, 1, 0.5],
                                    ['rest', 'smooth', 'wienersigma']).exp()
    fld = ift.from_random('normal', op.domain)
    ift.extra.check_jacobian_consistency(op, fld)
    plts = []
    for _ in range(50):
        fld = ift.from_random('normal', op.domain)
        plts.append(op(fld))
    ift.single_plot(plts, name='debug.png')
