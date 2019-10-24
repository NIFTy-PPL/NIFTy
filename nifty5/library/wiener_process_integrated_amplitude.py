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


def LogIntegratedWienerProcess(target, means, stddevs, wienersigmastddev,
                               wienersigmaprob, keys):
    # means and stddevs: zm, slope, yintercept
    # keys: rest smooth wienersigma
    if not (len(means) == 3 and len(stddevs) == 3 and len(keys) == 3):
        raise ValueError
    means = np.array(means)
    stddevs = np.array(stddevs)
    # FIXME More checks
    rest = _Rest(target)
    restmeans = ift.from_global_data(rest.domain, means)
    reststddevs = ift.from_global_data(rest.domain, stddevs)
    rest = rest @ ift.Adder(restmeans) @ ift.makeOp(reststddevs)

    expander = ift.VdotOperator(ift.full(target, 1.)).adjoint
    m = means[1]
    L = np.log(target.k_lengths[-1]) - np.log(target.k_lengths[1])

    s = np.log(np.abs(m/L))
    from scipy.special import erfinv
    wienermean = s - erfinv(wienersigmaprob)*wienersigmastddev
    print(s, wienermean, wienersigmastddev)
    sigma = ift.Adder(ift.full(expander.domain, wienermean)) @ (
        wienersigmastddev*ift.ducktape(expander.domain, None, keys[2]))
    sigma = expander @ sigma.exp()
    smooth = _TwoLogIntegrations(target).ducktape(keys[1])*sigma
    return rest.ducktape(keys[0]) + smooth


class Normalization(ift.Operator):
    def __init__(self, domain):
        self._domain = self._target = ift.makeDomain(domain)
        hspace = self._domain[0].harmonic_partner
        pd = ift.PowerDistributor(hspace, power_space=self._domain[0])
        # TODO Does not work on sphere yet
        self._cst = pd.adjoint(ift.full(pd.target, hspace.scalar_dvol))
        self._specsum = SpecialSum(self._domain)

    def apply(self, x):
        self._check_input(x)
        return self._specsum(self._cst*x).one_over()*x


class SpecialSum(ift.EndomorphicOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return ift.full(self._tgt(mode), x.sum())


def Amplitude(target, means, stddevs, wienersigmastddev, wienersigmaprob,
              keys):
    op = LogIntegratedWienerProcess(target, means, stddevs, wienersigmastddev,
                                    wienersigmaprob, keys)
    return Normalization(target) @ op.exp()


def mfplot(fld, A1, A2, name):
    p = ift.Plot()
    dom = fld.domain
    mi, ma = np.min(fld.val), np.max(fld.val)
    for ii in range(dom.shape[-1]):
        extr = ift.DomainTupleFieldInserter(op.target, 1, (ii,)).adjoint
        p.add(extr(fld), zmin=mi, zmax=ma)
    p.add(A1)
    p.add(A2)
    p.add(fld)
    p.output(name=name, xsize=14, ysize=14)


if __name__ == '__main__':
    np.random.seed(42)
    from correlated_fields_normalized_amplitude import CorrelatedFieldNormAmplitude
    sspace = ift.RGSpace((512, 512))
    hspace = sspace.get_default_codomain()
    target = ift.PowerSpace(hspace,
                            ift.PowerSpace.useful_binbounds(hspace, True))
    test0 = _Rest(target)
    test1 = _TwoLogIntegrations(target)
    ift.extra.consistency_check(test0)
    ift.extra.consistency_check(test1)
    # A = Amplitude(target, [5, -3, 1, 0], [1, 1, 1, 0.5],
    #               ['rest', 'smooth', 'wienersigma'])
    # fld = ift.from_random('normal', A.domain)
    # ift.extra.check_jacobian_consistency(A, fld)
    # op = CorrelatedFieldNormAmplitude(sspace, A, 3, 2)
    # plts = []
    # p = ift.Plot()
    # for _ in range(25):
    #     fld = ift.from_random('normal', op.domain)
    #     plts.append(A.force(fld))
    #     p.add(op(fld))
    # p.output(name='debug1.png', xsize=20, ysize=20)
    # ift.single_plot(plts, name='debug.png')

    # Multifrequency
    sspace1 = ift.RGSpace((512, 512))
    hspace1 = sspace1.get_default_codomain()
    target1 = ift.PowerSpace(hspace1,
                             ift.PowerSpace.useful_binbounds(hspace1, True))
    A1 = Amplitude(target1, [0, -2, 1], [1, 1, 1], 1, 0.99,
                   ['rest1', 'smooth1', 'wienersigma1'])
    sspace2 = ift.RGSpace((20,), distances=(1e7))
    hspace2 = sspace2.get_default_codomain()
    target2 = ift.PowerSpace(hspace2,
                             ift.PowerSpace.useful_binbounds(hspace2, True))
    A2 = Amplitude(target2, [0, -2, 1], [1, 1, 1], 1, 0.99,
                   ['rest2', 'smooth2', 'wienersigma2'])
    op = CorrelatedFieldNormAmplitude((sspace1, sspace2), (A1, A2), 3, 2)
    for jj in range(10):
        fld = ift.from_random('normal', op.domain)
        mfplot(op(fld), A1.force(fld), A2.force(fld), 'debug{}.png'.format(jj))
