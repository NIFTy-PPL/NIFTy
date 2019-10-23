import nifty5 as ift
import numpy as np


class SymmetrizingOperator(ift.EndomorphicOperator):
    def __init__(self, domain):
        self._domain = ift.DomainTuple.make(domain)
        if len(self._domain.shape) != 1:
            raise TypeError
        if not isinstance(self._domain[0], ift.RGSpace):
            raise TypeError
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = x.to_global_data_rw()
        res -= res[::-1]
        res /= 2
        return ift.from_global_data(self._tgt(mode), res)


def CepstrumOperator(target, a, k0):
    a = float(a)
    target = ift.DomainTuple.make(target)
    if a <= 0:
        raise ValueError
    if len(target) > 1 or target[0].harmonic:
        raise TypeError
    if isinstance(k0, (float, int)):
        k0 = np.array([k0]*len(target.shape))
    else:
        k0 = np.array(k0)
    if len(k0) != len(target.shape):
        raise ValueError
    if np.any(np.array(k0) <= 0):
        raise ValueError
    if not isinstance(target[0], ift.RGSpace):
        raise TypeError

    dom = target[0].get_default_codomain()
    qht = ift.HartleyOperator(dom, target=target[0])
    sym = SymmetrizingOperator(target)

    pspace = ift.PowerSpace(dom)
    ceps_field = ift.PS_field(pspace, lambda k: a/(1 + (k/k0)**2))
    ceps_field = ift.PowerDistributor(dom, power_space=pspace)(ceps_field)

    return sym @ qht @ ift.makeOp(ceps_field)


class SlopeOperator(ift.LinearOperator):
    def __init__(self, target):
        self._target = ift.DomainTuple.make(target)
        if len(self._target) > 1:
            raise TypeError
        if len(self._target[0].shape) > 1:
            raise TypeError
        if not isinstance(self._target[0], ift.RGSpace):
            raise TypeError
        self._domain = ift.DomainTuple.make(ift.UnstructuredDomain((2,)))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._pos = np.fft.fftfreq(
            self._target.size)*self._target.size*self._target[0].distances[0]

    def apply(self, x, mode):
        self._check_input(x, mode)
        inp = x.to_global_data()
        if mode == self.TIMES:
            res = inp[1] + inp[0]*self._pos
        else:
            res = np.array([np.sum(self._pos*inp), np.sum(inp)], dtype=x.dtype)
        return ift.Field.from_global_data(self._tgt(mode), res)


class ExpTransform(ift.LinearOperator):
    def __init__(self, domain, target):
        self._target = ift.makeDomain(target)
        self._domain = ift.makeDomain(domain)
        if not isinstance(self._domain, ift.MultiDomain):
            raise TypeError
        if set(self._domain.keys()) != set(['zeromode', 'nonzeromodes']):
            raise ValueError
        # Zeromode domain is Unstr(1)
        # Nonzeromode domain is RGSpace nonharmonic, 1D, even number of pixs

        self._capability = self.TIMES | self.ADJOINT_TIMES
        tgt = self._target[0]
        if not isinstance(tgt, ift.PowerSpace):
            raise ValueError("Target must be a power space.")

        log_k_array = np.log(tgt.k_lengths[1:])
        t_min = np.amin(log_k_array)
        bindistances = self._domain['nonzeromodes'][0].distances[0]

        coord = (log_k_array - t_min)/bindistances
        self._bindex = np.floor(coord).astype(int)
        self._frac = coord - self._bindex

    def apply(self, x, mode):
        self._check_input(x, mode)
        wgt = self._frac
        if mode == self.ADJOINT_TIMES:
            x = x.to_global_data()
            xnew = {}
            xnew['zeromode'] = ift.full(self._domain['zeromode'], x[0])
            foo = np.zeros(self._domain['nonzeromodes'].shape[0])
            np.add.at(foo, self._bindex, x[1:]*(1. - wgt))
            np.add.at(foo, self._bindex + 1, x[1:]*wgt)
            xnew['nonzeromodes'] = ift.from_global_data(
                self._domain['nonzeromodes'], foo)
            return ift.MultiField.from_dict(xnew, self._domain)
        else:  # TIMES
            xnew = np.empty(self._target.shape)
            zm = x['zeromode'].to_global_data()
            nzm = x['nonzeromodes'].to_global_data()
            xnew[1:] = nzm[self._bindex]*(1. - wgt)
            xnew[1:] += nzm[self._bindex + 1]*wgt
            xnew[0] = zm
            return ift.from_global_data(self._target, xnew)


def NormalizedAmplitude(target,
                        n_pix,
                        a,
                        k0,
                        sm,
                        sv,
                        im,
                        iv,
                        zmmean,
                        zmsig,
                        keys=['tau', 'phi', 'zeromode']):
    if not (isinstance(n_pix, int) and isinstance(target, ift.PowerSpace)):
        raise TypeError
    a, k0 = float(a), float(k0)
    sm, sv, im, iv = float(sm), float(sv), float(im), float(iv)
    zmmean, zmsig = float(zmmean), float(zmsig)
    if sv <= 0 or iv <= 0 or zmsig <= 0:
        raise ValueError
    dct = {'a': a, 'k0': k0}
    karray = target.k_lengths
    logkarray = np.log(karray[1:])
    t_0 = np.amin(logkarray)
    dst = (np.amax(logkarray) - t_0)/(n_pix - 1)
    dom = ift.RGSpace(2*n_pix, distances=dst)
    smooth = CepstrumOperator(dom, **dct).ducktape(keys[0])
    sl = SlopeOperator(dom)
    mean = np.array([sm, im + sm*t_0])
    sig = np.array([sv, iv])
    mean = ift.Field.from_global_data(sl.domain, mean)
    sig = ift.Field.from_global_data(sl.domain, sig)
    linear = sl @ ift.Adder(mean) @ ift.makeOp(sig).ducktape(keys[1])
    nonzeromodes = 0.5*(smooth + linear)
    ust = ift.UnstructuredDomain(1)
    zmmean = ift.full(ust, zmmean)
    zmsig = ift.full(ust, zmsig)
    zeromode = ift.Adder(zmmean) @ ift.makeOp(zmsig) @ ift.ducktape(
        ust, None, keys[2])
    allmodes = zeromode.ducktape_left('zeromode') + nonzeromodes.ducktape_left(
        'nonzeromodes')
    et = ExpTransform(allmodes.target, target)
    ift.extra.consistency_check(et)
    ift.extra.consistency_check(sl)
    ift.extra.consistency_check(smooth)
    return Normalization(et.target) @ (et @ allmodes).exp()


class Normalization(ift.Operator):
    def __init__(self, domain):
        self._domain = self._target = ift.makeDomain(domain)
        hspace = self._domain[0].harmonic_partner
        pd = ift.PowerDistributor(hspace, power_space=self._domain[0])
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


if __name__ == '__main__':
    np.random.seed(42)
    ndim = 1
    sspace = ift.RGSpace(
        np.linspace(16, 20, num=ndim).astype(np.int),
        np.linspace(2.3, 7.99, num=ndim))
    hspace = sspace.get_default_codomain()
    target = ift.PowerSpace(hspace)
    op = NormalizedAmplitude(target, 16, 1, 1, -3, 1, 0, 1, 0, 1)
    if not isinstance(op.target[0], ift.PowerSpace):
        raise RuntimeError
    fld = ift.from_random('normal', op.domain)
    ift.extra.check_jacobian_consistency(op, fld)

    pd = ift.PowerDistributor(hspace, power_space=target)
    ht = ift.HartleyOperator(pd.target)
    if np.testing.assert_allclose((pd @ op)(fld).integrate(), 1):
        raise RuntimeError
    if np.testing.assert_allclose(
        (ht @ pd @ op)(fld).to_global_data()[ndim*(0,)], 1):
        raise RuntimeError
