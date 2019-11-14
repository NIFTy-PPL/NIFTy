# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2019 Max-Planck-Society
# Authors: Philipp Frank, Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.power_space import PowerSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..operators.adder import Adder
from ..operators.contraction_operator import ContractionOperator
from ..operators.distributors import PowerDistributor
from ..operators.endomorphic_operator import EndomorphicOperator
from ..operators.harmonic_operators import HarmonicTransformOperator
from ..operators.linear_operator import LinearOperator
from ..operators.operator import Operator
from ..operators.simple_linear_operators import VdotOperator, ducktape
from ..operators.value_inserter import ValueInserter
from ..probing import StatCalculator
from ..sugar import from_global_data, full, makeDomain


def _lognormal_moments(mean, sig):
    mean, sig = float(mean), float(sig)
    assert sig > 0
    logsig = np.sqrt(np.log((sig/mean)**2 + 1))
    logmean = np.log(mean) - logsig**2/2
    return logmean, logsig


def _normal(mean, sig, key):
    return Adder(Field.scalar(mean)) @ (
        sig*ducktape(DomainTuple.scalar_domain(), None, key))


def _log_k_lengths(pspace):
    """Log(k_lengths) without zeromode"""
    return np.log(pspace.k_lengths[1:])


def _relative_log_k_lengths(power_space):
    """Log-distance to first bin
    logkl.shape==power_space.shape, logkl[0]=logkl[1]=0"""
    power_space = DomainTuple.make(power_space)
    assert isinstance(power_space[0], PowerSpace)
    assert len(power_space) == 1
    logkl = _log_k_lengths(power_space[0])
    assert logkl.shape[0] == power_space[0].shape[0] - 1
    logkl -= logkl[0]
    return np.insert(logkl, 0, 0)


def _log_vol(power_space):
    power_space = DomainTuple.make(power_space)
    assert isinstance(power_space[0], PowerSpace)
    assert len(power_space) == 1
    logk_lengths = _log_k_lengths(power_space[0])
    return logk_lengths[1:] - logk_lengths[:-1]


def _total_fluctuation_realized(samples):
    res = 0.
    for s in samples:
        res = res + (s - s.mean())**2
    return np.sqrt((res/len(samples)).mean())


def _stats(op, samples):
    sc = StatCalculator()
    for s in samples:
        sc.add(op(s.extract(op.domain)))
    return sc.mean.to_global_data(), sc.var.sqrt().to_global_data()


class _LognormalMomentMatching(Operator):
    def __init__(self, mean, sig, key):
        key = str(key)
        logmean, logsig = _lognormal_moments(mean, sig)
        self._mean = mean
        self._sig = sig
        op = _normal(logmean, logsig, key).exp()
        self._domain, self._target = op.domain, op.target
        self.apply = op.apply

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._sig


class _SlopeRemover(EndomorphicOperator):
    def __init__(self, domain):
        self._domain = makeDomain(domain)
        assert len(self._domain) == 1
        assert isinstance(self._domain[0], PowerSpace)
        logkl = _relative_log_k_lengths(self._domain)
        self._sc = logkl/float(logkl[-1])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = x - x[-1]*self._sc
        else:
            res = np.zeros(x.shape, dtype=x.dtype)
            res += x
            res[-1] -= (x*self._sc).sum()
        return from_global_data(self._tgt(mode), res)


class _TwoLogIntegrations(LinearOperator):
    def __init__(self, target):
        self._target = makeDomain(target)
        assert len(self._target) == 1
        assert isinstance(self._target[0], PowerSpace)
        self._domain = makeDomain(
            UnstructuredDomain((2, self.target.shape[0] - 2)))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        if not isinstance(self._target[0], PowerSpace):
            raise TypeError
        self._log_vol = _log_vol(self._target[0])

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            x = x.to_global_data()
            res = np.empty(self._target.shape)
            res[0] = res[1] = 0
            res[2:] = np.cumsum(x[1])
            res[2:] = (res[2:] + res[1:-1])/2*self._log_vol + x[0]
            res[2:] = np.cumsum(res[2:])
            return from_global_data(self._target, res)
        else:
            x = x.to_global_data_rw()
            res = np.zeros(self._domain.shape)
            x[2:] = np.cumsum(x[2:][::-1])[::-1]
            res[0] += x[2:]
            x[2:] *= self._log_vol/2.
            x[1:-1] += x[2:]
            res[1] += np.cumsum(x[2:][::-1])[::-1]
            return from_global_data(self._domain, res)


class _Normalization(Operator):
    def __init__(self, domain):
        self._domain = self._target = makeDomain(domain)
        assert len(self._domain) == 1
        assert isinstance(self._domain[0], PowerSpace)
        hspace = self._domain[0].harmonic_partner
        pd = PowerDistributor(hspace, power_space=self._domain[0])
        cst = pd.adjoint(full(pd.target, 1.)).to_global_data_rw()
        cst[0] = 0
        self._cst = from_global_data(self._domain, cst)
        self._specsum = _SpecialSum(self._domain)

    def apply(self, x):
        self._check_input(x)
        amp = x.exp()
        spec = (2*x).exp()
        # FIXME This normalizes also the zeromode which is supposed to be left
        # untouched by this operator
        return self._specsum(self._cst*spec)**(-0.5)*amp


class _SpecialSum(EndomorphicOperator):
    def __init__(self, domain):
        self._domain = makeDomain(domain)
        assert len(self._domain) == 1
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return full(self._tgt(mode), x.sum())


class _Amplitude(Operator):
    def __init__(self, target, fluctuations, flexibility, asperity,
                 loglogavgslope, key):
        """
        fluctuations > 0
        flexibility > 0
        asperity > 0
        loglogavgslope probably negative
        """
        assert isinstance(fluctuations, Operator)
        assert isinstance(flexibility, Operator)
        assert isinstance(asperity, Operator)
        assert isinstance(loglogavgslope, Operator)
        target = makeDomain(target)
        assert len(target) == 1
        assert isinstance(target[0], PowerSpace)

        twolog = _TwoLogIntegrations(target)
        dom = twolog.domain
        shp = dom.shape
        totvol = target[0].harmonic_partner.get_default_codomain().total_volume

        # Prepare constant fields
        foo = np.zeros(shp)
        foo[0] = foo[1] = np.sqrt(_log_vol(target))
        vflex = from_global_data(dom, foo)

        foo = np.zeros(shp, dtype=np.float64)
        foo[0] += 1
        vasp = from_global_data(dom, foo)

        foo = np.ones(shp)
        foo[0] = _log_vol(target)**2/12.
        shift = from_global_data(dom, foo)

        vslope = from_global_data(target, _relative_log_k_lengths(target))

        foo, bar = [np.zeros(target.shape) for _ in range(2)]
        bar[1:] = foo[0] = totvol
        vol0, vol1 = [from_global_data(target, aa) for aa in (foo, bar)]
        # End prepare constant fields

        slope = VdotOperator(vslope).adjoint @ loglogavgslope
        sig_flex = VdotOperator(vflex).adjoint @ flexibility
        sig_asp = VdotOperator(vasp).adjoint @ asperity
        sig_fluc = VdotOperator(vol1).adjoint @ fluctuations

        xi = ducktape(dom, None, key)
        sigma = sig_flex*(Adder(shift) @ sig_asp).sqrt()
        smooth = _SlopeRemover(target) @ twolog @ (sigma*xi)
        op = _Normalization(target) @ (slope + smooth)
        op = Adder(vol0) @ (sig_fluc*op)

        self.apply = op.apply
        self._fluc = fluctuations
        self._domain, self._target = op.domain, op.target

    @property
    def fluctuation_amplitude(self):
        return self._fluc


class CorrelatedFieldMaker:
    def __init__(self, amplitude_offset, prefix):
        self._a = []
        self._position_spaces = []
        
        self._azm = amplitude_offset
        self._prefix = prefix
    
    @staticmethod
    def make(offset_amplitude_mean, offset_amplitude_stddev, prefix):
        offset_amplitude_stddev = float(offset_amplitude_stddev)
        offset_amplitude_mean = float(offset_amplitude_mean)
        assert offset_amplitude_stddev > 0
        assert offset_amplitude_mean > 0
        zm = _LognormalMomentMatching(offset_amplitude_mean,
                                      offset_amplitude_stddev,
                                      prefix + 'zeromode')
        return CorrelatedFieldMaker(zm, prefix)

    def add_fluctuations(self,
                         position_space,
                         fluctuations_mean,
                         fluctuations_stddev,
                         flexibility_mean,
                         flexibility_stddev,
                         asperity_mean,
                         asperity_stddev,
                         loglogavgslope_mean,
                         loglogavgslope_stddev,
                         prefix='',
                         index=None):
        fluctuations_mean = float(fluctuations_mean)
        fluctuations_stddev = float(fluctuations_stddev)
        flexibility_mean = float(flexibility_mean)
        flexibility_stddev = float(flexibility_stddev)
        asperity_mean = float(asperity_mean)
        asperity_stddev = float(asperity_stddev)
        loglogavgslope_mean = float(loglogavgslope_mean)
        loglogavgslope_stddev = float(loglogavgslope_stddev)
        prefix = str(prefix)
        assert fluctuations_stddev > 0
        assert fluctuations_mean > 0
        assert flexibility_stddev > 0
        assert flexibility_mean > 0
        assert asperity_stddev > 0
        assert asperity_mean > 0
        assert loglogavgslope_stddev > 0

        fluct = _LognormalMomentMatching(fluctuations_mean,
                                         fluctuations_stddev,
                                         prefix + 'fluctuations')
        fluct = fluct*self._azm.one_over()
        flex = _LognormalMomentMatching(flexibility_mean, flexibility_stddev,
                                        prefix + 'flexibility')
        asp = _LognormalMomentMatching(asperity_mean, asperity_stddev,
                                       prefix + 'asperity')
        avgsl = _normal(loglogavgslope_mean, loglogavgslope_stddev,
                        prefix + 'loglogavgslope')
        amp = _Amplitude(PowerSpace(position_space.get_default_codomain()),
                         fluct, flex, asp, avgsl, prefix + 'spectrum')
        if index is not None:
            self._a.insert(index, amp)
            self._position_spaces.insert(index, position_space)
        else:
            self._a.append(amp)
            self._position_spaces.append(position_space)

    def finalize_from_op(self, zeromode, prefix=''):
        assert isinstance(zeromode, Operator)
        hspace = makeDomain([dd.get_default_codomain()
                             for dd in self._position_spaces])
        foo = np.ones(hspace.shape)
        zeroind = len(hspace.shape)*(0,)
        foo[zeroind] = 0
        azm = VdotOperator(full(hspace,1.)).adjoint @ zeromode

        n_amplitudes = len(self._a)
        ht = HarmonicTransformOperator(hspace, self._position_spaces[0],
                                       space=0)
        for i in range(1, n_amplitudes):
            ht = (HarmonicTransformOperator(ht.target,
                                            self._position_spaces[i],
                                            space=i) @ ht)

        pd = PowerDistributor(hspace, self._a[0].target[0], 0)
        for i in range(1, n_amplitudes):
            foo = PowerDistributor(pd.domain, self._a[i].target[0], space=i)
            pd = pd @ foo

        spaces = tuple(range(n_amplitudes))
        a = ContractionOperator(pd.domain, spaces[1:]).adjoint @ self._a[0]
        for i in range(1, n_amplitudes):
            co = ContractionOperator(pd.domain, spaces[:i] + spaces[(i + 1):])
            a = a*(co.adjoint @ self._a[i])

        return ht(azm*(pd @ a)*ducktape(hspace, None, prefix + 'xi'))

    def finalize(self,
                 offset=None,
                 prior_info=100):
        """
        offset vs zeromode: volume factor
        """
        if offset is not None:
            raise NotImplementedError
            offset = float(offset)
        
        op = self.finalize_from_op(self._azm, self._prefix)
        if prior_info > 0:
            from ..sugar import from_random
            samps = [
                from_random('normal', op.domain) for _ in range(prior_info)
            ]
            self.statistics_summary(samps)
        return op

    def statistics_summary(self, samples):
        lst = [('Offset amplitude', self.amplitude_total_offset),
               ('Total fluctuation amplitude', self.total_fluctuation)]

        namps = len(self.amplitudes)
        if namps > 1:
            for ii in range(namps):
                lst.append(('Slice fluctuation (space {})'.format(ii),
                            self.slice_fluctuation(ii)))
                lst.append(('Average fluctuation (space {})'.format(ii),
                            self.average_fluctuation(ii)))

        for kk, op in lst:
            mean, stddev = _stats(op, samples)
            print('{}: {:.02E} ± {:.02E}'.format(kk, mean, stddev))

    def moment_slice_to_average(self, fluctuations_slice_mean, nsamples=1000):
        fluctuations_slice_mean = float(fluctuations_slice_mean)
        assert fluctuations_slice_mean > 0
        from ..sugar import from_random
        scm = 1.
        for a in self._a:
            op = a.fluctuation_amplitude
            res= np.array([op(from_random('normal',op.domain)).to_global_data()
                            for _ in range(nsamples)])
            scm *= res**2 + 1.
        return fluctuations_slice_mean/np.mean(np.sqrt(scm))

    @property
    def amplitudes(self):
        return self._a

    @property
    def amplitude_total_offset(self):
        return self._azm

    @property
    def total_fluctuation(self):
        """Returns operator which acts on prior or posterior samples"""
        if len(self._a) == 0:
            raise NotImplementedError
        if len(self._a) == 1:
            return self.average_fluctuation(0)
        q = 1.
        for a in self._a:
            fl = a.fluctuation_amplitude
            q = q*(Adder(full(fl.target, 1.)) @ fl**2)
        return (Adder(full(q.target, -1.)) @ q).sqrt() * self._azm

    def slice_fluctuation(self, space):
        """Returns operator which acts on prior or posterior samples"""
        if len(self._a) == 0:
            raise NotImplementedError
        assert space < len(self._a)
        if len(self._a) == 1:
            return self.average_fluctuation(0)
        q = 1.
        for j in range(len(self._a)):
            fl = self._a[j].fluctuation_amplitude
            if j == space:
                q = q*fl**2
            else:
                q = q*(Adder(full(fl.target, 1.)) @ fl**2)
        return q.sqrt() * self._azm

    def average_fluctuation(self, space):
        """Returns operator which acts on prior or posterior samples"""
        if len(self._a) == 0:
            raise NotImplementedError
        assert space < len(self._a)
        if len(self._a) == 1:
            return self._a[0].fluctuation_amplitude*self._azm
        return self._a[space].fluctuation_amplitude*self._azm

    @staticmethod
    def offset_amplitude_realized(samples):
        res = 0.
        for s in samples:
            res = res + s.mean()**2
        return np.sqrt(res/len(samples))

    @staticmethod
    def total_fluctuation_realized(samples):
        return _total_fluctuation_realized(samples)

    @staticmethod
    def slice_fluctuation_realized(samples, space):
        """Computes slice fluctuations from collection of field (defined in signal
        space) realizations."""
        ldom = len(samples[0].domain)
        assert space < ldom
        if ldom == 1:
            return _total_fluctuation_realized(samples)
        res1, res2 = 0., 0.
        for s in samples:
            res1 = res1 + s**2
            res2 = res2 + s.mean(space)**2
        res1 = res1/len(samples)
        res2 = res2/len(samples)
        res = res1.mean() - res2.mean()
        return np.sqrt(res)


    @staticmethod
    def average_fluctuation_realized(samples, space):
        """Computes average fluctuations from collection of field (defined in signal
        space) realizations."""
        ldom = len(samples[0].domain)
        assert space < ldom
        if ldom == 1:
            return _total_fluctuation_realized(samples)
        spaces = ()
        for i in range(ldom):
            if i != space:
                spaces += (i,)
        res = 0.
        for s in samples:
            r = s.mean(spaces)
            res = res + (r - r.mean())**2
        res = res/len(samples)
        return np.sqrt(res.mean())
