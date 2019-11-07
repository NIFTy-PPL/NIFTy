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
from ..sugar import from_global_data, full, makeDomain


def _lognormal_moments(mean, sig):
    mean, sig = float(mean), float(sig)
    assert sig > 0
    logsig = np.sqrt(np.log((sig/mean)**2 + 1))
    logmean = np.log(mean) - logsig**2/2
    return logmean, logsig


def _lognormal_moment_matching(mean, sig, key):
    key = str(key)
    logmean, logsig = _lognormal_moments(mean, sig)
    return _normal(logmean, logsig, key).exp()


def _normal(mean, sig, key):
    return Adder(Field.scalar(mean)) @ (
        sig*ducktape(DomainTuple.scalar_domain(), None, key))


class _SlopeRemover(EndomorphicOperator):
    def __init__(self, domain, logkl):
        self._domain = makeDomain(domain)
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


def _log_k_lengths(pspace):
    return np.log(pspace.k_lengths[1:])


class _TwoLogIntegrations(LinearOperator):
    def __init__(self, target):
        self._target = makeDomain(target)
        self._domain = makeDomain(
            UnstructuredDomain((2, self.target.shape[0] - 2)))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        if not isinstance(self._target[0], PowerSpace):
            raise TypeError
        logk_lengths = _log_k_lengths(self._target[0])
        self._logvol = logk_lengths[1:] - logk_lengths[:-1]

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            x = x.to_global_data()
            res = np.empty(self._target.shape)
            res[0] = 0
            res[1] = 0
            res[2:] = np.cumsum(x[1])
            res[2:] = (res[2:] + res[1:-1])/2*self._logvol + x[0]
            res[2:] = np.cumsum(res[2:])
            return from_global_data(self._target, res)
        else:
            x = x.to_global_data_rw()
            res = np.zeros(self._domain.shape)
            x[2:] = np.cumsum(x[2:][::-1])[::-1]
            res[0] += x[2:]
            x[2:] *= self._logvol/2.
            x[1:-1] += x[2:]
            res[1] += np.cumsum(x[2:][::-1])[::-1]
            return from_global_data(self._domain, res)


class _Normalization(Operator):
    def __init__(self, domain):
        self._domain = self._target = makeDomain(domain)
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
        dt = twolog._logvol
        sc = np.zeros(twolog.domain.shape)
        sc[0] = sc[1] = np.sqrt(dt)
        sc = from_global_data(twolog.domain, sc)
        expander = VdotOperator(sc).adjoint
        sigmasq = expander @ flexibility

        dist = np.zeros(twolog.domain.shape)
        dist[0] += 1.
        dist = from_global_data(twolog.domain, dist)
        scale = VdotOperator(dist).adjoint @ asperity

        shift = np.ones(scale.target.shape)
        shift[0] = dt**2/12.
        shift = from_global_data(scale.target, shift)
        scale = sigmasq*(Adder(shift) @ scale).sqrt()

        smooth = twolog @ (scale*ducktape(scale.target, None, key))
        tg = smooth.target
        logkl = _log_k_lengths(tg[0])
        assert logkl.shape[0] == tg[0].shape[0] - 1
        logkl -= logkl[0]
        logkl = np.insert(logkl, 0, 0)
        noslope = _SlopeRemover(tg, logkl) @ smooth
        _t = VdotOperator(from_global_data(tg, logkl)).adjoint
        smoothslope = _t @ loglogavgslope + noslope

        normal_ampl = _Normalization(target) @ smoothslope
        vol = target[0].harmonic_partner.get_default_codomain().total_volume
        arr = np.zeros(target.shape)
        arr[1:] = vol
        expander = VdotOperator(from_global_data(target, arr)).adjoint
        mask = np.zeros(target.shape)
        mask[0] = vol
        adder = Adder(from_global_data(target, mask))
        self._op = adder @ ((expander @ fluctuations)*normal_ampl)

        self._domain = self._op.domain
        self._target = self._op.target

    def apply(self, x):
        self._check_input(x)
        return self._op(x)


class CorrelatedFieldMaker:
    def __init__(self):
        self._a = []

    def add_fluctuations(self,
                         target,
                         fluctuations_mean,
                         fluctuations_stddev,
                         flexibility_mean,
                         flexibility_stddev,
                         asperity_mean,
                         asperity_stddev,
                         loglogavgslope_mean,
                         loglogavgslope_stddev,
                         prefix=''):
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

        fluct = _lognormal_moment_matching(fluctuations_mean,
                                           fluctuations_stddev,
                                           prefix + 'fluctuations')
        flex = _lognormal_moment_matching(flexibility_mean, flexibility_stddev,
                                          prefix + 'flexibility')
        asp = _lognormal_moment_matching(asperity_mean, asperity_stddev,
                                         prefix + 'asperity')
        avgsl = _normal(loglogavgslope_mean, loglogavgslope_stddev,
                        prefix + 'loglogavgslope')
        self._a.append(
            _Amplitude(target, fluct, flex, asp, avgsl, prefix + 'spectrum'))

    def finalize_from_op(self, zeromode, prefix=''):
        assert isinstance(zeromode, Operator)
        hspace = makeDomain([dd.target[0].harmonic_partner for dd in self._a])
        foo = np.ones(hspace.shape)
        zeroind = len(hspace.shape)*(0,)
        foo[zeroind] = 0
        azm = Adder(from_global_data(hspace, foo)) @ ValueInserter(
            hspace, zeroind) @ zeromode

        n_amplitudes = len(self._a)
        ht = HarmonicTransformOperator(hspace, space=0)
        for i in range(1, n_amplitudes):
            ht = HarmonicTransformOperator(ht.target, space=i) @ ht

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
                 offset_amplitude_mean,
                 offset_amplitude_stddev,
                 prefix='',
                 offset=None):
        """
        offset vs zeromode: volume factor
        """
        offset_amplitude_stddev = float(offset_amplitude_stddev)
        offset_amplitude_mean = float(offset_amplitude_mean)
        assert offset_amplitude_stddev > 0
        assert offset_amplitude_mean > 0
        if offset is not None:
            raise NotImplementedError
            offset = float(offset)
        azm = _lognormal_moment_matching(offset_amplitude_mean,
                                         offset_amplitude_stddev,
                                         prefix + 'zeromode')
        return self.finalize_from_op(azm, prefix)

    @property
    def amplitudes(self):
        return self._a

    def effective_total_fluctuation(self,
                                    fluctuations_means,
                                    fluctuations_stddevs,
                                    nsamples=100):
        namps = len(fluctuations_means)
        xis = np.random.normal(size=namps*nsamples).reshape((namps, nsamples))
        q = np.ones(nsamples)
        for i in range(len(fluctuations_means)):
            m, sig = _lognormal_moments(fluctuations_means[i],
                                        fluctuations_stddevs[i])
            f = np.exp(m + sig*xis[i])
            q *= (1. + f**2)
        q = np.sqrt(q - 1.)
        return np.mean(q), np.std(q)
