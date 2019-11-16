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
from ..operators.diagonal_operator import DiagonalOperator
from ..operators.operator import Operator
from ..operators.simple_linear_operators import VdotOperator, ducktape
from ..operators.value_inserter import ValueInserter
from ..probing import StatCalculator
from ..sugar import from_global_data, from_random, full, makeDomain, get_default_codomain


def _reshaper(x, N):
    x = np.asfarray(x)
    if x.shape in [(), (1,)]:
        return np.full(N, x) if N != 1 else x.reshape(())
    elif x.shape == (N,):
        return x
    else:
        raise TypeError("Shape of parameters cannot be interpreted")


def _lognormal_moments(mean, sig, N = 1):
    mean, sig = (_reshaper(param, N) for param in (mean, sig))
    assert np.all(mean > 0 )
    assert np.all(sig > 0)
    logsig = np.sqrt(np.log((sig/mean)**2 + 1))
    logmean = np.log(mean) - logsig**2/2
    return logmean, logsig


def _normal(mean, sig, key, N = 1):
    if N == 1:
        domain = DomainTuple.scalar_domain()
    else:
        domain = UnstructuredDomain(N)
    mean, sig = (_reshaper(param, N) for param in (mean, sig))
    return Adder(from_global_data(domain, mean)) @ (
        DiagonalOperator(from_global_data(domain,sig))
        @ ducktape(domain, None, key))


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
    power_space = makeDomain(power_space)
    assert isinstance(power_space[0], PowerSpace)
    logk_lengths = _log_k_lengths(power_space[0])
    return logk_lengths[1:] - logk_lengths[:-1]


def _total_fluctuation_realized(samples, space = 0):
    res = 0.
    for s in samples:
        res = res + (s - s.mean(space, keepdims = True))**2
    return np.sqrt((res/len(samples)).mean(space))


def _stats(op, samples):
    sc = StatCalculator()
    for s in samples:
        sc.add(op(s.extract(op.domain)))
    return sc.mean.to_global_data(), sc.var.sqrt().to_global_data()


class _LognormalMomentMatching(Operator):
    def __init__(self, mean, sig, key, N_copies):
        key = str(key)
        logmean, logsig = _lognormal_moments(mean, sig, N_copies)
        self._mean = mean
        self._sig = sig
        op = _normal(logmean, logsig, key, N_copies).exp()
        self._domain, self._target = op.domain, op.target
        self.apply = op.apply

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._sig


class _SlopeRemover(EndomorphicOperator):
    def __init__(self, domain, space = 0):
        self._domain = makeDomain(domain)
        assert isinstance(self._domain[space], PowerSpace)
        logkl = _relative_log_k_lengths(self._domain[space])
        self._sc = logkl/float(logkl[-1])

        self._space = space
        axis = self._domain.axes[space][0]
        self._last = (slice(None),)*axis + (-1,) + (None,)
        self._extender = (None,)*(axis) + (slice(None),) + (None,)*(self._domain.axes[-1][-1]-axis)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = x - x[self._last]*self._sc[self._extender]
        else:
            res = x.copy()
            res[self._last] -= (x*self._sc[self._extender]).sum(
                    axis = self._space, keepdims = True)
        return from_global_data(self._tgt(mode), res)


class _TwoLogIntegrations(LinearOperator):
    def __init__(self, target, space = 0):
        self._target = makeDomain(target)
        assert isinstance(self.target[space], PowerSpace)
        dom = list(self._target)
        dom[space] = UnstructuredDomain((2, self.target[space].shape[0]-2))
        self._domain = makeDomain(dom)
        self._space = space
        logk_lengths = _log_k_lengths(self._target[space])
        self._log_vol = _log_vol(self._target[space])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)

        #Maybe make class properties
        axis = self._target.axes[self._space][0]
        sl = (slice(None),)*axis
        extender_sl = (None,)*axis + (slice(None),) + (None,)*(self._target.axes[-1][-1] - axis)
        first = sl + (0,)
        second = sl + (1,)
        from_third = sl + (slice(2,None),)
        no_border = sl + (slice(1,-1),)
        reverse = sl + (slice(None,None,-1),)

        x = x.to_global_data_rw()
        if mode == self.TIMES:
            res = np.empty(self._target.shape)
            res[first] = res[second] = 0
            res[from_third] = np.cumsum(x[second], axis = axis)
            res[from_third] = (res[from_third] + res[no_border])/2*self._log_vol[extender_sl] + x[first]
            res[from_third] = np.cumsum(res[from_third], axis = axis)
        else:
            res = np.zeros(self._domain.shape)
            x[from_third] = np.cumsum(x[from_third][reverse], axis = axis)[reverse]
            res[first] += x[from_third]
            x[from_third] *= (self._log_vol/2.)[extender_sl]
            x[no_border] += x[from_third]
            res[second] += np.cumsum(x[from_third][reverse], axis = axis)[reverse]
        return from_global_data(self._tgt(mode), res)


class _Normalization(Operator):
    def __init__(self, domain, space = 0):
        self._domain = self._target = makeDomain(domain)
        assert isinstance(self._domain[space], PowerSpace)
        hspace = list(self._domain)
        hspace[space] = hspace[space].harmonic_partner
        hspace = makeDomain(hspace)
        pd = PowerDistributor(hspace, power_space=self._domain[space], space = space)
        # TODO Does not work on sphere yet
        mode_multiplicity = pd.adjoint(full(pd.target, 1.)).to_global_data_rw()
        zero_mode = (slice(None),)*self._domain.axes[space][0] + (0,)
        mode_multiplicity[zero_mode] = 0
        self._mode_multiplicity = from_global_data(self._domain, mode_multiplicity)
        self._specsum = _SpecialSum(self._domain, space)

    def apply(self, x):
        self._check_input(x)
        amp = x.exp()
        spec = (2*x).exp()
        # FIXME This normalizes also the zeromode which is supposed to be left
        # untouched by this operator
        return self._specsum(self._mode_multiplicity*spec)**(-0.5)*amp


class _SpecialSum(EndomorphicOperator):
    def __init__(self, domain, space = 0):
        self._domain = makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._contractor = ContractionOperator(domain, space)

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._contractor.adjoint(self._contractor(x))


class _slice_extractor(LinearOperator):
    #FIXME it should be tested if the the domain and target are consistent with the slice
    def __init__(self, domain, target, sl):
        self._domain = makeDomain(domain)
        self._target = makeDomain(target)
        self._sl = sl
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = x[self._sl]
            res = res.reshape(self._target.shape)
        else:
            res = np.zeros(self._domain.shape)
            res[self._sl] = x
        return from_global_data(self._tgt(mode), res)


class _Distributor(LinearOperator):
    def __init__(self, dofdex, domain, target, space = 0):
        self._dofdex = dofdex

        self._target = makeDomain(target)
        self._domain = makeDomain(domain)
        self._sl = (slice(None),)*space
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = x[self._dofdex]
        else:
            res = np.empty(self._tgt(mode).shape)
            res[self._dofdex] = x
        return from_global_data(self._tgt(mode), res)

    

class _Amplitude(Operator):
    def __init__(self, target, fluctuations, flexibility, asperity,
                 loglogavgslope, azm, key, dofdex):
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

        N_copies = max(dofdex) + 1
        assert N_copies > 0
        if N_copies > 1:
            space = 1
            distributed_tgt = makeDomain((UnstructuredDomain(len(dofdex)), target))
            target = makeDomain((UnstructuredDomain(N_copies), target))
            Distributor = _Distributor(dofdex, target, distributed_tgt, 0)
        else:
            space = 0
            target = makeDomain(target)
        assert isinstance(target[space], PowerSpace)
        
        twolog = _TwoLogIntegrations(target, space)
        dom = twolog.domain
        shp = dom[space].shape
        totvol = target[space].harmonic_partner.get_default_codomain().total_volume
        expander = ContractionOperator(dom, spaces = space).adjoint
        ps_expander = ContractionOperator(twolog.target, spaces = space).adjoint

        # Prepare constant fields
        foo = np.zeros(shp)
        foo[0] = foo[1] = np.sqrt(_log_vol(target[space]))
        vflex = DiagonalOperator(from_global_data(dom[space], foo), dom, space)

        foo = np.zeros(shp, dtype=np.float64)
        foo[0] += 1
        vasp = DiagonalOperator(from_global_data(dom[space], foo), dom, space)

        foo = np.ones(shp)
        foo[0] = _log_vol(target[space])**2/12.
        shift = DiagonalOperator(from_global_data(dom[space], foo), dom, space)
        
        vslope = DiagonalOperator(
                    from_global_data(target[space], _relative_log_k_lengths(target[space])),
                    target, space)

        foo, bar = [np.zeros(target[space].shape) for _ in range(2)]
        bar[1:] = foo[0] = totvol
        vol0, vol1 = [DiagonalOperator(from_global_data(target[space], aa), 
                target, space) for aa in (foo, bar)]

        #Prepare fields for Adder
        #NOTE alternative would be adjoint contraction_operator acting
        #on every space except the specified on
        shift, vol0 = [op(full(op.domain, 1)) for op in (shift, vol0)]
        # End prepare constant fields

        slope = vslope @ ps_expander @ loglogavgslope
        sig_flex = vflex @ expander @ flexibility
        sig_asp = vasp @ expander @ asperity
        sig_fluc = vol1 @ ps_expander @ fluctuations
        sig_fluc = vol1 @ ps_expander @ fluctuations

        xi = ducktape(dom, None, key)
        sigma = sig_flex*(Adder(shift) @ sig_asp).sqrt()
        smooth = _SlopeRemover(target, space) @ twolog @ (sigma*xi)
        op = _Normalization(target, space) @ (slope + smooth)
        if space == 1:
            op = Distributor @ op
            sig_fluc = Distributor @ sig_fluc
            op = (Distributor @ Adder(vol0)) @ (sig_fluc*(ps_expander @ azm.one_over())*op)
        else:
            op = (Adder(vol0)) @ (sig_fluc*(ps_expander @ azm.one_over())*op)

        self.apply = op.apply
        self._fluc = fluctuations
        self._domain, self._target = op.domain, op.target
        self._space = space

    @property
    def fluctuation_amplitude(self):
        return self._fluc


class CorrelatedFieldMaker:
    def __init__(self, amplitude_offset, prefix, total_N):
        self._a = []
        self._spaces = []
        self._position_spaces = []
        
        self._azm = amplitude_offset
        self._prefix = prefix
        self._total_N = total_N
    
    @staticmethod
    def make(offset_amplitude_mean, offset_amplitude_stddev, prefix, total_N = 1):
        offset_amplitude_stddev = float(offset_amplitude_stddev)
        offset_amplitude_mean = float(offset_amplitude_mean)
        assert offset_amplitude_stddev > 0
        assert offset_amplitude_mean > 0
        zm = _LognormalMomentMatching(offset_amplitude_mean,
                                      offset_amplitude_stddev,
                                      prefix + 'zeromode',
                                      total_N)
        return CorrelatedFieldMaker(zm, prefix, total_N)

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
                         prefix = '',
                         index = None,
                         dofdex = None):
        if dofdex is None:
            dofdex = np.full(self._total_N, 0)
        else:
            assert len(dofdex) == self._total_N
        N = max(dofdex)

        if self._total_N > 1:
            space = 1
            position_space = makeDomain((UnstructuredDomain(self._total_N), position_space))
        else:
            space = 0
            position_space = makeDomain(position_space)
            N = 1
        power_space = PowerSpace(position_space[space].get_default_codomain())
        prefix = str(prefix)
        #assert isinstance(position_space[space], (RGSpace, HPSpace, GLSpace)

        fluct = _LognormalMomentMatching(fluctuations_mean,
                                         fluctuations_stddev,
                                         prefix + 'fluctuations',
                                         N)

        #if copies:
        #    fluct = fluct*self._azm.one_over()
        #else:
        #    #print(fluct.
        #    co = ContractionOperator(self._azm.target, None).adjoint
        #    fluct = (co @ fluct)*self._azm.one_over()

        flex = _LognormalMomentMatching(flexibility_mean, flexibility_stddev,
                                        prefix + 'flexibility',
                                        N)
        asp = _LognormalMomentMatching(asperity_mean, asperity_stddev,
                                       prefix + 'asperity', 
                                       N)
        avgsl = _normal(loglogavgslope_mean, loglogavgslope_stddev,
                        prefix + 'loglogavgslope', N)
        amp = _Amplitude(power_space,
                         fluct, flex, asp, avgsl, self._azm, prefix + 'spectrum', dofdex)

        if index is not None:
            self._a.insert(index, amp)
            self._position_spaces.insert(index, position_space)
            self._spaces.insert(index, space)
        else:
            self._a.append(amp)
            self._position_spaces.append(position_space)
            self._spaces.append(space)

    def finalize_from_op(self, zeromode, prefix=''):
        assert isinstance(zeromode, Operator)
        self._azm = zeromode
        n_amplitudes = len(self._a)
        if self._total_N > 1:
            hspace = makeDomain([UnstructuredDomain(self._total_N)] +
                    [dd[-1].get_default_codomain() for dd in self._position_spaces])
            spaces = tuple(len(dd) for dd in self._position_spaces)
            spaces = 1 + np.cumsum(spaces)
        else:
            hspace = makeDomain(
                    [dd[-1].get_default_codomain() for dd in self._position_spaces])
            spaces = tuple(range(n_amplitudes))
        zeroind = (slice(None),)*(1 - 1//self._total_N) + (0,)*(len(hspace.shape)-1+1//self._total_N)

        foo = np.ones(hspace.shape)
        foo[zeroind] = 0

        ZeroModeInserter = _slice_extractor(hspace, 
                     self._azm.target, zeroind).adjoint
        azm = Adder(from_global_data(hspace, foo)) @ ZeroModeInserter @ zeromode

        spaces = np.array(range(n_amplitudes)) + 1 - 1//self._total_N
        ht = HarmonicTransformOperator(hspace,
                                   self._position_spaces[0][self._spaces[0]],
                                   space=spaces[0])
        for i in range(1, n_amplitudes):
            ht = (HarmonicTransformOperator(ht.target,
                                    self._position_spaces[i][self._spaces[i]],
                                    space=spaces[i]) @ ht)

        pd = PowerDistributor(hspace, self._a[0].target[self._spaces[0]], self._spaces[0])
        for i in range(1, n_amplitudes):
            pd = (pd @ PowerDistributor(pd.domain,
                                   self._a[i].target[self._spaces[i]],
                                   space=spaces[i]))

        #breakpoint()
        all_spaces = list(range(len(hspace)))
        a = ContractionOperator(pd.domain, spaces[1:]).adjoint @ self._a[0]
        for i in range(1, n_amplitudes):
            co = ContractionOperator(pd.domain,
                    all_spaces[:spaces[i]] + all_spaces[spaces[i] + 1:])
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
            for m, s in zip(mean.flatten(), stddev.flatten()):
                print('{}: {:.02E} ± {:.02E}'.format(kk, m, s))

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
