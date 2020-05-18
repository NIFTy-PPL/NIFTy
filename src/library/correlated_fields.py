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
# Authors: Philipp Frank, Philipp Arras, Philipp Haim
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
from operator import mul

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.power_space import PowerSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..linearization import Linearization
from ..logger import logger
from ..multi_field import MultiField
from ..operators.adder import Adder
from ..operators.contraction_operator import ContractionOperator
from ..operators.diagonal_operator import DiagonalOperator
from ..operators.distributors import PowerDistributor
from ..operators.endomorphic_operator import EndomorphicOperator
from ..operators.harmonic_operators import HarmonicTransformOperator
from ..operators.linear_operator import LinearOperator
from ..operators.operator import Operator
from ..operators.simple_linear_operators import ducktape
from ..probing import StatCalculator
from ..sugar import full, makeDomain, makeField, makeOp


def _reshaper(x, N):
    x = np.asfarray(x)
    if x.shape in [(), (1,)]:
        return np.full(N, x) if N != 0 else x.reshape(())
    elif x.shape == (N,):
        return x
    else:
        raise TypeError("Shape of parameters cannot be interpreted")


def _lognormal_moments(mean, sig, N=0):
    if N == 0:
        mean, sig = np.asfarray(mean), np.asfarray(sig)
    else:
        mean, sig = (_reshaper(param, N) for param in (mean, sig))
    if not np.all(mean > 0):
        raise ValueError("mean must be greater 0; got {!r}".format(mean))
    if not np.all(sig > 0):
        raise ValueError("sig must be greater 0; got {!r}".format(sig))

    logsig = np.sqrt(np.log1p((sig/mean)**2))
    logmean = np.log(mean) - logsig**2/2
    return logmean, logsig


def _normal(mean, sig, key, N=0):
    if N == 0:
        domain = DomainTuple.scalar_domain()
        mean, sig = np.asfarray(mean), np.asfarray(sig)
        return Adder(makeField(domain, mean)) @ (
            sig * ducktape(domain, None, key))

    domain = UnstructuredDomain(N)
    mean, sig = (_reshaper(param, N) for param in (mean, sig))
    return Adder(makeField(domain, mean)) @ (DiagonalOperator(
        makeField(domain, sig)) @ ducktape(domain, None, key))


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


def _structured_spaces(domain):
    if isinstance(domain[0], UnstructuredDomain):
        return np.arange(1, len(domain))
    return np.arange(len(domain))


def _total_fluctuation_realized(samples):
    spaces = _structured_spaces(samples[0].domain)
    co = ContractionOperator(samples[0].domain, spaces)
    size = co.domain.size/co.target.size
    res = 0.
    for s in samples:
        res = res + (s - co.adjoint(co(s)/size))**2
    res = res.mean(spaces)/len(samples)
    return np.sqrt(res if np.isscalar(res) else res.val)


class _LognormalMomentMatching(Operator):
    def __init__(self, mean, sig, key, N_copies):
        key = str(key)
        logmean, logsig = _lognormal_moments(mean, sig, N_copies)
        self._mean = mean
        self._sig = sig
        op = _normal(logmean, logsig, key, N_copies).ptw("exp")
        self._domain, self._target = op.domain, op.target
        self.apply = op.apply

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._sig


class _SlopeRemover(EndomorphicOperator):
    def __init__(self, domain, space=0):
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
        x = x.val
        if mode == self.TIMES:
            res = x - x[self._last]*self._sc[self._extender]
        else:
            res = x.copy()
            res[self._last] -= (x*self._sc[self._extender]).sum(
                axis=self._space, keepdims=True)
        return makeField(self._tgt(mode), res)


class _TwoLogIntegrations(LinearOperator):
    def __init__(self, target, space=0):
        self._target = makeDomain(target)
        assert isinstance(self.target[space], PowerSpace)
        dom = list(self._target)
        dom[space] = UnstructuredDomain((2, self.target[space].shape[0]-2))
        self._domain = makeDomain(dom)
        self._space = space
        self._log_vol = _log_vol(self._target[space])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)

        # Maybe make class properties
        axis = self._target.axes[self._space][0]
        sl = (slice(None),)*axis
        extender_sl = (None,)*axis + (slice(None),) + (None,)*(self._target.axes[-1][-1] - axis)
        first = sl + (0,)
        second = sl + (1,)
        from_third = sl + (slice(2, None),)
        no_border = sl + (slice(1, -1),)
        reverse = sl + (slice(None, None, -1),)

        if mode == self.TIMES:
            x = x.val
            res = np.empty(self._target.shape)
            res[first] = res[second] = 0
            res[from_third] = np.cumsum(x[second], axis=axis)
            res[from_third] = (res[from_third] + res[no_border])/2*self._log_vol[extender_sl] + x[first]
            res[from_third] = np.cumsum(res[from_third], axis=axis)
        else:
            x = x.val_rw()
            res = np.zeros(self._domain.shape)
            x[from_third] = np.cumsum(x[from_third][reverse], axis=axis)[reverse]
            res[first] += x[from_third]
            x[from_third] *= (self._log_vol/2.)[extender_sl]
            x[no_border] += x[from_third]
            res[second] += np.cumsum(x[from_third][reverse], axis=axis)[reverse]
        return makeField(self._tgt(mode), res)


class _Normalization(Operator):
    def __init__(self, domain, space=0):
        self._domain = self._target = DomainTuple.make(domain)
        assert isinstance(self._domain[space], PowerSpace)
        hspace = list(self._domain)
        hspace[space] = hspace[space].harmonic_partner
        hspace = makeDomain(hspace)
        pd = PowerDistributor(hspace,
                              power_space=self._domain[space],
                              space=space)
        mode_multiplicity = pd.adjoint(full(pd.target, 1.)).val_rw()
        zero_mode = (slice(None),)*self._domain.axes[space][0] + (0,)
        mode_multiplicity[zero_mode] = 0
        self._mode_multiplicity = makeOp(makeField(self._domain, mode_multiplicity))
        self._specsum = _SpecialSum(self._domain, space)

    def apply(self, x):
        self._check_input(x)
        amp = x.ptw("exp")
        spec = amp**2
        # FIXME This normalizes also the zeromode which is supposed to be left
        # untouched by this operator
        return self._specsum(self._mode_multiplicity(spec))**(-0.5)*amp


class _SpecialSum(EndomorphicOperator):
    def __init__(self, domain, space=0):
        self._domain = makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._contractor = ContractionOperator(domain, space)

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._contractor.adjoint(self._contractor(x))


class _Distributor(LinearOperator):
    def __init__(self, dofdex, domain, target):
        self._dofdex = dofdex

        self._target = makeDomain(target)
        self._domain = makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = x[self._dofdex]
        else:
            res = np.empty(self._tgt(mode).shape)
            res[self._dofdex] = x
        return makeField(self._tgt(mode), res)


class _Amplitude(Operator):
    def __init__(self, target, fluctuations, flexibility, asperity,
                 loglogavgslope, azm, totvol, key, dofdex):
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

        if len(dofdex) > 0:
            N_copies = max(dofdex) + 1
            space = 1
            distributed_tgt = makeDomain((UnstructuredDomain(len(dofdex)),
                                          target))
            target = makeDomain((UnstructuredDomain(N_copies), target))
            Distributor = _Distributor(dofdex, target, distributed_tgt)
        else:
            N_copies = 0
            space = 0
            distributed_tgt = target = makeDomain(target)
        azm_expander = ContractionOperator(distributed_tgt, spaces=space).adjoint
        assert isinstance(target[space], PowerSpace)

        twolog = _TwoLogIntegrations(target, space)
        dom = twolog.domain

        shp = dom[space].shape
        expander = ContractionOperator(dom, spaces=space).adjoint
        ps_expander = ContractionOperator(twolog.target, spaces=space).adjoint

        # Prepare constant fields
        foo = np.zeros(shp)
        foo[0] = foo[1] = np.sqrt(_log_vol(target[space]))
        vflex = DiagonalOperator(makeField(dom[space], foo), dom, space)

        foo = np.zeros(shp, dtype=np.float64)
        foo[0] += 1
        vasp = DiagonalOperator(makeField(dom[space], foo), dom, space)

        foo = np.ones(shp)
        foo[0] = _log_vol(target[space])**2/12.
        shift = DiagonalOperator(makeField(dom[space], foo), dom, space)

        vslope = DiagonalOperator(
            makeField(target[space], _relative_log_k_lengths(target[space])),
            target, space)

        foo, bar = [np.zeros(target[space].shape) for _ in range(2)]
        bar[1:] = foo[0] = totvol
        vol0, vol1 = [
            DiagonalOperator(makeField(target[space], aa), target, space)
            for aa in (foo, bar)
        ]

        # Prepare fields for Adder
        shift, vol0 = [op(full(op.domain, 1)) for op in (shift, vol0)]
        # End prepare constant fields

        slope = vslope @ ps_expander @ loglogavgslope
        sig_flex = vflex @ expander @ flexibility
        sig_asp = vasp @ expander @ asperity
        sig_fluc = vol1 @ ps_expander @ fluctuations
        sig_fluc = vol1 @ ps_expander @ fluctuations

        xi = ducktape(dom, None, key)
        sigma = sig_flex*(Adder(shift) @ sig_asp).ptw("sqrt")
        smooth = _SlopeRemover(target, space) @ twolog @ (sigma*xi)
        op = _Normalization(target, space) @ (slope + smooth)
        if N_copies > 0:
            op = Distributor @ op
            sig_fluc = Distributor @ sig_fluc
            op = Adder(Distributor(vol0)) @ (sig_fluc*(azm_expander @ azm.ptw("reciprocal"))*op)
            self._fluc = (_Distributor(dofdex, fluctuations.target,
                                       distributed_tgt[0]) @ fluctuations)
        else:
            op = Adder(vol0) @ (sig_fluc*(azm_expander @ azm.ptw("reciprocal"))*op)
            self._fluc = fluctuations

        self.apply = op.apply
        self._domain, self._target = op.domain, op.target
        self._space = space

    @property
    def fluctuation_amplitude(self):
        return self._fluc


class CorrelatedFieldMaker:
    def __init__(self, offset_mean, offset_fluctuations_op, prefix, total_N):
        if not isinstance(offset_fluctuations_op, Operator):
            raise TypeError("offset_fluctuations_op needs to be an operator")
        self._a = []
        self._position_spaces = []

        self._offset_mean = offset_mean
        self._azm = offset_fluctuations_op
        self._prefix = prefix
        self._total_N = total_N

    @staticmethod
    def make(offset_mean, offset_std_mean, offset_std_std, prefix,
             total_N=0,
             dofdex=None):
        """Returns a CorrelatedFieldMaker object.

        Parameters
        ----------
        offset_mean : float
            Mean offset from zero of the correlated field to be made.
        offset_std_mean : float
            Mean standard deviation of the offset value.
        offset_std_std : float
            Standard deviation of the stddev of the offset value.
        prefix : string
            Prefix to the names of the domains of the cf operator to be made.
        total_N : integer
            ?
        dofdex : np.array
            ?
        """
        if dofdex is None:
            dofdex = np.full(total_N, 0)
        elif len(dofdex) != total_N:
            raise ValueError("length of dofdex needs to match total_N")
        N = max(dofdex) + 1 if total_N > 0 else 0
        zm = _LognormalMomentMatching(offset_std_mean,
                                      offset_std_std,
                                      prefix + 'zeromode',
                                      N)
        if total_N > 0:
            zm = _Distributor(dofdex, zm.target, UnstructuredDomain(total_N)) @ zm
        return CorrelatedFieldMaker(offset_mean, zm, prefix, total_N)

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
                         index=None,
                         dofdex=None,
                         harmonic_partner=None):
        if harmonic_partner is None:
            harmonic_partner = position_space.get_default_codomain()
        else:
            position_space.check_codomain(harmonic_partner)
            harmonic_partner.check_codomain(position_space)

        if dofdex is None:
            dofdex = np.full(self._total_N, 0)
        elif len(dofdex) != self._total_N:
            raise ValueError("length of dofdex needs to match total_N")

        if self._total_N > 0:
            N = max(dofdex) + 1
            position_space = makeDomain((UnstructuredDomain(N), position_space))
        else:
            N = 0
            position_space = makeDomain(position_space)
        prefix = str(prefix)
        # assert isinstance(position_space[space], (RGSpace, HPSpace, GLSpace)

        fluct = _LognormalMomentMatching(fluctuations_mean,
                                         fluctuations_stddev,
                                         self._prefix + prefix + 'fluctuations',
                                         N)
        flex = _LognormalMomentMatching(flexibility_mean, flexibility_stddev,
                                        self._prefix + prefix + 'flexibility',
                                        N)
        asp = _LognormalMomentMatching(asperity_mean, asperity_stddev,
                                       self._prefix + prefix + 'asperity', N)
        avgsl = _normal(loglogavgslope_mean, loglogavgslope_stddev,
                        self._prefix + prefix + 'loglogavgslope', N)
        amp = _Amplitude(PowerSpace(harmonic_partner), fluct, flex, asp, avgsl,
                         self._azm, position_space[-1].total_volume,
                         self._prefix + prefix + 'spectrum', dofdex)

        if index is not None:
            self._a.insert(index, amp)
            self._position_spaces.insert(index, position_space)
        else:
            self._a.append(amp)
            self._position_spaces.append(position_space)

    def _finalize_from_op(self):
        n_amplitudes = len(self._a)
        if self._total_N > 0:
            hspace = makeDomain(
                [UnstructuredDomain(self._total_N)] +
                [dd.target[-1].harmonic_partner for dd in self._a])
            spaces = tuple(range(1, n_amplitudes + 1))
            amp_space = 1
        else:
            hspace = makeDomain(
                [dd.target[0].harmonic_partner for dd in self._a])
            spaces = tuple(range(n_amplitudes))
            amp_space = 0

        expander = ContractionOperator(hspace, spaces=spaces).adjoint
        azm = expander @ self._azm

        ht = HarmonicTransformOperator(hspace,
                                       self._position_spaces[0][amp_space],
                                       space=spaces[0])
        for i in range(1, n_amplitudes):
            ht = HarmonicTransformOperator(ht.target,
                                           self._position_spaces[i][amp_space],
                                           space=spaces[i]) @ ht
        a = []
        for ii in range(n_amplitudes):
            co = ContractionOperator(hspace, spaces[:ii] + spaces[ii + 1:])
            pp = self._a[ii].target[amp_space]
            pd = PowerDistributor(co.target, pp, amp_space)
            a.append(co.adjoint @ pd @ self._a[ii])
        corr = reduce(mul, a)
        return ht(azm*corr*ducktape(hspace, None, self._prefix + 'xi'))

    def finalize(self, prior_info=100):
        """
        offset vs zeromode: volume factor
        """
        op = self._finalize_from_op()
        if self._offset_mean is not None:
            offset = self._offset_mean
            # Deviations from this offset must not be considered here as they
            # are learned by the zeromode
            if isinstance(offset, (Field, MultiField)):
                op = Adder(offset) @ op
            else:
                offset = float(offset)
                op = Adder(full(op.target, offset)) @ op
        self.statistics_summary(prior_info)
        return op

    def statistics_summary(self, prior_info):
        from ..sugar import from_random

        if prior_info == 0:
            return

        lst = [('Offset amplitude', self.amplitude_total_offset),
               ('Total fluctuation amplitude', self.total_fluctuation)]
        namps = len(self._a)
        if namps > 1:
            for ii in range(namps):
                lst.append(('Slice fluctuation (space {})'.format(ii),
                            self.slice_fluctuation(ii)))
                lst.append(('Average fluctuation (space {})'.format(ii),
                            self.average_fluctuation(ii)))

        for kk, op in lst:
            sc = StatCalculator()
            for _ in range(prior_info):
                sc.add(op(from_random('normal', op.domain)))
            mean = sc.mean.val
            stddev = sc.var.ptw("sqrt").val
            for m, s in zip(mean.flatten(), stddev.flatten()):
                logger.info('{}: {:.02E} ± {:.02E}'.format(kk, m, s))

    def moment_slice_to_average(self, fluctuations_slice_mean, nsamples=1000):
        fluctuations_slice_mean = float(fluctuations_slice_mean)
        if not fluctuations_slice_mean > 0:
            msg = "fluctuations_slice_mean must be greater zero; got {!r}"
            raise ValueError(msg.format(fluctuations_slice_mean))
        from ..sugar import from_random
        scm = 1.
        for a in self._a:
            op = a.fluctuation_amplitude*self._azm.ptw("reciprocal")
            res = np.array([op(from_random('normal', op.domain)).val
                            for _ in range(nsamples)])
            scm *= res**2 + 1.
        return fluctuations_slice_mean/np.mean(np.sqrt(scm))

    @property
    def normalized_amplitudes(self):
        return self._a

    @property
    def amplitude(self):
        if len(self._a) > 1:
            s = ('If more than one spectrum is present in the model,',
                 ' no unique set of amplitudes exist because only the',
                 ' relative scale is determined.')
            raise NotImplementedError(s)
        dom = self._a[0].target
        expand = ContractionOperator(dom, len(dom)-1).adjoint
        return self._a[0]*(expand @ self.amplitude_total_offset)

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
            fl = a.fluctuation_amplitude*self._azm.ptw("reciprocal")
            q = q*(Adder(full(fl.target, 1.)) @ fl**2)
        return (Adder(full(q.target, -1.)) @ q).ptw("sqrt")*self._azm

    def slice_fluctuation(self, space):
        """Returns operator which acts on prior or posterior samples"""
        if len(self._a) == 0:
            raise NotImplementedError
        if space >= len(self._a):
            raise ValueError("invalid space specified; got {!r}".format(space))
        if len(self._a) == 1:
            return self.average_fluctuation(0)
        q = 1.
        for j in range(len(self._a)):
            fl = self._a[j].fluctuation_amplitude*self._azm.ptw("reciprocal")
            if j == space:
                q = q*fl**2
            else:
                q = q*(Adder(full(fl.target, 1.)) @ fl**2)
        return q.ptw("sqrt")*self._azm

    def average_fluctuation(self, space):
        """Returns operator which acts on prior or posterior samples"""
        if len(self._a) == 0:
            raise NotImplementedError
        if space >= len(self._a):
            raise ValueError("invalid space specified; got {!r}".format(space))
        if len(self._a) == 1:
            return self._a[0].fluctuation_amplitude
        return self._a[space].fluctuation_amplitude

    @staticmethod
    def offset_amplitude_realized(samples):
        spaces = _structured_spaces(samples[0].domain)
        res = 0.
        for s in samples:
            res = res + s.mean(spaces)**2
        res = res/len(samples)
        return np.sqrt(res if np.isscalar(res) else res.val)

    @staticmethod
    def total_fluctuation_realized(samples):
        return _total_fluctuation_realized(samples)

    @staticmethod
    def slice_fluctuation_realized(samples, space):
        """Computes slice fluctuations from collection of field (defined in signal
        space) realizations."""
        spaces = _structured_spaces(samples[0].domain)
        if space >= len(spaces):
            raise ValueError("invalid space specified; got {!r}".format(space))
        if len(spaces) == 1:
            return _total_fluctuation_realized(samples)
        space = space + spaces[0]
        res1, res2 = 0., 0.
        for s in samples:
            res1 = res1 + s**2
            res2 = res2 + s.mean(space)**2
        res1 = res1/len(samples)
        res2 = res2/len(samples)
        res = res1.mean(spaces) - res2.mean(spaces[:-1])
        return np.sqrt(res if np.isscalar(res) else res.val)

    @staticmethod
    def average_fluctuation_realized(samples, space):
        """Computes average fluctuations from collection of field (defined in signal
        space) realizations."""
        spaces = _structured_spaces(samples[0].domain)
        if space >= len(spaces):
            raise ValueError("invalid space specified; got {!r}".format(space))
        if len(spaces) == 1:
            return _total_fluctuation_realized(samples)
        space = space + spaces[0]
        sub_spaces = set(spaces)
        sub_spaces.remove(space)
        # Domain containing domain[space] and domain[0] iff total_N>0
        sub_dom = makeDomain([samples[0].domain[ind]
                              for ind in (set([0])-set(spaces)) | set([space])])
        co = ContractionOperator(sub_dom, len(sub_dom)-1)
        size = co.domain.size/co.target.size
        res = 0.
        for s in samples:
            r = s.mean(sub_spaces)
            res = res + (r - co.adjoint(co(r)/size))**2
        res = res.mean(spaces[0])/len(samples)
        return np.sqrt(res if np.isscalar(res) else res.val)
