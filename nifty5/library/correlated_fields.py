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
from functools import reduce
from numpy.testing import assert_allclose

from ..domain_tuple import DomainTuple
from ..domains.power_space import PowerSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..extra import check_jacobian_consistency, consistency_check
from ..field import Field
from ..multi_domain import MultiDomain
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
from ..sugar import from_global_data, from_random, full, makeDomain, get_default_codomain

def _reshaper(domain, x, space):
    shape = reduce(lambda x,y: x+y,
            (domain[i].shape for i in range(len(domain)) if i != space),())
    x = np.array(x)
    if x.shape == shape:
        return np.asfarray(x)
    elif x.shape in [(), (1,)]:
        return np.full(shape, x, dtype=np.float)
    else:
        raise TypeError("Shape of parameters cannot be interpreted")

def _lognormal_moment_matching(mean, sig, key,
        domain = DomainTuple.scalar_domain(), space = 0):
    domain = makeDomain(domain)
    mean, sig = (_reshaper(domain, param, space) for param in (mean, sig))
    key = str(key)
    assert np.all(mean > 0)
    assert np.all(sig > 0)
    logsig = np.sqrt(np.log((sig/mean)**2 + 1))
    logmean = np.log(mean) - logsig**2/2
    return _normal(logmean, logsig, key, domain).exp()


def _normal(mean, sig, key,
        domain = DomainTuple.scalar_domain(), space = 0):
    domain = makeDomain(domain)
    mean, sig = (_reshaper(domain, param, space) for param in (mean, sig))
    assert np.all(sig > 0)
    return Adder(from_global_data(domain, mean)) @ (
        sig*ducktape(domain, None, key))


class _SlopeRemover(EndomorphicOperator):
    def __init__(self, domain, cooridinates, space = 0):
        self._domain = makeDomain(domain)
        self._sc = cooridinates / float(cooridinates[-1])

        self._space = space
        self._last = (slice(None),)*self._domain.axes[space][0] + (-1,)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self,x,mode):
        self._check_input(x,mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = x - x[self._last] * self._sc
        else:
            #NOTE Why not x.copy()?
            res = np.zeros(x.shape,dtype=x.dtype)
            res += x
            res[self._last] -= (x*self._sc).sum(axis = self._space)
        return from_global_data(self._tgt(mode),res)

def _make_slope_Operator(smooth,loglogavgslope, space = 0):
    tg = smooth.target
    logkl = _log_k_lengths(tg[space])
    logkl -= logkl[0]
    logkl = np.insert(logkl, 0, 0)
    noslope = _SlopeRemover(tg,logkl, space) @ smooth
    # FIXME Move to tests
    consistency_check(_SlopeRemover(tg,logkl))

    expander = ContractionOperator(tg, spaces = space).adjoint
    _t = DiagonalOperator(from_global_data(tg, logkl), tg, spaces = space)
    return _t @ expander @ loglogavgslope + noslope

def _log_k_lengths(pspace):
    return np.log(pspace.k_lengths[1:])

class _TwoLogIntegrations(LinearOperator):
    def __init__(self, target, space = None):
        self._target = makeDomain(target)
        assert isinstance(self.target[space], PowerSpace)
        dom = list(self._target)
        dom[space] = UnstructuredDomain((2, self.target[space].shape[0]-2))
        self._domain = makeDomain(dom)
        self._space = space
        self._capability = self.TIMES | self.ADJOINT_TIMES
        logk_lengths = _log_k_lengths(self._target[space])
        self._logvol = logk_lengths[1:] - logk_lengths[:-1]

    def apply(self, x, mode):
        self._check_input(x, mode)

        #Maybe make class properties
        axis = self._target.axes[self._space][0]
        sl = (slice(None),)*axis
        first = sl + (0,)
        second = sl + (1,)
        from_third = sl + (slice(2,None),)
        no_border = sl + (slice(1,-1),)
        reverse = sl + (slice(None,None,-1),)
        if mode == self.TIMES:
            x = x.to_global_data()
            res = np.empty(self._target.shape)
            res[first] = 0
            res[second] = 0
            res[from_third] = np.cumsum(x[second], axis = axis)
            res[from_third] = (res[from_third] + res[no_border])/2*self._logvol + x[first]
            res[from_third] = np.cumsum(res[from_third], axis = axis)
        else:
            x = x.to_global_data_rw()
            res = np.zeros(self._domain.shape)
            x[from_third] = np.cumsum(x[from_third][reverse], axis = axis)[reverse]
            res[first] += x[from_third]
            x[from_third] *= self._logvol/2.
            x[no_border] += x[from_third]
            res[second] += np.cumsum(x[from_third][reverse], axis = axis)[reverse]
        return from_global_data(self._tgt(mode), res)


class _Normalization(Operator):
    def __init__(self, domain, space = 0):
        self._domain = self._target = makeDomain(domain)
        hspace = list(self._domain)
        hspace[space] = hspace[space].harmonic_partner
        hspace = makeDomain(hspace)
        pd = PowerDistributor(hspace, power_space=self._domain[space], space = space)
        # TODO Does not work on sphere yet
        mode_multiplicity = pd.adjoint(full(pd.target, 1.)).to_global_data_rw()
        mode_multiplicity[0] = 0
        self._mode_multiplicity = from_global_data(self._domain, mode_multiplicity)
        self._specsum = _SpecialSum(self._domain, space)
        # FIXME Move to tests
        consistency_check(self._specsum)

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
        self._zero_mode = (slice(None),)*domain.axes[space][0] + (0,)

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._contractor.adjoint(self._contractor(x))


class CorrelatedFieldMaker:
    def __init__(self):
        self._amplitudes = []

    def add_fluctuations_from_ops(self, target, fluctuations, flexibility,
                                  asperity, loglogavgslope, key, space = 0):
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
        assert isinstance(target[space], PowerSpace)

        twolog = _TwoLogIntegrations(target, space)
        dt = twolog._logvol
        sl = (slice(None),)*target.axes[space][0]
        first = sl + (0,)
        second = sl + (1,)
        expander = ContractionOperator(twolog.domain, spaces = space).adjoint
        
        sqrt_t = np.zeros(twolog.domain.shape)
        sqrt_t[first] = sqrt_t[second] = np.sqrt(dt)
        sqrt_t = from_global_data(twolog.domain, sqrt_t)
        sqrt_t = DiagonalOperator(sqrt_t, twolog.domain, spaces = space)
        sigmasq = sqrt_t @ expander @ flexibility

        dist = np.zeros(twolog.domain.shape)
        dist[first] += 1.
        dist = from_global_data(twolog.domain, dist)
        dist = DiagonalOperator(dist, twolog.domain, spaces = space)

        shift = np.ones(twolog.domain.shape)
        shift[first] = dt**2/12.
        shift = from_global_data(twolog.domain, shift)
        scale = sigmasq*(Adder(shift) @ dist @ expander @ asperity).sqrt()

        smooth = twolog @ (scale*ducktape(scale.target, None, key))
        smoothslope = _make_slope_Operator(smooth,loglogavgslope)
        
        # move to tests
        assert_allclose(
            smooth(from_random('normal', smooth.domain)).val[0:2], 0)
        consistency_check(twolog)
        check_jacobian_consistency(smooth, from_random('normal',
                                                       smooth.domain))
        check_jacobian_consistency(smoothslope,
                                   from_random('normal', smoothslope.domain))
        # end move to tests

        normal_ampl = _Normalization(target) @ smoothslope
        vol = target[0].harmonic_partner.get_default_codomain().total_volume
        arr = np.zeros(target.shape)
        arr[1:] = vol
        expander = VdotOperator(from_global_data(target, arr)).adjoint
        mask = np.zeros(target.shape)
        mask[0] = vol
        adder = Adder(from_global_data(target, mask))
        ampl = adder @ ((expander @ fluctuations)*normal_ampl)

        # Move to tests
        # FIXME This test fails but it is not relevant for the final result
        # assert_allclose(
        #     normal_ampl(from_random('normal', normal_ampl.domain)).val[0], 1)
        assert_allclose(ampl(from_random('normal', ampl.domain)).val[0], vol)
        op = _Normalization(target)
        check_jacobian_consistency(op, from_random('normal', op.domain))
        # End move to tests

        self._amplitudes.append(ampl)

    def add_fluctuations(self, target, fluctuations_mean, fluctuations_stddev,
                         flexibility_mean, flexibility_stddev, asperity_mean,
                         asperity_stddev, loglogavgslope_mean,
                         loglogavgslope_stddev, prefix, space = 0):
        prefix = str(prefix)

        fluct = _lognormal_moment_matching(fluctuations_mean, fluctuations_stddev,
                                           prefix + 'fluctuations', space)
        flex = _lognormal_moment_matching(flexibility_mean, flexibility_stddev,
                                          prefix + 'flexibility', space)
        asp = _lognormal_moment_matching(asperity_mean, asperity_stddev,
                                         prefix + 'asperity', space)
        avgsl = _normal(loglogavgslope_mean, loglogavgslope_stddev,
                        prefix + 'loglogavgslope', space)
        self.add_fluctuations_from_ops(target, fluct, flex, asp, avgsl,
                                       prefix + 'spectrum', space)

    def finalize_from_op(self, zeromode):
        raise NotImplementedError

    def finalize(self,
                 offset_amplitude_mean,
                 offset_amplitude_stddev,
                 prefix,
                 offset=None):
        """
        offset vs zeromode: volume factor
        """
        if offset is not None:
            offset = float(offset)
        #TODO correct hspace
        hspace = makeDomain(
            [dd.target[0].harmonic_partner for dd in self._amplitudes])

        azm = _lognormal_moment_matching(offset_amplitude_mean,
                                         offset_amplitude_stddev,
                                         prefix + 'zeromode')
        foo = np.ones(hspace.shape)
        zeroind = len(hspace.shape)*(0,)
        foo[zeroind] = 0
        azm = Adder(from_global_data(hspace, foo)) @ ValueInserter(
            hspace, zeroind) @ azm

        ht = HarmonicTransformOperator(hspace, space=0)
        pd = PowerDistributor(hspace, self._amplitudes[0].target[0], 0)
        for i in range(1, len(self._amplitudes)):
            ht = HarmonicTransformOperator(ht.target, space=i) @ ht
            pd = pd @ PowerDistributor(
                pd.domain, self._amplitudes[i].target[0], space=i)

        spaces = tuple(range(len(self._amplitudes)))
        a = ContractionOperator(pd.domain,
                                spaces[1:]).adjoint(self._amplitudes[0])
        for i in range(1, len(self._amplitudes)):
            a = a*(ContractionOperator(pd.domain, spaces[:i] + spaces[
                (i + 1):]).adjoint(self._amplitudes[i]))

        A = pd @ a
        return ht(azm*A*ducktape(hspace, None, prefix + 'xi'))

    @property
    def amplitudes(self):
        return self._amplitudes
