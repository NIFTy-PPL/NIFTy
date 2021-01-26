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
# Copyright(C) 2013-2020 Max-Planck-Society
# Authors: Philipp Frank, Philipp Arras, Philipp Haim;
#          Matern Kernel by Matteo Guardiani, Jakob Roth
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
from operator import mul

import numpy as np

from .. import utilities
from ..domain_tuple import DomainTuple
from ..domains.power_space import PowerSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
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
from ..operators.simple_linear_operators import ducktape, VdotOperator
from ..operators.normal_operators import NormalTransform, LognormalTransform
from ..probing import StatCalculator
from ..sugar import full, makeDomain, makeField, makeOp


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


class _SlopeRemover(EndomorphicOperator):
    def __init__(self, domain, space=0):
        self._domain = makeDomain(domain)
        assert isinstance(self._domain[space], PowerSpace)
        logkl = _relative_log_k_lengths(self._domain[space])
        sc = logkl/float(logkl[-1])

        self._space = space
        axis = self._domain.axes[space][0]
        self._last = (slice(None),)*axis + (-1,) + (None,)
        extender = (None,)*(axis) + (slice(None),) + (None,)*(self._domain.axes[-1][-1]-axis)
        self._sc = sc[extender]
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            x = x.val
            res = x - x[self._last]*self._sc
        else:
            res = x.val_rw()
            res[self._last] -= (res*self._sc).sum(axis=self._space, keepdims=True)
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
        multipl = makeOp(makeField(self._domain, mode_multiplicity))
        self._specsum = _SpecialSum(self._domain, space) @ multipl

    def apply(self, x):
        self._check_input(x)
        spec = x.ptw("exp")
        # FIXME This normalizes also the zeromode which is supposed to be left
        # untouched by this operator
        return (self._specsum(spec).reciprocal()*spec).sqrt()


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
        self._dofdex = np.array(dofdex)
        self._target = DomainTuple.make(target)
        self._domain = DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = x[self._dofdex]
        else:
            res = np.zeros(self._tgt(mode).shape, dtype=x.dtype)
            res = utilities.special_add_at(res, 0, self._dofdex, x)
        return makeField(self._tgt(mode), res)


class _AmplitudeMatern(Operator):
    def __init__(self, pow_spc, scale, cutoff, logloghalfslope, totvol):
        expander = VdotOperator(full(pow_spc, 1.)).adjoint
        k_squared = makeField(pow_spc, pow_spc.k_lengths**2)

        a = expander @ scale.log()  # FIXME: look for nicer implementation
        b = VdotOperator(k_squared).adjoint @ cutoff.power(-2.)
        c = expander.scale(-1) @ logloghalfslope

        ker = Adder(full(pow_spc, 1.)) @ b
        ker = c * ker.log() + a
        op = ker.exp()
        # Account for the volume of the position space (dvol in harmonic space
        # is 1/volume-of-position-space) in the definition of the amplitude as
        # to make the parametric model agnostic to changes in the volume of the
        # position space
        op = totvol**0.5 * op

        # std = sqrt of integral of power spectrum
        self._fluc = op.power(2).integrate().sqrt()
        self.apply = op.apply
        self._domain, self._target = op.domain, op.target
        self._repr_str = "_AmplitudeMatern: " + op.__repr__()

    @property
    def fluctuation_amplitude(self):
        return self._fluc

    def __repr__(self):
        return self._repr_str


class _Amplitude(Operator):
    def __init__(self, target, fluctuations, flexibility, asperity,
                 loglogavgslope, azm, totvol, key, dofdex):
        """
        fluctuations > 0
        flexibility > 0 or None
        asperity > 0 or None
        loglogavgslope probably negative
        """
        assert isinstance(fluctuations, Operator)
        assert isinstance(flexibility, Operator) or flexibility is None
        assert isinstance(asperity, Operator) or asperity is None
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
        vflex = np.zeros(shp)
        vflex[0] = vflex[1] = np.sqrt(_log_vol(target[space]))
        vflex = DiagonalOperator(makeField(dom[space], vflex), dom, space)

        vasp = np.zeros(shp, dtype=np.float64)
        vasp[0] += 1
        vasp = DiagonalOperator(makeField(dom[space], vasp), dom, space)

        shift = np.ones(shp)
        shift[0] = _log_vol(target[space])**2 / 12.
        shift = DiagonalOperator(makeField(dom[space], shift), dom, space)
        shift = shift(full(shift.domain, 1))

        vslope = DiagonalOperator(
            makeField(target[space], _relative_log_k_lengths(target[space])),
            target, space)

        vol0, vol1 = [np.zeros(target[space].shape) for _ in range(2)]
        vol1[1:] = vol0[0] = totvol
        vol0, vol1 = [
            DiagonalOperator(makeField(target[space], aa), target, space)
            for aa in (vol0, vol1)
        ]
        vol0 = vol0(full(vol0.domain, 1))
        # End prepare constant fields

        slope = vslope @ ps_expander @ loglogavgslope
        sig_flex = vflex @ expander @ flexibility if flexibility is not None else None
        sig_asp = vasp @ expander @ asperity if asperity is not None else None
        sig_fluc = vol1 @ ps_expander @ fluctuations
        sig_fluc = vol1 @ ps_expander @ fluctuations

        if sig_asp is None and sig_flex is None:
            op = _Normalization(target, space) @ slope
        elif sig_asp is None:
            xi = ducktape(dom, None, key)
            sigma = DiagonalOperator(shift.ptw("sqrt"), dom) @ sig_flex
            smooth = _SlopeRemover(target, space) @ twolog @ (sigma * xi)
            op = _Normalization(target, space) @ (slope + smooth)
        elif sig_flex is None:
            raise ValueError("flexibility may not be disabled on its own")
        else:
            xi = ducktape(dom, None, key)
            sigma = sig_flex * (Adder(shift) @ sig_asp).ptw("sqrt")
            smooth = _SlopeRemover(target, space) @ twolog @ (sigma * xi)
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
        self._repr_str = "_Amplitude: " + op.__repr__()

    @property
    def fluctuation_amplitude(self):
        return self._fluc

    def __repr__(self):
        return self._repr_str


class CorrelatedFieldMaker:
    """Construction helper for hierarchical correlated field models.

    The correlated field models are parametrized by creating
    power spectrum operators ("amplitudes") via calls to
    :func:`add_fluctuations` that act on the targeted field subdomains.
    During creation of the :class:`CorrelatedFieldMaker` via
    :func:`make`, a global offset from zero of the field model
    can be defined and an operator applying fluctuations
    around this offset is parametrized.

    The resulting correlated field model operator has a
    :class:`~nifty7.multi_domain.MultiDomain` as its domain and
    expects its input values to be univariately gaussian.

    The target of the constructed operator will be a
    :class:`~nifty7.domain_tuple.DomainTuple` containing the
    `target_subdomains` of the added fluctuations in the order of
    the `add_fluctuations` calls.

    Creation of the model operator is completed by calling the method
    :func:`finalize`, which returns the configured operator.

    An operator representing an array of correlated field models
    can be constructed by setting the `total_N` parameter of
    :func:`make`. It will have an
    :class:`~nifty.domains.unstructucture_domain.UnstructureDomain`
    of shape `(total_N,)` prepended to its target domain and represent
    `total_N` correlated fields simulataneously.
    The degree of information sharing between the correlated field
    models can be configured via the `dofdex` parameters
    of :func:`make` and :func:`add_fluctuations`.

    See the methods :func:`make`, :func:`add_fluctuations`
    and :func:`finalize` for further usage information."""
    def __init__(self, offset_mean, offset_fluctuations_op, prefix, total_N):
        if not isinstance(offset_fluctuations_op, Operator):
            raise TypeError("offset_fluctuations_op needs to be an operator")
        self._a = []
        self._target_subdomains = []

        self._offset_mean = offset_mean
        self._azm = offset_fluctuations_op
        self._prefix = prefix
        self._total_N = total_N

    @staticmethod
    def make(offset_mean, offset_std, prefix, total_N=0, dofdex=None):
        """Returns a CorrelatedFieldMaker object.

        Parameters
        ----------
        offset_mean : float
            Mean offset from zero of the correlated field to be made.
        offset_std : tuple of float
            Mean standard deviation and standard deviation of the standard
            deviation of the offset. No, this is not a word duplication.
        prefix : string
            Prefix to the names of the domains of the cf operator to be made.
            This determines the names of the operator domain.
        total_N : integer, optional
            Number of field models to create.
            If not 0, the first entry of the operators target will be an
            :class:`~nifty.domains.unstructured_domain.UnstructuredDomain`
            with length `total_N`.
        dofdex : np.array of integers, optional
            An integer array specifying the zero mode models used if
            total_N > 1. It needs to have length of total_N. If total_N=3 and
            dofdex=[0,0,1], that means that two models for the zero mode are
            instantiated; the first one is used for the first and second
            field model and the second is used for the third field model.
            *If not specified*, use the same zero mode model for all
            constructed field models.
        """
        if dofdex is None:
            dofdex = np.full(total_N, 0)
        elif len(dofdex) != total_N:
            raise ValueError("length of dofdex needs to match total_N")
        N = max(dofdex) + 1 if total_N > 0 else 0
        if len(offset_std) != 2:
            raise TypeError
        zm = LognormalTransform(*offset_std, prefix + 'zeromode', N)
        if total_N > 0:
            zm = _Distributor(dofdex, zm.target, UnstructuredDomain(total_N)) @ zm
        return CorrelatedFieldMaker(offset_mean, zm, prefix, total_N)

    def add_fluctuations(self,
                         target_subdomain,
                         fluctuations,
                         flexibility,
                         asperity,
                         loglogavgslope,
                         prefix='',
                         index=None,
                         dofdex=None,
                         harmonic_partner=None):
        """Function to add correlation structures to the field to be made.

        Correlations are described by their power spectrum and the subdomain
        on which they apply.

        The parameters `fluctuations`, `flexibility`, `asperity` and
        `loglogavgslope` configure the power spectrum model used on the target
        field subdomain `target_subdomain`. It is assembled as the sum of a
        power law component (linear slope in log-log power-frequency-space), a
        smooth varying component (integrated Wiener process) and a ragged
        component (un-integrated Wiener process).

        Multiple calls to `add_fluctuations` are possible, in which case
        the constructed field will have the outer product of the individual
        power spectra as its global power spectrum.

        Parameters
        ----------
        target_subdomain : :class:`~nifty7.domain.Domain`, \
                           :class:`~nifty7.domain_tuple.DomainTuple`
            Target subdomain on which the correlation structure defined
            in this call should hold.
        fluctuations : tuple of float (mean, std)
            Total spectral energy -> Amplitude of the fluctuations
            LogNormal distribution
        flexibility : tuple of float (mean, std) or None
            Amplitude of the non-power-law power spectrum component
            LogNormal distribution
        asperity : tuple of float (mean, std) or None
            Roughness of the non-power-law power spectrum component
            Used to accommodate single frequency peaks
            LogNormal distribution
        loglogavgslope : tuple of float (mean, std)
            Power law component exponent
            Normal distribution
        prefix : string
            prefix of the power spectrum parameter domain names
        index : int
            Position target_subdomain in the final total domain of the
            correlated field operator.
        dofdex : np.array, optional
            An integer array specifying the power spectrum models used if
            total_N > 1. It needs to have length of total_N. If total_N=3 and
            dofdex=[0,0,1], that means that two power spectrum models are
            instantiated; the first one is used for the first and second
            field model and the second one is used for the third field model.
            *If not given*, use the same power spectrum model for all
            constructed field models.
        harmonic_partner : :class:`~nifty7.domain.Domain`, \
                           :class:`~nifty7.domain_tuple.DomainTuple`
            In which harmonic space to define the power spectrum
        """
        if harmonic_partner is None:
            harmonic_partner = target_subdomain.get_default_codomain()
        else:
            target_subdomain.check_codomain(harmonic_partner)
            harmonic_partner.check_codomain(target_subdomain)

        if dofdex is None:
            dofdex = np.full(self._total_N, 0)
        elif len(dofdex) != self._total_N:
            raise ValueError("length of dofdex needs to match total_N")

        if self._total_N > 0:
            N = max(dofdex) + 1
            target_subdomain = makeDomain((UnstructuredDomain(N), target_subdomain))
        else:
            N = 0
            target_subdomain = makeDomain(target_subdomain)
        # assert isinstance(target_subdomain[space], (RGSpace, HPSpace, GLSpace))

        for arg in [fluctuations, loglogavgslope]:
            if len(arg) != 2:
                raise TypeError
        for kw, arg in [("flexibility", flexibility), ("asperity", asperity)]:
            if arg is None:
                continue
            if len(arg) != 2:
                raise TypeError
            if len(arg) == 2 and (arg[0] <= 0. or arg[1] <= 0.):
                ve = "{0} must be strictly positive (or None)"
                raise ValueError(ve.format(kw))
        if flexibility is None and asperity is not None:
            raise ValueError("flexibility may not be disabled on its own")

        pre = self._prefix + str(prefix)
        fluct = LognormalTransform(*fluctuations, pre + 'fluctuations', N)
        if flexibility is not None:
            flex = LognormalTransform(*flexibility, pre + 'flexibility', N)
        else:
            flex = None
        if asperity is not None:
            asp = LognormalTransform(*asperity, pre + 'asperity', N)
        else:
            asp = None
        avgsl = NormalTransform(*loglogavgslope, pre + 'loglogavgslope', N)

        amp = _Amplitude(PowerSpace(harmonic_partner), fluct, flex, asp, avgsl,
                         self._azm, target_subdomain[-1].total_volume,
                         pre + 'spectrum', dofdex)

        if index is not None:
            self._a.insert(index, amp)
            self._target_subdomains.insert(index, target_subdomain)
        else:
            self._a.append(amp)
            self._target_subdomains.append(target_subdomain)

    def add_fluctuations_matern(self,
                                target_subdomain,
                                scale,
                                cutoff,
                                logloghalfslope,
                                prefix='',
                                adjust_for_volume=True,
                                harmonic_partner=None):
        """Function to add matern kernels to the field to be made.

        The matern kernel amplitude is parametrized in the following way:

        .. math ::
            A(|k|) = \\sqrt{V} \cdot \\frac{a}{\\left(1 + { \
                \\left(\\frac{|k|}{b}\\right) \
            }^2\\right)^c}

        with 'a' being the scale, 'b' the cutoff, 'c' half the slope of the
        power law and 'V' the volume in position space.

        Parameters
        ----------
        target_subdomain : :class:`~nifty7.domain.Domain`, \
                           :class:`~nifty7.domain_tuple.DomainTuple`
            Target subdomain on which the correlation structure defined
            in this call should hold.
        scale : tuple of float (mean, std)
            Overall scale of the fluctuations in the target subdomain.
        cutoff : tuple of float (mean, std)
            Frequency at which the power spectrum should transition into
            a spectra following a power-law.
        logloghalfslope : tuple of float (mean, std)
            Half of the slope of the amplitude spectrum.
        prefix : string
            Prefix of the power spectrum parameter domain names.
        adjust_for_volume : bool, optional
            Whether to implicitly adjust the parameters of the Matern kernel
            for the volume in the target subdomain or assume them to be
            adjusted already.
        harmonic_partner : :class:`~nifty7.domain.Domain`, \
                           :class:`~nifty7.domain_tuple.DomainTuple`
            Harmonic space in which to define the power spectrum.

        Notes
        -----
        The parameters of the amplitude model are assumed to be relative to a
        unit-less power spectrum, i.e. the parameters are assumed to be
        agnostic to changes in the volume of the target subdomain. This is in
        steep contrast to the non-parametric amplitude operator in
        :class:`~nifty7.CorrelatedFieldMaker.add_fluctuations`.
        """
        if harmonic_partner is None:
            harmonic_partner = target_subdomain.get_default_codomain()
        else:
            target_subdomain.check_codomain(harmonic_partner)
            harmonic_partner.check_codomain(target_subdomain)
        target_subdomain = makeDomain(target_subdomain)

        scale = LognormalTransform(*scale, self._prefix + prefix + 'scale', 0)
        prfx = self._prefix + prefix + 'cutoff'
        cutoff = LognormalTransform(*cutoff, prfx, 0)
        prfx = self._prefix + prefix + 'logloghalfslope'
        logloghalfslope = NormalTransform(*logloghalfslope, prfx, 0)

        totvol = 1.
        if adjust_for_volume:
            totvol = target_subdomain[-1].total_volume
        pow_spc = PowerSpace(harmonic_partner)
        amp = _AmplitudeMatern(pow_spc, scale, cutoff, logloghalfslope, totvol)

        self._a.append(amp)
        self._target_subdomains.append(target_subdomain)

    def finalize(self, prior_info=100):
        """Finishes model construction process and returns the constructed
        operator.

        Parameters
        ----------
        prior_info : integer
            How many prior samples to draw for property verification statistics
            If zero, skips calculating and displaying statistics.
        """
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
                                       self._target_subdomains[0][amp_space],
                                       space=spaces[0])
        for i in range(1, n_amplitudes):
            ht = HarmonicTransformOperator(ht.target,
                                           self._target_subdomains[i][amp_space],
                                           space=spaces[i]) @ ht
        a = []
        for ii in range(n_amplitudes):
            co = ContractionOperator(hspace, spaces[:ii] + spaces[ii + 1:])
            pp = self._a[ii].target[amp_space]
            pd = PowerDistributor(co.target, pp, amp_space)
            a.append(co.adjoint @ pd @ self._a[ii])
        corr = reduce(mul, a)
        op = ht(azm*corr*ducktape(hspace, None, self._prefix + 'xi'))

        if self._offset_mean is not None:
            offset = self._offset_mean
            # Deviations from this offset must not be considered here as they
            # are learned by the zeromode
            if isinstance(offset, (Field, MultiField)):
                op = Adder(offset) @ op
            else:
                offset = float(offset)
                op = Adder(full(op.target, offset)) @ op
        #FIXME why does prior_info no longer works???
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
                sc.add(op(from_random(op.domain, 'normal')))
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
            res = np.array([op(from_random(op.domain, 'normal')).val
                            for _ in range(nsamples)])
            scm *= res**2 + 1.
        return fluctuations_slice_mean/np.mean(np.sqrt(scm))

    @property
    def normalized_amplitudes(self):
        """Returns the amplitude operators used in the model"""
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
    def power_spectrum(self):
        return self.amplitude**2

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
