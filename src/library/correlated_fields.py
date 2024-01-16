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
# Copyright(C) 2013-2021 Max-Planck-Society
# Authors: Philipp Frank, Philipp Arras, Philipp Haim;
#          Matern Kernel by Matteo Guardiani, Jakob Roth and
#          Gordian Edenhofer
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
from operator import mul
from warnings import warn

import numpy as np

from .. import utilities
from ..domain_tuple import DomainTuple
from ..domains.power_space import PowerSpace
from ..domains.rg_space import RGSpace
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
from ..operators.normal_operators import LognormalTransform, NormalTransform
from ..operators.operator import Operator
from ..operators.simple_linear_operators import VdotOperator, ducktape
from ..probing import StatCalculator
from ..sugar import full, makeDomain, makeField, makeOp
from ..utilities import myassert


def _log_k_lengths(pspace):
    """Log(k_lengths) without zeromode"""
    return np.log(pspace.k_lengths[1:])


def _relative_log_k_lengths(power_space):
    """Log-distance to first bin
    logkl.shape==power_space.shape, logkl[0]=logkl[1]=0"""
    power_space = DomainTuple.make(power_space)
    myassert(isinstance(power_space[0], PowerSpace))
    myassert(len(power_space) == 1)
    logkl = _log_k_lengths(power_space[0])
    myassert(logkl.shape[0] == power_space[0].shape[0] - 1)
    logkl -= logkl[0]
    return np.insert(logkl, 0, 0)


def _log_vol(power_space):
    power_space = makeDomain(power_space)
    myassert(isinstance(power_space[0], PowerSpace))
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
        myassert(isinstance(self._domain[space], PowerSpace))
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
        myassert(isinstance(self.target[space], PowerSpace))
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
    """Exponentiate the logarithmic power spectrum, normalize by the sum over
    all modes and return the square root of the "normalized" power spectrum.

    Notes
    -----
    The operator does not perform a proper normalization as it does not account
    for changes in position space volume. This leads to an additional factor of
    `1 / \\sqrt{totvol}` being introduced into the result with `totvol`
    referring to the total volume in position space. The exact value of the
    additional factor stems from the fact that the volume in harmonic space is
    solely dependent on the distances in position space. Thus, if the position
    spaces is enlarged without changing its distances, the volume in harmonic
    space is kept constant. Doubling the number of pixels though also doubles
    the number of harmonic modes with each then occupy a smaller volume. This
    linear decrease in per pixel volume in harmonic space is not captured by
    just summing up the modes.
    """
    def __init__(self, domain, space=0):
        self._domain = self._target = DomainTuple.make(domain)
        myassert(isinstance(self._domain[space], PowerSpace))
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
        # NOTE, see the note in the doc-string on why this is not a proper
        # normalization!
        # NOTE, this "normalizes" also the zero-mode which is supposed to be
        # left untouched by this operator. Since the multiplicity of the
        # zero-mode is set to 0, the norm does not contain traces of it.
        # However, it wrongly sets the zeroth entry of the result. Luckily,
        # in subsequent calls, the zeroth entry is not used in the CF model.
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
            np.add.at(res, self._dofdex, x)
        return makeField(self._tgt(mode), res)


class _AmplitudeMatern(Operator):
    def __init__(self, pow_spc, scale, cutoff, loglogslope, totvol):
        expander = ContractionOperator(pow_spc, spaces=None).adjoint
        k_squared = makeField(pow_spc, pow_spc.k_lengths**2)

        scale = expander @ scale.log()
        cutoff = VdotOperator(k_squared).adjoint @ cutoff.power(-2.)
        spectral_idx = expander.scale(0.25) @ loglogslope

        ker = Adder(full(pow_spc, 1.)) @ cutoff
        ker = spectral_idx * ker.log() + scale
        op = ker.exp()

        # Account for the volume of the position space (dvol in harmonic space
        # is 1/volume-of-position-space) in the definition of the amplitude as
        # to make the parametric model agnostic to changes in the volume of the
        # position space
        vol0, vol1 = [np.zeros(pow_spc.shape) for _ in range(2)]
        # The zero-mode scales linearly with the volume in position space
        vol0[0] = totvol
        # Variances decrease linearly with the volume in position space after a
        # harmonic transformation (var{HT(randn)} \propto 1/\sqrt{totvol} for
        # randn standard normally distributed variables).  This needs to be
        # accounted for in the amplitude model.
        vol1[1:] = totvol**0.5
        vol0 = Adder(makeField(pow_spc, vol0))
        vol1 = DiagonalOperator(makeField(pow_spc, vol1))
        op = vol0 @ vol1 @ op

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
                 loglogavgslope, totvol, key, dofdex):
        """
        fluctuations > 0
        flexibility > 0 or None
        asperity > 0 or None
        loglogavgslope probably negative
        """
        myassert(isinstance(fluctuations, Operator))
        myassert(isinstance(flexibility, Operator) or flexibility is None)
        myassert(isinstance(asperity, Operator) or asperity is None)
        myassert(isinstance(loglogavgslope, Operator))

        if len(dofdex) > 0:
            N_copies = max(dofdex) + 1
            space = 1
            target = makeDomain((UnstructuredDomain(N_copies), target))
            if N_copies != len(dofdex):
                distributed_tgt = makeDomain((UnstructuredDomain(len(dofdex)),
                                              target[1]))
                Distributor = _Distributor(dofdex, target, distributed_tgt)
            else:
                distributed_tgt = target
        else:
            N_copies = 0
            space = 0
            distributed_tgt = target = makeDomain(target)
        myassert(isinstance(target[space], PowerSpace))

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
        # In the harmonic transform convention used here, the zero-mode scales
        # linearly with the volume while all other modes scale with the square
        # root of the volume.  However, as the `_Normalization` operator
        # introduces an additional factor of `1 / \sqrt{totvol}` for all modes
        # but the zero-mode, the modes here all apparently have the same
        # scaling factor. See the respective note in `_Normalization` for
        # details.
        vol1[1:] = vol0[0] = totvol
        vol0 = DiagonalOperator(makeField(target[space], vol0), target, space)
        vol0 = vol0(full(vol0.domain, 1))
        vol1 = DiagonalOperator(makeField(target[space], vol1), target, space)
        # End prepare constant fields

        slope = vslope @ ps_expander @ loglogavgslope
        sig_flex = vflex @ expander @ flexibility if flexibility is not None else None
        sig_asp = vasp @ expander @ asperity if asperity is not None else None
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

        if N_copies != len(dofdex):
            op = Distributor @ op
            sig_fluc = Distributor @ sig_fluc
            op = Adder(Distributor(vol0)) @ (sig_fluc * op)
            self._fluc = (_Distributor(dofdex, fluctuations.target,
                                       distributed_tgt[0]) @ fluctuations)
        else:
            op = Adder(vol0) @ (sig_fluc * op)
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
    square roots of power spectrum operators ("amplitudes") via calls to
    :func:`add_fluctuations*` that act on the targeted field subdomains.
    During creation of the :class:`CorrelatedFieldMaker`, a global
    offset from zero of the field model can be defined and an operator
    applying fluctuations around this offset is parametrized.

    The resulting correlated field model operator has a
    :class:`~nifty8.multi_domain.MultiDomain` as its domain and
    expects its input values to be univariately gaussian.

    The target of the constructed operator will be a
    :class:`~nifty8.domain_tuple.DomainTuple` containing the
    `target_subdomains` of the added fluctuations in the order of
    the `add_fluctuations` calls.

    Creation of the model operator is completed by calling the method
    :func:`finalize`, which returns the configured operator.

    An operator representing an array of correlated field models
    can be constructed by setting the `total_N` parameter of. It will
    have an :class:`~nifty8.domains.unstructured_domain.UnstructuredDomain`
    of shape `(total_N,)` prepended to its target domain and represent
    `total_N` correlated fields simulataneously.
    The degree of information sharing between the correlated field
    models can be configured via the `dofdex` parameter of
    :func:`add_fluctuations`.

    See the methods :func:`add_fluctuations*` and :func:`finalize` for
    further usage information.

    See also
    --------
    * For one power spectrum, the correlated field model has first been
      described in "Comparison of classical and Bayesian imaging in radio
      interferometry", A&A 646, A84 (2021) by P. Arras et al.
      `<https://doi.org/10.1051/0004-6361/202039258>`_
    * For multiple power spectra, it has first been used in "M87* in space,
      time and frequency", Nature Astronomy (2022), by P. Arras et al.
      `<https://doi.org/10.1038/s41550-021-01548-0>`_

    Consider citing these papers, if you use the correlated field model.
    """
    def __init__(self, prefix, total_N=0):
        """Instantiate a CorrelatedFieldMaker object.

        Parameters
        ----------
        prefix : string
            Prefix to the names of the domains of the cf operator to be made.
            This determines the names of the operator domain.
        total_N : integer, optional
            Number of field models to create.
            If not 0, the first entry of the operators target will be an
            :class:`~nifty.domains.unstructured_domain.UnstructuredDomain`
            with length `total_N`.
        """
        self._azm = None
        self._offset_mean = None
        self._a = []
        self._target_subdomains = []

        self._prefix = prefix
        self._total_N = total_N

        try:
            from .. import re as jft

            if total_N != 0:
                # warn(f"unable to add JAX operator for total_N={total_N}")
                raise ImportError("short-circuit JAX init")
            self._jax_cfm = jft.CorrelatedFieldMaker(prefix=prefix)
        except ImportError:
            self._jax_cfm = None

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
        target_subdomain : :class:`~nifty8.domain.Domain`, \
                           :class:`~nifty8.domain_tuple.DomainTuple`
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
        harmonic_partner : :class:`~nifty8.domain.Domain`, \
                           :class:`~nifty8.domain_tuple.DomainTuple`
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

        _check_dofdex(dofdex, self._total_N)

        if self._total_N > 0:
            N = max(dofdex) + 1
            target_subdomain = makeDomain((UnstructuredDomain(N), target_subdomain))
        else:
            N = 0
            target_subdomain = makeDomain(target_subdomain)
        # myassert(isinstance(target_subdomain[space], (RGSpace, HPSpace, GLSpace)))

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
                         target_subdomain[-1].total_volume,
                         pre + 'spectrum', dofdex)

        is_rg = all(isinstance(dom, RGSpace) for dom in target_subdomain)
        if self._jax_cfm is not None and (len(dofdex) > 0 or index or not is_rg):
            # warn(f"unable to add JAX operator for {target_subdomain}")
            self._jax_cfm = None
        if self._jax_cfm is not None:
            dists = tuple(e for di in target_subdomain for e in di.distances)
            self._jax_cfm.add_fluctuations(
                shape=target_subdomain.shape,
                distances=dists,
                fluctuations=fluctuations,
                loglogavgslope=loglogavgslope,
                flexibility=flexibility,
                asperity=asperity,
                prefix=str(prefix),
                harmonic_type="fourier",
                non_parametric_kind="power",
            )
            amp._jax_expr = self._jax_cfm.fluctuations[-1]

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
                                loglogslope,
                                prefix='',
                                adjust_for_volume=True,
                                harmonic_partner=None):
        """Function to add matern kernels to the field to be made.

        The matern kernel amplitude is parametrized in the following way:

        .. math ::
            A(k) = \\frac{a}{\\left(1 + { \
                \\left(\\frac{|k|}{b}\\right) \
            }^2\\right)^{-c/4}}

        where :math:`a` is the scale, :math:`b` the cutoff and :math:`c` the
        spectral index of the power spectrum.

        Parameters
        ----------
        target_subdomain : :class:`~nifty8.domains.domain.Domain`, \
                           :class:`~nifty8.domain_tuple.DomainTuple`
            Target subdomain on which the correlation structure defined
            in this call should hold.
        scale : tuple of float (mean, std)
            Overall scale of the fluctuations in the target subdomain.
            The parameter is a-priori lognormal distribution.
        cutoff : tuple of float (mean, std)
            Frequency at which the power spectrum should transition into
            a spectra following a power-law.
            The parameter is a-priori lognormal distribution.
        loglogslope : tuple of float (mean, std)
            The slope of the power-spectrum spectrum on double logarithmic
            scales, i.e. the spectral index.
            The parameter is a-priori normal distribution.
        prefix : string
            Prefix of the power spectrum parameter domain names.
        adjust_for_volume : bool, optional
            Whether to implicitly adjust the scale parameter of the Matern
            kernel and the zero-mode of the overall model for the volume in the
            target subdomain or assume them to be adjusted already.
        harmonic_partner : :class:`~nifty8.domains.domain.Domain`, \
                           :class:`~nifty8.domain_tuple.DomainTuple`
            Harmonic space in which to define the power spectrum.

        Notes
        -----
        The parameters of the amplitude model are assumed to be relative to a
        unit-less power spectrum, i.e. the parameters are assumed to be
        agnostic to changes in the volume of the target subdomain. This is in
        steep contrast to the non-parametric amplitude operator in
        :class:`~nifty8.library.correlated_fields.CorrelatedFieldMaker.add_fluctuations`.

        Up to the Matern amplitude only works for `total_N == 0`.
        """
        if self._total_N > 0:
            raise NotImplementedError()
        if harmonic_partner is None:
            harmonic_partner = target_subdomain.get_default_codomain()
        else:
            target_subdomain.check_codomain(harmonic_partner)
            harmonic_partner.check_codomain(target_subdomain)
        target_subdomain = makeDomain(target_subdomain)

        scale = LognormalTransform(*scale, self._prefix + prefix + 'scale', 0)
        prfx = self._prefix + prefix + 'cutoff'
        cutoff = LognormalTransform(*cutoff, prfx, 0)
        prfx = self._prefix + prefix + 'loglogslope'
        loglogslope = NormalTransform(*loglogslope, prfx, 0)

        totvol = 1.
        if adjust_for_volume:
            totvol = target_subdomain[-1].total_volume
        pow_spc = PowerSpace(harmonic_partner)
        amp = _AmplitudeMatern(pow_spc, scale, cutoff, loglogslope,
                               totvol)

        if self._jax_cfm is not None:
            # warn(f"unable to add JAX operator for Matern fluctuations")
            self._jax_cfm = None

        self._a.append(amp)
        self._target_subdomains.append(target_subdomain)

    def set_amplitude_total_offset(self, offset_mean, offset_std, dofdex=None):
        """Sets the zero-mode for the combined amplitude operator

        Parameters
        ----------
        offset_mean : float
            Mean offset from zero of the correlated field to be made.
        offset_std : tuple of float, instance of \
                :class:`~nifty8.operators.operator.Operator` acting on scalar \
                domain, scalar or None
            Mean standard deviation and standard deviation of the standard
            deviation of the offset. No, this is not a word duplication.
            The option to specify `None` or equivalently a scalar value of `1.`
            only really makes sense for one dimensional amplitude spectral.
            Take special care if using this option for multi-dimensional
            amplitude spectra that this is really what you want.
        dofdex : np.array of integers, optional
            An integer array specifying the zero mode models used if
            total_N > 1. It needs to have length of total_N. If total_N=3 and
            dofdex=[0,0,1], that means that two models for the zero mode are
            instantiated; the first one is used for the first and second
            field model and the second is used for the third field model.
            *If not specified*, use the same zero mode model for all
            constructed field models.
        """
        if self._offset_mean is not None and self._azm is not None:
            logger.warning("Overwriting the previous mean offset and zero-mode")

        self._offset_mean = offset_mean
        jax_offset_std = offset_std
        if offset_std is None:
            self._azm = 0.
        elif np.isscalar(offset_std) and offset_std == 1.:
            self._azm = 1.
            jax_offset_std = lambda _: 1.
        elif isinstance(offset_std, Operator):
            self._azm = offset_std
            jax_offset_std = offset_std.jax_expr
        else:
            if dofdex is None:
                dofdex = np.full(self._total_N, 0)
            elif len(dofdex) != self._total_N:
                raise ValueError("length of dofdex needs to match total_N")

            _check_dofdex(dofdex, self._total_N)

            N = max(dofdex) + 1 if self._total_N > 0 else 0
            if len(offset_std) != 2:
                te = (
                    "`offset_std` of invalid type and/or shape"
                    f"; expected a 2D tuple of floats; got '{offset_std!r}'"
                )
                raise TypeError(te)
            zm = LognormalTransform(*offset_std, self._prefix + 'zeromode', N)
            if self._total_N > 0:
                zm = _Distributor(dofdex, zm.target, UnstructuredDomain(self._total_N)) @ zm
            self._azm = zm

        if self._jax_cfm is not None and dofdex is not None and len(dofdex) > 0:
            # warn(f"unable to add JAX operator for dofdex={dofdex}")
            self._jax_cfm = None
        if self._jax_cfm is not None:
            try:
                self._jax_cfm.set_amplitude_total_offset(
                    offset_mean=offset_mean, offset_std=jax_offset_std
                )
                if not isinstance(self._azm, float):
                    self._azm._jax_expr = self._jax_cfm.azm
            except TypeError as e:
                self._jax_cfm = None
                # if isinstance(e, TypeError):
                #     warn(f"no JAX operator for this configuration;\n{e}")

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

        ht = HarmonicTransformOperator(hspace,
                                       self._target_subdomains[0][amp_space],
                                       space=spaces[0])
        for i in range(1, n_amplitudes):
            ht = HarmonicTransformOperator(ht.target,
                                           self._target_subdomains[i][amp_space],
                                           space=spaces[i]) @ ht

        a = list(self.get_normalized_amplitudes())
        for ii in range(n_amplitudes):
            co = ContractionOperator(hspace, spaces[:ii] + spaces[ii + 1:])
            pp = a[ii].target[amp_space]
            pd = PowerDistributor(co.target, pp, amp_space)
            a[ii] = co.adjoint @ pd @ a[ii]
        corr = reduce(mul, a)
        xi = ducktape(hspace, None, self._prefix + 'xi')
        if np.isscalar(self.azm):
            op = ht(corr * xi)
        else:
            expander = ContractionOperator(hspace, spaces=spaces).adjoint
            azm = expander @ self.azm
            op = ht(azm * corr * xi)

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

        if self._jax_cfm is not None:
            op._jax_expr = self._jax_cfm.finalize()
        return op

    def statistics_summary(self, prior_info):
        from ..sugar import from_random

        if prior_info == 0:
            return

        lst = []
        try:
            lst.append(('Offset amplitude', self.amplitude_total_offset))
        except NotImplementedError:  # AZM mustn't be set to get stats
            pass
        lst.append(('Total fluctuation amplitude', self.total_fluctuation))
        namps = len(self._a)
        if namps > 1:
            for ii in range(namps):
                lst.append(('Average fluctuation (space {})'.format(ii),
                            self.average_fluctuation(ii)))
                try:
                    lst.append(('Slice fluctuation (space {})'.format(ii),
                                self.slice_fluctuation(ii)))
                except NotImplementedError:  # AZM mustn't be set to get stats
                    pass

        for kk, op in lst:
            sc = StatCalculator()
            for _ in range(prior_info):
                sc.add(op(from_random(op.domain, 'normal')))
            mean = sc.mean.val
            stddev = sc.var.ptw("sqrt").val
            for m, s in zip(mean.flatten(), stddev.flatten()):
                logger.info('{}: {:.02E} ± {:.02E}'.format(kk, m, s))

    @property
    def fluctuations(self):
        """Returns the added fluctuations operators used in the model"""
        return tuple(self._a)

    def get_normalized_amplitudes(self):
        """Returns the normalized amplitude operators used in the final model

        The amplitude operators are corrected for the otherwise degenerate
        zero-mode. Their scales are only meaningful relative to one another.
        Their absolute scale bares no information.

        Notes
        -----
        In the case of no zero-mode, i.e. an assumed zero-mode of unity, this
        call is equivalent to the `fluctuations` property.
        """
        if self._azm == 0:
            if not len(self.fluctuations) == 1:
                raise RuntimeError("Zeromode can not be disabled for product spectra")
            sp = self.fluctuations[0].target
            maskzm = np.ones(self.fluctuations[0].target.shape)
            maskzm[0] = 0
            maskzm = makeOp(makeField(sp, maskzm))
            a = [maskzm @ self.fluctuations[0]]
            return tuple(a)
        elif self.azm == 1:
            return self.fluctuations

        if self._jax_cfm:
            normed_amps_jax = self._jax_cfm.get_normalized_amplitudes()
        else:
            normed_amps_jax = (None, ) * len(self._a)

        normal_amp = []
        for amp, na_jax in zip(self._a, normed_amps_jax):
            a_target = amp.target
            a_space = 0 if not hasattr(amp, "_space") else amp._space
            a_pp = amp.target[a_space]
            myassert(isinstance(a_pp, PowerSpace))

            azm_expander = ContractionOperator(
                a_target, spaces=a_space
            ).adjoint
            zm_unmask, zm_mask = [np.zeros(a_pp.shape) for _ in range(2)]
            zm_mask[1:] = zm_unmask[0] = 1.
            zm_mask = DiagonalOperator(
                makeField(a_pp, zm_mask), a_target, a_space
            )
            zm_unmask = DiagonalOperator(
                makeField(a_pp, zm_unmask), a_target, a_space
            )
            zm_unmask = Adder(zm_unmask(full(zm_unmask.domain, 1)))

            zm_normalization = zm_unmask @ (
                zm_mask @ azm_expander(self.azm.ptw("reciprocal"))
            )
            na = zm_normalization * amp
            na._jax_expr = na_jax
            normal_amp.append(na)
        return tuple(normal_amp)

    @property
    def amplitude(self):
        if len(self._a) > 1:
            s = ('If more than one spectrum is present in the model,',
                 ' no unique set of amplitudes exist because only the',
                 ' relative scale is determined.')
            raise NotImplementedError(s)
        normal_amp = self.get_normalized_amplitudes()[0]

        if np.isscalar(self.azm):
            na = normal_amp
        else:
            expand = ContractionOperator(
                normal_amp.target, len(normal_amp.target) - 1
            ).adjoint
            na = normal_amp * (expand @ self.azm)

        if self._jax_cfm:
            na._jax_expr = self._jax_cfm.amplitude
        return na

    @property
    def power_spectrum(self):
        return self.amplitude**2

    @property
    def amplitude_total_offset(self):
        """Returns the total offset of the amplitudes"""
        if self._azm is None:
            nie = "You need to set the `amplitude_total_offset` first"
            raise NotImplementedError(nie)
        return self._azm

    @property
    def azm(self):
        """Alias for `amplitude_total_offset`"""
        return self.amplitude_total_offset

    def moment_slice_to_average(self, fluctuations_slice_mean, nsamples=1000):
        """Translates the slice fluctuations into average flucutations to
        use single space results in multi-space setups.

        This method allows to use single-space reconstruction results to set
        the hyperparameters in multi-space settings. Given the results of a
        reconstruction in a single space setting (say for example an image at a
        specific frequency or a specific moment in time), it is possible to use
        the fluctuations of these results to determine the fluctuations of this
        (sub-)space in a multi-space setup (e.g. when reconstructing a
        collection of images over a frequency range or in a time interval). To
        do so, the single-space fluctuations have to be translated to match
        their multi-space counterparts, since the single-space results represent
        a slice of the multi-space setting. The fluctuations in the multi-space
        setting represent average fluctuations (i.E. the variability that
        remains when integrating over all other spaces) and therefore the slice
        fluctuations have to be rescaled. After all new sub-spaces (say time
        and/or frequency) have been added to the model, this method can be used
        to obtain the mean of `fluctuations` of the last sub-space from the
        fluctuations given from the single space result. Note that to properly
        use this method it should be called only after all other sub-spaces
        (time/frequency) as well as the `amplitude_total_offset` have been set
        in the `CorrelatedFieldMaker` (see example below).

        Parameters
        ----------
        fluctuations_slice_mean : float
            Mean fluctuations of the single space reconstruction that is a slice
            of this multi-space setting.
        nsamples : int, optional
            Number of samples used internally to estimate the rescaling of the
            fluctuations. Default is 1000.

        Returns
        -------
        out : float
            Mean of the average fluctuations that can be used as an input to add
            the final sub-space matching the space used for the slice case.

        Examples
        --------
        >>> slice_fluct = ... # Fluctuations obtained from the single-space run
        >>> cf = ift.CorrelatedFieldMaker(...) # The cf of the multi-space case
        >>> cf.add_fluctuations(...) # Add a sub-space (e.g. frequency)
        >>> cf.add_fluctuations(**freq_params) # An optional second space (time)
        >>> cf.set_amplitude_total_offset(...) # Set zero mode of the spectrum
        >>> avg_fluct = cf.moment_slice_to_average(slice_fluct)
        >>> cf.add_fluctuations(fluctuations=(avg_fluct, ...), ...)
        >>> cf.finalize()
        """
        fluctuations_slice_mean = float(fluctuations_slice_mean)
        if not fluctuations_slice_mean > 0:
            msg = "fluctuations_slice_mean must be greater zero; got {!r}"
            raise ValueError(msg.format(fluctuations_slice_mean))
        from ..sugar import from_random
        scm = 1.
        for a in self._a:
            op = a.fluctuation_amplitude * self.azm.ptw("reciprocal")
            res = np.array([op(from_random(op.domain, 'normal')).val
                            for _ in range(nsamples)])
            scm *= res**2 + 1.
        return fluctuations_slice_mean / np.mean(np.sqrt(scm))

    @property
    def total_fluctuation(self):
        """Returns operator which acts on prior or posterior samples"""
        if len(self._a) == 0:
            raise NotImplementedError
        if len(self._a) == 1:
            return self.average_fluctuation(0)
        q = 1.
        for a in self._a:
            fl = a.fluctuation_amplitude*self.azm.ptw("reciprocal")
            q = q*(Adder(full(fl.target, 1.)) @ fl**2)
        return (Adder(full(q.target, -1.)) @ q).ptw("sqrt")*self.azm

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
            fl = self._a[j].fluctuation_amplitude*self.azm.ptw("reciprocal")
            if j == space:
                q = q*fl**2
            else:
                q = q*(Adder(full(fl.target, 1.)) @ fl**2)
        return q.ptw("sqrt")*self.azm

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


def _check_dofdex(dofdex, total_N):
    if not (list(dofdex) == list(range(total_N)) or list(dofdex) == total_N*[0]):
        warn("In the upcoming release only dofdex==range(total_N) or dofdex==total_N*[0] "
             f"will be supported. You can use dofdex={dofdex}.\n"
             "Please report at `c@philipp-arras.de` if you use this "
             "feature and would like to see it continued.", DeprecationWarning)
