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
# Authors: Philipp Arras
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
from ..operators.normal_operators import LognormalTransform, NormalTransform
from ..operators.operator import Operator
from ..operators.simple_linear_operators import ducktape
from ..probing import StatCalculator
from ..sugar import full, makeDomain, makeField, makeOp
from .correlated_fields import (_Distributor, _log_vol, _Normalization,
                                _relative_log_k_lengths, _SlopeRemover)


class _SimpleTwoLogIntegrations(LinearOperator):
    def __init__(self, target):
        self._target = makeDomain(target)
        assert len(self._target) == 1
        tgt = self._target[0]
        assert isinstance(tgt, PowerSpace)
        self._domain = makeDomain(UnstructuredDomain((2, tgt.shape[0] - 2)))
        self._log_vol = _log_vol(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        from_third = slice(2, None)
        no_border = slice(1, -1)
        reverse = slice(None, None, -1)
        if mode == self.TIMES:
            x = x.val
            res = np.empty(self._target.shape)
            res[0] = res[1] = 0
            res[from_third] = np.cumsum(x[1])
            res[from_third] = (res[from_third] +
                               res[no_border])/2*self._log_vol + x[0]
            res[from_third] = np.cumsum(res[from_third])
        else:
            x = x.val_rw()
            res = np.zeros(self._domain.shape)
            x[from_third] = np.cumsum(x[from_third][reverse])[reverse]
            res[0] += x[from_third]
            x[from_third] *= self._log_vol/2.
            x[no_border] += x[from_third]
            res[1] += np.cumsum(x[from_third][reverse])[reverse]
        return makeField(self._tgt(mode), res)


class _SimpleAmplitude(Operator):
    def __init__(self, target, fluctuations, flexibility, asperity,
                 loglogavgslope, azm, totvol, key):
        assert isinstance(fluctuations, Operator)
        assert isinstance(flexibility, Operator)
        assert isinstance(asperity, Operator)
        assert isinstance(loglogavgslope, Operator)

        distributed_tgt = target = makeDomain(target)
        azm_expander = ContractionOperator(distributed_tgt, spaces=0).adjoint
        assert isinstance(target[0], PowerSpace)

        twolog = _SimpleTwoLogIntegrations(target)
        dom = twolog.domain

        shp = dom[0].shape
        expander = ContractionOperator(dom, spaces=0).adjoint
        ps_expander = ContractionOperator(twolog.target, spaces=0).adjoint

        # Prepare constant fields
        foo = np.zeros(shp)
        foo[0] = foo[1] = np.sqrt(_log_vol(target[0]))
        vflex = DiagonalOperator(makeField(dom[0], foo), dom, 0)

        foo = np.zeros(shp, dtype=np.float64)
        foo[0] += 1
        vasp = DiagonalOperator(makeField(dom[0], foo), dom, 0)

        foo = np.ones(shp)
        foo[0] = _log_vol(target[0])**2/12.
        shift = DiagonalOperator(makeField(dom[0], foo), dom, 0)

        vslope = DiagonalOperator(
            makeField(target[0], _relative_log_k_lengths(target[0])), target,
            0)

        foo, bar = [np.zeros(target[0].shape) for _ in range(2)]
        bar[1:] = foo[0] = totvol
        vol0, vol1 = [
            DiagonalOperator(makeField(target[0], aa), target, 0)
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
        smooth = _SlopeRemover(target, 0) @ twolog @ (sigma*xi)
        op = _Normalization(target, 0) @ (slope + smooth)
        op = Adder(vol0) @ (sig_fluc*(azm_expander @ azm.ptw("reciprocal"))*op)
        self.apply = op.apply
        self._domain, self._target = op.domain, op.target
        self._op = op

    def __repr__(self):
        return self._op.__repr__


class SimpleCorrelatedFieldMaker:
    def __init__(self,
                 target,
                 offset_mean,
                 offset_std_mean,
                 offset_std_std,
                 fluctuations_mean,
                 fluctuations_stddev,
                 flexibility_mean,
                 flexibility_stddev,
                 asperity_mean,
                 asperity_stddev,
                 loglogavgslope_mean,
                 loglogavgslope_stddev,
                 prefix='',
                 harmonic_partner=None):
        if harmonic_partner is None:
            harmonic_partner = target.get_default_codomain()
        else:
            target.check_codomain(harmonic_partner)
            harmonic_partner.check_codomain(target)
        fluct = LognormalTransform(fluctuations_mean, fluctuations_stddev,
                                   prefix + 'fluctuations', 0)
        flex = LognormalTransform(flexibility_mean, flexibility_stddev,
                                  prefix + 'flexibility', 0)
        asp = LognormalTransform(asperity_mean, asperity_stddev,
                                 prefix + 'asperity', 0)
        avgsl = NormalTransform(loglogavgslope_mean, loglogavgslope_stddev,
                                prefix + 'loglogavgslope', 0)
        zm = LognormalTransform(offset_std_mean, offset_std_std,
                                prefix + 'zeromode', 0)
        amp = _SimpleAmplitude(PowerSpace(harmonic_partner), fluct, flex, asp,
                               avgsl, zm, target.total_volume,
                               prefix + 'spectrum')
        ht = HarmonicTransformOperator(harmonic_partner, target)
        pd = PowerDistributor(harmonic_partner, amp.target[0])
        expander = ContractionOperator(harmonic_partner, spaces=0).adjoint
        self._op = ht(
            expander(zm)*pd(amp)*
            ducktape(harmonic_partner, None, prefix + 'xi'))
        if offset_mean is not None:
            self._op = Adder(full(self._op.target,
                                  float(offset_mean))) @ self._op

    def finalize(self):
        return self._op
