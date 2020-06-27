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


import numpy as np

from ..domains.power_space import PowerSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..operators.adder import Adder
from ..operators.contraction_operator import ContractionOperator
from ..operators.distributors import PowerDistributor
from ..operators.harmonic_operators import HarmonicTransformOperator
from ..operators.linear_operator import LinearOperator
from ..operators.normal_operators import LognormalTransform, NormalTransform
from ..operators.operator import Operator
from ..operators.simple_linear_operators import ducktape
from ..sugar import full, makeDomain, makeField, makeOp
from .correlated_fields import (_log_vol, _Normalization,
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


class SimpleCorrelatedField(Operator):
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

        tgt = PowerSpace(harmonic_partner)
        azm_expander = ContractionOperator(tgt, 0).adjoint
        twolog = _SimpleTwoLogIntegrations(tgt)
        dom = twolog.domain[0]
        vflex = np.zeros(dom.shape)
        vasp = np.zeros(dom.shape, dtype=np.float64)
        shift = np.ones(dom.shape, dtype=np.float64)
        vol0 = np.zeros(tgt.shape, dtype=np.float64)
        vol1 = np.zeros(tgt.shape, dtype=np.float64)
        vflex[0] = vflex[1] = np.sqrt(_log_vol(tgt))
        vasp[0] = 1
        shift[0] = _log_vol(tgt)**2/12.
        vol1[1:] = vol0[0] = target.total_volume
        vflex = makeOp(makeField(dom, vflex))
        vasp = makeOp(makeField(dom, vasp))
        shift = makeOp(makeField(dom, shift))
        vol0 = makeField(tgt, vol0)
        vol1 = makeOp(makeField(tgt, vol1))
        vslope = makeOp(makeField(tgt, _relative_log_k_lengths(tgt)))
        shift = shift(full(shift.domain, 1))
        expander = ContractionOperator(twolog.domain, spaces=0).adjoint
        ps_expander = ContractionOperator(twolog.target, spaces=0).adjoint
        slope = vslope @ ps_expander @ avgsl
        sig_flex = vflex @ expander @ flex
        sig_asp = vasp @ expander @ asp
        sig_fluc = vol1 @ ps_expander @ fluct
        sig_fluc = vol1 @ ps_expander @ fluct
        xi = ducktape(dom, None, prefix + 'spectrum')
        sigma = sig_flex*(Adder(shift) @ sig_asp).ptw("sqrt")
        smooth = _SlopeRemover(tgt, 0) @ twolog @ (sigma*xi)
        op = _Normalization(tgt, 0) @ (slope + smooth)
        amp = Adder(vol0) @ (sig_fluc*(azm_expander @ zm.ptw("reciprocal"))*op)

        ht = HarmonicTransformOperator(harmonic_partner, target)
        pd = PowerDistributor(harmonic_partner, amp.target[0])
        expander = ContractionOperator(harmonic_partner, spaces=0).adjoint
        xi = ducktape(harmonic_partner, None, prefix + 'xi')
        op = ht(expander(zm)*pd(amp)*xi)
        if offset_mean is not None:
            op = Adder(full(op.target, float(offset_mean))) @ op
        self.apply = op.apply
        self._domain = op.domain
        self._target = op.target
