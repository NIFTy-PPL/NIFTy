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
from ..operators.adder import Adder
from ..operators.contraction_operator import ContractionOperator
from ..operators.distributors import PowerDistributor
from ..operators.harmonic_operators import HarmonicTransformOperator
from ..operators.normal_operators import LognormalTransform, NormalTransform
from .correlated_fields import _TwoLogIntegrations
from ..operators.operator import Operator
from ..operators.simple_linear_operators import ducktape
from ..operators.value_inserter import ValueInserter
from ..sugar import full, makeField, makeOp
from .correlated_fields import (_log_vol, _Normalization,
                                _relative_log_k_lengths, _SlopeRemover)


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

        pspace = PowerSpace(harmonic_partner)
        twolog = _TwoLogIntegrations(pspace)
        dom = twolog.domain[0]
        vflex = np.zeros(dom.shape)
        vasp = np.zeros(dom.shape)
        shift = np.ones(dom.shape)
        vflex[0] = vflex[1] = np.sqrt(_log_vol(pspace))
        vasp[0] = 1
        shift[0] = _log_vol(pspace)**2/12.
        vflex = makeOp(makeField(dom, vflex))
        vasp = makeOp(makeField(dom, vasp))
        shift = makeField(dom, shift)
        vslope = makeOp(makeField(pspace, _relative_log_k_lengths(pspace)))

        expander = ContractionOperator(twolog.domain, 0).adjoint
        ps_expander = ContractionOperator(pspace, 0).adjoint
        slope = vslope @ ps_expander @ avgsl
        sig_flex = vflex @ expander @ flex
        sig_asp = vasp @ expander @ asp
        xi = ducktape(dom, None, prefix + 'spectrum')
        smooth = xi*sig_flex*(Adder(shift) @ sig_asp).ptw("sqrt")
        smooth = _SlopeRemover(pspace, 0) @ twolog @ smooth
        a = _Normalization(pspace, 0) @ (slope + smooth)

        maskzm = np.ones(pspace.shape)
        maskzm[0] = 0
        maskzm = makeOp(makeField(pspace, maskzm))
        insert = ValueInserter(pspace, (0,))
        a = (maskzm @ ((ps_expander @ fluct)*a)) + insert(zm)
        self._a = a.scale(target.total_volume)

        ht = HarmonicTransformOperator(harmonic_partner, target)
        pd = PowerDistributor(harmonic_partner, pspace)
        xi = ducktape(harmonic_partner, None, prefix + 'xi')
        op = ht(pd(self._a)*xi)
        if offset_mean is not None:
            op = Adder(full(op.target, float(offset_mean))) @ op
        self.apply = op.apply
        self._domain = op.domain
        self._target = op.target

    @property
    def amplitude(self):
        return self._a
