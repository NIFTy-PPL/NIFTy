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
from ..operators.operator import Operator
from ..operators.simple_linear_operators import ducktape
from ..operators.normal_operators import NormalTransform, LognormalTransform
from ..probing import StatCalculator
from ..sugar import full, makeDomain, makeField, makeOp


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
        from .correlated_fields import _Amplitude
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
        amp = _Amplitude(PowerSpace(harmonic_partner), fluct, flex, asp, avgsl,
                         zm, target.total_volume, prefix + 'spectrum', [])
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
