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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function

from ..compat import *
from ..domain_tuple import DomainTuple
from ..multi_field import MultiField
from ..multi_domain import MultiDomain
from ..operators.domain_distributor import DomainDistributor
from ..operators.harmonic_operators import HarmonicTransformOperator
from ..operators.power_distributor import PowerDistributor
from ..operators.operator import Operator
from ..operators.simple_linear_operators import FieldAdapter


def CorrelatedField(s_space, amplitude_model):
    '''
    Function for construction of correlated fields

    Parameters
    ----------
    s_space : Field domain

    amplitude_model : model for correlation structure
    '''
    h_space = s_space.get_default_codomain()
    ht = HarmonicTransformOperator(h_space, s_space)
    p_space = amplitude_model.target[0]
    power_distributor = PowerDistributor(h_space, p_space)
    A = power_distributor(amplitude_model)
    domain = MultiDomain.union(
        (amplitude_model.domain, MultiDomain.make({"xi": h_space})))
    return ht(A*FieldAdapter(domain, "xi"))


# def make_mf_correlated_field(s_space_spatial, s_space_energy,
#                              amplitude_model_spatial, amplitude_model_energy):
#     '''
#     Method for construction of correlated multi-frequency fields
#     '''
#     h_space_spatial = s_space_spatial.get_default_codomain()
#     h_space_energy = s_space_energy.get_default_codomain()
#     h_space = DomainTuple.make((h_space_spatial, h_space_energy))
#     ht1 = HarmonicTransformOperator(h_space, space=0)
#     ht2 = HarmonicTransformOperator(ht1.target, space=1)
#     ht = ht2(ht1)
#
#     p_space_spatial = amplitude_model_spatial.target[0]
#     p_space_energy = amplitude_model_energy.target[0]
#
#     pd_spatial = PowerDistributor(h_space, p_space_spatial, 0)
#     pd_energy = PowerDistributor(pd_spatial.domain, p_space_energy, 1)
#     pd = pd_spatial(pd_energy)
#
#     dom_distr_spatial = DomainDistributor(pd.domain, 0)
#     dom_distr_energy = DomainDistributor(pd.domain, 1)
#
#     a_spatial = dom_distr_spatial(amplitude_model_spatial)
#     a_energy = dom_distr_energy(amplitude_model_energy)
#     a = a_spatial*a_energy
#     A = pd(a)
#
#     domain = MultiDomain.union(
#         (amplitude_model_spatial.domain, amplitude_model_energy.domain,
#          MultiDomain.make({"xi": h_space})))
#     return exp(ht(A*FieldAdapter(domain, "xi")))
