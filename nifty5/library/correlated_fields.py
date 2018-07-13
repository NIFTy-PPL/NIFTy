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
from ..field import Field
from ..models.local_nonlinearity import PointwiseExponential
from ..models.variable import Variable
from ..multi.multi_field import MultiField
from ..operators.domain_distributor import DomainDistributor
from ..operators.fft_operator import FFTOperator
from ..operators.harmonic_transform_operator import HarmonicTransformOperator
from ..operators.power_distributor import PowerDistributor


def make_correlated_field(s_space, amplitude_model):
    '''
    Method for construction of correlated fields

    Parameters
    ----------
    s_space : Field domain

    amplitude_model : model for correlation structure
    '''
    h_space = s_space.get_default_codomain()
    ht = FFTOperator(h_space, s_space)
    p_space = amplitude_model.value.domain[0]
    power_distributor = PowerDistributor(h_space, p_space)

    position = MultiField.from_dict({'xi': Field.full(h_space, 0.)})
    xi = Variable(position)['xi']

    A = power_distributor(amplitude_model)
    correlated_field_h = A * xi
    correlated_field = ht(correlated_field_h)
    internals = {'correlated_field_h': correlated_field_h,
                 'power_distributor': power_distributor,
                 'ht': ht}
    return correlated_field, internals


def make_mf_correlated_field(s_space_spatial, s_space_energy,
                             amplitude_model_spatial, amplitude_model_energy):
    '''
    Method for construction of correlated multi-frequency fields
    '''
    h_space_spatial = s_space_spatial.get_default_codomain()
    h_space_energy = s_space_energy.get_default_codomain()
    h_space = DomainTuple.make((h_space_spatial, h_space_energy))
    ht1 = HarmonicTransformOperator(h_space, space=0)
    ht2 = HarmonicTransformOperator(ht1.target, space=1)
    ht = ht2*ht1

    p_space_spatial = amplitude_model_spatial.value.domain[0]
    p_space_energy = amplitude_model_energy.value.domain[0]

    pd_spatial = PowerDistributor(h_space, p_space_spatial, 0)
    pd_energy = PowerDistributor(pd_spatial.domain, p_space_energy, 1)
    pd = pd_spatial*pd_energy

    dom_distr_0 = DomainDistributor(pd.domain, 0)
    dom_distr_1 = DomainDistributor(pd.domain, 1)
    a_spatial = dom_distr_1(amplitude_model_spatial)
    a_energy = dom_distr_0(amplitude_model_energy)
    a = a_spatial*a_energy
    A = pd(a)

    position = MultiField.from_dict(
        {'xi': Field.from_random('normal', h_space)})
    xi = Variable(position)['xi']
    correlated_field_h = A*xi
    correlated_field = ht(correlated_field_h)
    return PointwiseExponential(correlated_field)
