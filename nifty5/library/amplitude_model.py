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

import numpy as np

from ..compat import *
from ..domains.power_space import PowerSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..multi_field import MultiField
from ..multi_domain import MultiDomain
from ..sugar import makeOp, sqrt
from ..operators.operator import Operator


def _ceps_kernel(dof_space, k, a, k0):
    return a**2/(1+(k/(k0*dof_space.bindistances[0]))**2)**2


def create_cepstrum_amplitude_field(domain, cepstrum):
    """Creates a ...
    Writes the sum of all modes into the zero-mode.

    Parameters
    ----------
    domain: ???
        ???
    cepstrum: Callable
        ???
    """

    dim = len(domain.shape)
    dist = domain.bindistances
    shape = domain.shape

    # Prepare q_array
    q_array = np.zeros((dim,) + shape)
    if dim == 1:
        ks = domain.get_k_length_array().to_global_data()
        q_array = np.array([ks])
    else:
        for i in range(dim):
            ks = np.minimum(shape[i] - np.arange(shape[i]) +
                            1, np.arange(shape[i])) * dist[i]
            q_array[i] += ks.reshape((1,)*i + (shape[i],) + (1,)*(dim-i-1))

    # Fill cepstrum field (all non-zero modes)
    no_zero_modes = (slice(1, None),) * dim
    ks = q_array[(slice(None),) + no_zero_modes]
    cepstrum_field = np.zeros(shape)
    cepstrum_field[no_zero_modes] = cepstrum(ks)

    # Fill cepstrum field (zero-mode subspaces)
    for i in range(dim):
        # Prepare indices
        fst_dims = (slice(None),)*i
        sl = fst_dims + (slice(1, None),)
        sl2 = fst_dims + (0,)

        # Do summation
        cepstrum_field[sl2] = np.sum(cepstrum_field[sl], axis=i)

    return Field.from_global_data(domain, cepstrum_field)


class AmplitudeModel(Operator):
    '''
    Computes a smooth power spectrum.
    Output lives in PowerSpace.

    Parameters
    ----------

    Npixdof : #pix in dof_space

    ceps_a, ceps_k0 : Smoothness parameters in ceps_kernel
                        eg. ceps_kernel(k) = (a/(1+(k/k0)**2))**2
                        a = ceps_a,  k0 = ceps_k0

    sm, sv : slope_mean = expected exponent of power law (e.g. -4),
                slope_variance (default=1)

    im, iv : y-intercept_mean, y-intercept_variance  of power_slope
    '''
    def __init__(self, s_space, Npixdof, ceps_a, ceps_k, sm, sv, im, iv,
                 keys=['tau', 'phi']):
        from ..operators.exp_transform import ExpTransform
        from ..operators.qht_operator import QHTOperator
        from ..operators.slope_operator import SlopeOperator
        from ..operators.symmetrizing_operator import SymmetrizingOperator

        h_space = s_space.get_default_codomain()
        p_space = PowerSpace(h_space)
        self._exp_transform = ExpTransform(p_space, Npixdof)
        logk_space = self._exp_transform.domain[0]
        qht = QHTOperator(target=logk_space)
        dof_space = qht.domain[0]
        param_space = UnstructuredDomain(2)
        sym = SymmetrizingOperator(logk_space)

        phi_mean = np.array([sm, im])
        phi_sig = np.array([sv, iv])

        self._slope = SlopeOperator(param_space, logk_space, phi_sig)
        self._norm_phi_mean = Field.from_global_data(param_space,
                                                     phi_mean/phi_sig)

        self._domain = MultiDomain.make({keys[0]: dof_space,
                                         keys[1]: param_space})
        self._target = self._exp_transform.target

        kern = lambda k: _ceps_kernel(dof_space, k, ceps_a, ceps_k)
        cepstrum = create_cepstrum_amplitude_field(dof_space, kern)

        ceps = makeOp(sqrt(cepstrum))
        self._smooth_op = sym(qht(ceps))
        self._keys = tuple(keys)

    def apply(self, x):
        smooth_spec = self._smooth_op(x[self._keys[0]])
        phi = x[self._keys[1]] + self._norm_phi_mean
        linear_spec = self._slope(phi)
        loglog_spec = smooth_spec + linear_spec
        return self._exp_transform((0.5*loglog_spec).exp())
