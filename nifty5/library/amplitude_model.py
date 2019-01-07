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
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..compat import *
from ..domains.power_space import PowerSpace
from ..field import Field
from ..sugar import makeOp, sqrt


def _ceps_kernel(dof_space, k, a, k0):
    return a**2/(1 + (k/k0)**2)**2


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
    shape = domain.shape

    q_array = domain.get_k_array()

    # Fill cepstrum field (all non-zero modes)
    no_zero_modes = (slice(1, None),)*dim
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


def CepstrumOperator(logk_space, ceps_a, ceps_k, zero_mode=True):
    '''
    Parameters
    ----------
    ceps_a, ceps_k0 : Smoothness parameters in ceps_kernel
                        eg. ceps_kernel(k) = (a/(1+(k/k0)**2))**2
                        a = ceps_a,  k0 = ceps_k0
    '''

    from ..operators.qht_operator import QHTOperator
    from ..operators.symmetrizing_operator import SymmetrizingOperator
    qht = QHTOperator(target=logk_space)
    dof_space = qht.domain[0]
    sym = SymmetrizingOperator(logk_space)
    kern = lambda k: _ceps_kernel(dof_space, k, ceps_a, ceps_k)
    cepstrum = create_cepstrum_amplitude_field(dof_space, kern)
    res = sym(qht(makeOp(sqrt(cepstrum))))
    if not zero_mode:
        shp = res.target.shape
        mask = np.ones(shp)
        mask[(0,)*len(shp)] = 0.
        mask = makeOp(Field.from_global_data(res.target, mask))
        res = mask(res)
    return res


def SlopeModel(logk_space, sm, sv, im, iv):
    '''
    Parameters
    ----------

    sm, sv : slope_mean = expected exponent of power law (e.g. -4),
                slope_variance (default=1)

    im, iv : y-intercept_mean, y-intercept_std  of power_slope
    '''

    from ..operators.slope_operator import SlopeOperator
    from ..operators.offset_operator import OffsetOperator
    phi_mean = np.array([sm, im + sm*logk_space.t_0[0]])
    phi_sig = np.array([sv, iv])
    slope = SlopeOperator(logk_space)
    phi_mean = Field.from_global_data(slope.domain, phi_mean)
    phi_sig = Field.from_global_data(slope.domain, phi_sig)
    return slope(OffsetOperator(phi_mean)(makeOp(phi_sig)))


def AmplitudeModel(s_space, Npixdof, ceps_a, ceps_k, sm, sv, im, iv,
                   keys=['tau', 'phi'], zero_mode=True):
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

    from ..operators.exp_transform import ExpTransform
    from ..operators.scaling_operator import ScalingOperator

    h_space = s_space.get_default_codomain()
    et = ExpTransform(PowerSpace(h_space), Npixdof)
    logk_space = et.domain[0]

    smooth = CepstrumOperator(logk_space, ceps_a, ceps_k, zero_mode)
    smooth = smooth.ducktape(keys[0])
    linear = SlopeModel(logk_space, sm, sv, im, iv)
    linear = linear.ducktape(keys[1])

    fac = ScalingOperator(0.5, smooth.target)
    return et((fac(smooth + linear)).exp())
