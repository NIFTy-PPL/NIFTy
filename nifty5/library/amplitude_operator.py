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

from ..domains.power_space import PowerSpace
from ..field import Field
from ..sugar import makeOp


def _ceps_kernel(dof_space, k, a, k0):
    return a**2/(1 + (k/k0)**2)**2


def _create_cepstrum_amplitude_field(domain, cepstrum):
    dim = len(domain.shape)
    shape = domain.shape
    q_array = domain.get_k_array()

    # Fill all non-zero modes
    no_zero_modes = (slice(1, None), )*dim
    ks = q_array[(slice(None), ) + no_zero_modes]
    cepstrum_field = np.zeros(shape)
    cepstrum_field[no_zero_modes] = cepstrum(ks)

    # Fill zero-mode subspaces
    for i in range(dim):
        fst_dims = (slice(None), )*i
        sl = fst_dims + (slice(1, None), )
        sl2 = fst_dims + (0, )
        cepstrum_field[sl2] = np.sum(cepstrum_field[sl], axis=i)
    return Field.from_global_data(domain, cepstrum_field)


def CepstrumOperator(domain, a, k0):
    '''
    .. math::
        C(k) = \\left(\\frac{a}{1+(k/k0)^2}\\right)^2
    '''
    from ..operators.qht_operator import QHTOperator
    from ..operators.symmetrizing_operator import SymmetrizingOperator

    # FIXME a>0 k0>0
    qht = QHTOperator(target=domain)
    dof_space = qht.domain[0]
    sym = SymmetrizingOperator(domain)
    kern = lambda k: _ceps_kernel(dof_space, k, a, k0)
    cepstrum = _create_cepstrum_amplitude_field(dof_space, kern)
    return sym @ qht @ makeOp(cepstrum.sqrt())


def SlopeOperator(domain, sm, sv, im, iv):
    '''
    Parameters
    ----------

    sm, sv : slope_mean = expected exponent of power law (e.g. -4),
                slope_variance (default=1)

    im, iv : y-intercept_mean, y-intercept_std  of power_slope

    '''
    from ..operators.slope_operator import SlopeOperator
    from ..operators.offset_operator import OffsetOperator

    # sv, iv>0

    phi_mean = np.array([sm, im + sm*domain.t_0[0]])
    phi_sig = np.array([sv, iv])
    slope = SlopeOperator(domain)
    phi_mean = Field.from_global_data(slope.domain, phi_mean)
    phi_sig = Field.from_global_data(slope.domain, phi_sig)
    return slope(OffsetOperator(phi_mean)(makeOp(phi_sig)))


def AmplitudeOperator(
        target, n_pix, a, k0, sm, sv, im, iv, keys=['tau', 'phi']):
    '''Operator for parametrizing smooth power spectra.

    The general guideline for setting up generative models in IFT is to
    transform the problem into the eigenbase of the prior and formulate the
    generative model in this base. This is done here for the case of a power
    spectrum which is smooth and has a linear component (both on
    double-logarithmic scale).

    This function assembles an :class:`Operator` which maps two a-priori white
    Gaussian random fields to a smooth power spectrum which is composed out of
    a linear and a smooth component.

    On double-logarithmic scale, i.e. both x and y-axis on logarithmic scale,
    the output of the generated operator is:

        AmplitudeOperator = 0.5*(smooth_component + linear_component)

    This is then exponentiated and exponentially binned (via a :class:`ExpTransform`).

    The prior on the linear component is parametrized by four real numbers,
    being expected value and prior variance on the slope and the y-intercept
    of the linear function.

    The prior on the smooth component is parametrized by two real numbers: the
    strength and the cutoff of the smoothness prior (see :class:`CepstrumOperator`).

    Parameters
    ----------
    n_pix : int
        Number of pixels of the space in which the .
    target : PowerSpace
        Target of the Operator.
    a : float
        Strength of smoothness prior (see :class:`CepstrumOperator`).
    k0 : float
        Cutoff of smothness prior in quefrency space (see :class:`CepstrumOperator`).
    sm : float
        Expected exponent of power law (see :class:`SlopeOperator`).
    sv : float
        Prior variance of exponent of power law (see :class:`SlopeOperator`).
    im : float
        Expected y-intercept of power law (see :class:`SlopeOperator`).
    iv : float
        Prior variance of y-intercept of power law (see :class:`SlopeOperator`).

    Returns
    -------
    Operator
        Operator which is defined on the space of white excitations fields and
        which returns on its target a power spectrum which consists out of a
        smooth and a linear part.
    '''
    from ..operators.exp_transform import ExpTransform

    if not (isinstance(n_pix, int) and isinstance(target, PowerSpace)):
        raise TypeError

    a, k0 = float(a), float(k0)
    sm, sv, im, iv = float(sm), float(sv), float(im), float(iv)

    et = ExpTransform(target, n_pix)
    dom = et.domain[0]

    dct = {'a': a, 'k0': k0}
    smooth = CepstrumOperator(dom, **dct).ducktape(keys[0])

    dct = {'sm': sm, 'sv': sv, 'im': im, 'iv': iv}
    linear = SlopeOperator(dom, **dct).ducktape(keys[1])

    return et((0.5*(smooth + linear)).exp())
