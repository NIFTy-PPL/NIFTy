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

from ..domain_tuple import DomainTuple
from ..domains.power_space import PowerSpace
from ..field import Field
from ..operators.adder import Adder
from ..operators.exp_transform import ExpTransform
from ..operators.qht_operator import QHTOperator
from ..operators.slope_operator import SlopeOperator
from ..operators.symmetrizing_operator import SymmetrizingOperator
from ..utilities import infer_space
from ..sugar import makeOp


def _parameter_shaper(p, shape):
    p = np.array(p)
    if p.shape is shape:
        return np.asfarray(p)
    elif p.shape is () or (1,):
        return np.full(shape, p, dtype=np.float)
    else:
        raise TypeError("Shape of parameters cannot be interpreted")


def _ceps_kernel(k, a, k0):
    return (a/(1 + np.sum((k/k0)**2, axis=0)))**2


def CepstrumOperator(target, a, k0, space=0):
    """Turns a white Gaussian random field into a smooth field on a LogRGSpace.

    Composed out of three operators:

        sym @ qht @ diag(sqrt_ceps),

    where sym is a :class:`SymmetrizingOperator`, qht is a :class:`QHTOperator`
    and ceps is the so-called cepstrum:

    .. math::
        \\mathrm{sqrt\\_ceps}(k) = \\frac{a}{1+(k/k0)^2}

    These operators are combined in this fashion in order to generate:

    - A field which is smooth, i.e. second derivatives are punished (note
      that the sqrt-cepstrum is essentially proportional to 1/k**2).

    - A field which is symmetric around the pixel in the middle of the space.
      This is result of the :class:`SymmetrizingOperator` and needed in order
      to decouple the degrees of freedom at the beginning and the end of the
      amplitude whenever :class:`CepstrumOperator` is used as in
      :class:`SLAmplitude`.

    The prior on the zero mode (or zero subspaces for more than one dimensions)
    is the integral of the prior over all other modes along the corresponding
    axis.

    Parameters
    ----------
    target : LogRGSpace
        Target domain of the operator, needs to be non-harmonic.
    a : float
        Cutoff of smoothness prior (positive only). Controls the
        regularization of the inverse laplace operator to be finite at zero.
        Larger values for the cutoff results in a weaker constraining prior.
    k0 : float, list of float
        Strength of smoothness prior in quefrency space (positive only) along
        each axis. If float then the strength is the same along each axis.
        Larger values result in a weaker constraining prior.
    """
    target = DomainTuple.make(target)
    space = infer_space(target, space)
    dim = len(target[space].shape)
    shape = tuple(s for i in range(len(target)) if i is not space for s in target[i].shape)
    a = _parameter_shaper(a, shape)
    k0 = _parameter_shaper(k0, (dim,)+shape)

    if target[space].harmonic:
        raise TypeError
    if np.any(a <= 0) or np.any(k0 <= 0):
        raise ValueError

    qht = QHTOperator(target, space)
    dom = qht.domain
    sym = SymmetrizingOperator(target, space)

    # Compute cepstrum field
    n = dom.axes[space][0]
    N = dom.axes[-1][-1]
    shape = dom.shape
    sl_extender = (slice(None),) + (None,)*n + (slice(None),)*dim + (None,)*(N-n-dim+1)
    q_array = dom[space].get_k_array()[sl_extender]
    a = a[(slice(None),)*n+(None,)*dim]
    k0 = k0[(slice(None),)*(1+n)+(None,)*dim]

    # Fill all non-zero modes
    no_zero_modes = (slice(None),)*(n) + (slice(1, None),)*dim
    ks = q_array[(slice(None),)+no_zero_modes]
    cepstrum_field = np.zeros(shape)
    cepstrum_field[no_zero_modes] = _ceps_kernel(ks, a, k0)

    # Fill zero-mode subspaces
    for i in range(dim):
        sl = (slice(None),)*(n+i) + (slice(1, None),)
        sl2 = (slice(None),)*(n+i) + (0,)
        cepstrum_field[sl2] = np.sum(cepstrum_field[sl], axis=n+i)
    cepstrum = Field.from_global_data(dom, cepstrum_field)

    return sym @ qht @ makeOp(cepstrum.sqrt())


def SLAmplitude(*, target, n_pix, a, k0, sm, sv, im, iv, keys=['tau', 'phi'],
                space=0):
    '''Operator for parametrizing smooth amplitudes (square roots of power
    spectra).

    The general guideline for setting up generative models in IFT is to
    transform the problem into the eigenbase of the prior and formulate the
    generative model in this base. This is done here for the case of an
    amplitude which is smooth and has a linear component (both on
    double-logarithmic scale).

    This function assembles an :class:`Operator` which maps two a-priori white
    Gaussian random fields to a smooth amplitude which is composed out of
    a linear and a smooth component.

    On double-logarithmic scale, i.e. both x and y-axis on logarithmic scale,
    the output of the generated operator is:

        AmplitudeOperator = 0.5*(smooth_component + linear_component)

    This is then exponentiated and exponentially binned (in this order).

    The prior on the linear component is parametrized by four real numbers,
    being expected value and prior variance on the slope and the y-intercept
    of the linear function.

    The prior on the smooth component is parametrized by two real numbers: the
    strength and the cutoff of the smoothness prior
    (see :class:`CepstrumOperator`).

    Parameters
    ----------
    n_pix : int
        Number of pixels of the space in which the .
    target : PowerSpace
        Target of the Operator.
    a : float
        Strength of smoothness prior (see :class:`CepstrumOperator`).
    k0 : float
        Cutoff of smothness prior in quefrency space (see
        :class:`CepstrumOperator`).
    sm : float
        Expected exponent of power law.
    sv : float
        Prior standard deviation of exponent of power law.
    im : float
        Expected y-intercept of power law. This is the value at t_0 of the
        LogRGSpace (see :class:`ExpTransform`).
    iv : float
        Prior standard deviation of y-intercept of power law.

    Returns
    -------
    Operator
        Operator which is defined on the space of white excitations fields and
        which returns on its target a power spectrum which consists out of a
        smooth and a linear part.
    '''
    return LinearSLAmplitude(target=target, n_pix=n_pix, a=a, k0=k0, sm=sm,
                             sv=sv, im=im, iv=iv, keys=keys, space=space).exp()


def LinearSLAmplitude(*, target, n_pix, a, k0, sm, sv, im, iv,
                      keys=['tau', 'phi'], space=0):
    '''LinearOperator for parametrizing smooth log-amplitudes (square roots of
    power spectra).

    Logarithm of SLAmplitude
    See documentation of SLAmplitude for more details
    '''
    target = DomainTuple.make(target)
    space = infer_space(target, space)
    if not (isinstance(n_pix, int) and isinstance(target[space], PowerSpace)):
        raise TypeError

    shape = tuple(s for i in range(len(target)) if i is not space for s in target[i].shape)
    sm, sv, im, iv = (_parameter_shaper(a, shape) for a in (sm, sv, im, iv))
    if np.any(sv <= 0) or np.any(iv <= 0):
        raise ValueError

    et = ExpTransform(target, n_pix, space)
    dom = et.domain

    # Smooth component
    dct = {'a': a, 'k0': k0, 'space': space}
    smooth = CepstrumOperator(dom, **dct).ducktape(keys[0])

    # Linear component
    sl = SlopeOperator(dom, space)
    mean = np.array([sm, im + sm*dom[space].t_0[0]])
    sig = np.array([sv, iv])
    axis = sl.domain.axes[space][0]
    mean = Field.from_global_data(sl.domain, np.moveaxis(mean, 0, axis))
    sig = Field.from_global_data(sl.domain, np.moveaxis(sig, 0, axis))
    linear = sl @ Adder(mean) @ makeOp(sig).ducktape(keys[1])

    # Combine linear and smooth component
    loglog_ampl = 0.5*(smooth + linear)

    # Go from loglog-space to linear-linear-space
    return et @ loglog_ampl
