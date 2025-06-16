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
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..operators.diagonal_operator import DiagonalOperator
from ..operators.field_zero_padder import FieldZeroPadder
from ..operators.harmonic_operators import FFTOperator
from ..operators.scaling_operator import ScalingOperator
from ..operators.simple_linear_operators import FieldAdapter, Realizer
from ..sugar import makeOp
from .light_cone_operator import LightConeOperator, _field_from_function


def _float_or_listoffloat(inp):
    return [float(x) for x in inp] if isinstance(inp, list) else float(inp)


def _make_dynamic_operator(target, harmonic_padding, sm_s0, sm_x0, cone, keys, causal,
                           minimum_phase, sigc=None, quant=None, codomain=None):
    if not isinstance(target, RGSpace):
        raise TypeError("RGSpace required")
    if not target.harmonic:
        raise TypeError("Target space must be harmonic")
    if not (isinstance(harmonic_padding, int) or harmonic_padding is None
            or all(isinstance(ii, int) for ii in harmonic_padding)):
        raise TypeError
    sm_s0 = float(sm_s0)
    sm_x0 = _float_or_listoffloat(sm_x0)
    cone = bool(cone)
    if not all(isinstance(ss, str) for ss in keys):
        raise TypeError
    causal, minimum_phase = bool(causal), bool(minimum_phase)
    if sigc is not None:
        sigc = _float_or_listoffloat(sigc)
    if quant is not None:
        quant = float(quant)
    if cone and (sigc is None or quant is None):
        raise RuntimeError

    if codomain is None:
        codomain = target.get_default_codomain()
    dom = DomainTuple.make(codomain)
    ops = {}
    FFT = FFTOperator(dom)
    Real = Realizer(dom)
    ops['FFT'] = FFT
    ops['Real'] = Real
    if harmonic_padding is None:
        CentralPadd = ScalingOperator(FFT.target, 1.)
    else:
        if isinstance(harmonic_padding, int):
            harmonic_padding = list((harmonic_padding,)*len(FFT.target.shape))
        elif len(harmonic_padding) != len(FFT.target.shape):
            raise (ValueError, "Shape mismatch!")
        shp = [
            sh + harmonic_padding[ii] for ii, sh in enumerate(FFT.target.shape)
        ]
        CentralPadd = FieldZeroPadder(FFT.target, shp, central=True)
    ops['central_padding'] = CentralPadd
    sdom = CentralPadd.target[0].get_default_codomain()
    FFTB = FFTOperator(sdom)(Realizer(sdom))

    m = FieldAdapter(sdom, keys[0])

    dists = m.target[0].distances
    if isinstance(sm_x0, float):
        sm_x0 = list((sm_x0,)*len(dists))
    elif len(sm_x0) != len(dists):
        raise (ValueError, "Shape mismatch!")

    def smoothness_prior_func(x):
        res = 1.
        for i in range(len(dists)):
            res = res + (x[i]/sm_x0[i]/dists[i])**2
        return sm_s0/res

    Sm = makeOp(_field_from_function(m.target, smoothness_prior_func))
    m = CentralPadd.adjoint(FFTB(Sm(m)))
    ops['smoothed_dynamics'] = m

    m = -m.ptw("log")
    if not minimum_phase:
        m = m.ptw("exp")
    if causal or minimum_phase:
        m = Real.adjoint(FFT.inverse(Realizer(FFT.target).adjoint(m)))
        kernel = makeOp(
            _field_from_function(FFT.domain, (lambda x: 1. + np.sign(x[0]))))
        m = kernel(m)

    if cone and len(m.target.shape) > 1:
        if isinstance(sigc, float):
            sigc = list((sigc,)*(len(m.target.shape) - 1))
        elif len(sigc) != len(m.target.shape) - 1:
            raise (ValueError, "Shape mismatch!")
        c = FieldAdapter(UnstructuredDomain(len(sigc)), keys[1])
        c = makeOp(Field(c.target, np.array(sigc)))(c)

        lightspeed = ScalingOperator(c.target, -0.5)(c).ptw("exp")
        scaling = np.array(m.target[0].distances[1:])/m.target[0].distances[0]
        scaling = DiagonalOperator(Field(c.target, scaling))
        ops['lightspeed'] = scaling(lightspeed)

        c = LightConeOperator(c.target, m.target, quant) @ c.ptw("exp")
        ops['light_cone'] = c
        m = c*m

    if causal or minimum_phase:
        m = FFT(Real(m))
    if minimum_phase:
        m = m.ptw("exp")
    return m, ops


def dynamic_operator(*, target, harmonic_padding, sm_s0, sm_x0, key, causal=True,
                     minimum_phase=False):
    """Constructs an operator encoding the Green's function of a linear
    homogeneous dynamic system.

    When evaluated, this operator returns the Green's function representation
    in harmonic space. This result can be used as a convolution kernel to
    construct solutions of the homogeneous stochastic differential equation
    encoded in this operator. Note that if causal is True, the Green's function
    is convolved with a step function in time, where the temporal axis is the
    first axis of the space. In this case the resulting function only extends
    up to half the length of the first axis of the space to avoid boundary
    effects during convolution. If minimum_phase is true then the spectrum of
    the Green's function is used to construct a corresponding minimum phase
    filter.

    Parameters
    ----------
    target : RGSpace
        The harmonic space in which the Green's function shall be constructed.
    harmonic_padding : None, int, list of int
        Amount of central padding in harmonic space in pixels. If None the
        field is not padded at all.
    sm_s0 : float
        Cutoff for dynamic smoothness prior.
    sm_x0 : float, list of float
        Scaling of dynamic smoothness along each axis.
    key : String
        key for dynamics encoding parameter.
    causal : boolean
        Whether or not the Green's function shall be causal in time.
        Default is True.
    minimum_phase: boolean
        Whether or not the Green's function shall be a minimum phase filter.
        Default is False.

    Returns
    -------
    Operator
        The Operator encoding the dynamic Green's function in target space.
    Dictionary of Operator
        A collection of sub-chains of Operator which can be used for plotting
        and evaluation.

    Notes
    -----
    The first axis of the domain is interpreted the time axis.
    """
    dct = {
        'target': target,
        'harmonic_padding': harmonic_padding,
        'sm_s0': sm_s0,
        'sm_x0': sm_x0,
        'keys': [key],
        'causal': causal,
        'cone': False,
        'minimum_phase': minimum_phase,
    }
    return _make_dynamic_operator(**dct)


def dynamic_lightcone_operator(*, target, harmonic_padding, sm_s0, sm_x0, key, lightcone_key,
                               sigc, quant, causal=True, minimum_phase=False):
    '''Extends the functionality of :func:`dynamic_operator` to a Green's
    function which is constrained to be within a light cone.

    The resulting Green's function is constrained to be within a light cone.
    This is achieved via convolution of the function with a light cone in
    space-time. Thereby the first axis of the space is set to be the teporal
    axis.

    Parameters
    ----------
    target : RGSpace
        The harmonic space in which the Green's function shall be constructed.
        It needs to have at least two dimensions.
    harmonic_padding : None, int, list of int
        Amount of central padding in harmonic space in pixels. If None the
        field is not padded at all.
    sm_s0 : float
        Cutoff for dynamic smoothness prior.
    sm_x0 : float, list of float
        Scaling of dynamic smoothness along each axis.
    key : String
        Key for dynamics encoding parameter.
    lightcone_key: String
        Key for lightspeed paramteter.
    sigc : float, list of float
        Variance of lightspeed parameter.
    quant : float
        Quantization of the light cone in pixels.
    causal : boolean
        Whether or not the Green's function shall be causal in time.
        Default is True.
    minimum_phase: boolean
        Whether or not the Green's function shall be a minimum phase filter.
        Default is False.

    Returns
    -------
    Operator
        The Operator encoding the dynamic Green's function in harmonic space.
    Dictionary of Operator
        A collection of sub-chains of Operator which can be used for plotting
        and evaluation.

    Notes
    -----
    The first axis of the domain is interpreted the time axis.
    '''

    if len(target.shape) < 2:
        raise ValueError("Space must be at least 2 dimensional!")
    dct = {
        'target': target,
        'harmonic_padding': harmonic_padding,
        'sm_s0': sm_s0,
        'sm_x0': sm_x0,
        'keys': [key, lightcone_key],
        'causal': causal,
        'cone': True,
        'minimum_phase': minimum_phase,
        'sigc': sigc,
        'quant': quant
    }
    return _make_dynamic_operator(**dct)
