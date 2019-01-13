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


def _make_dynamic_operator(domain, harmonic_padding, sm_s0, sm_x0, keys=['f', 'c'],
                          causal=True, cone=True, minimum_phase=False, sigc=3.,
                          quant=5.):
    dom = DomainTuple.make(domain)
    if not isinstance(dom[0], RGSpace):
        raise TypeError("RGSpace required")
    ops = {}
    FFT = FFTOperator(dom)
    Real = Realizer(dom)
    ops['FFT'] = FFT
    ops['Real'] = Real
    if harmonic_padding is None:
        CentralPadd = ScalingOperator(1., FFT.target)
    else:
        if isinstance(harmonic_padding, int):
            harmonic_padding = list((harmonic_padding,)*len(FFT.target.shape))
        elif len(harmonic_padding) != len(FFT.target.shape):
            raise (ValueError, "Shape mismatch!")
        shp = ()
        for i in range(len(FFT.target.shape)):
            shp += (FFT.target.shape[i] + harmonic_padding[i],)
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

    m = -m.log()
    if not minimum_phase:
        m = m.exp()
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
        c = makeOp(Field.from_global_data(c.target, np.array(sigc)))(c)

        lightspeed = ScalingOperator(-0.5, c.target)(c).exp()
        scaling = np.array(m.target[0].distances[1:])/m.target[0].distances[0]
        scaling = DiagonalOperator(Field.from_global_data(c.target, scaling))
        ops['lightspeed'] = scaling(lightspeed)

        c = LightConeOperator(c.target, m.target, quant)(c.exp())
        ops['light_cone'] = c
        m = c*m

    if causal or minimum_phase:
        m = FFT(Real(m))
    if minimum_phase:
        m = m.exp()
    return m, ops

def dynamic_operator(domain, harmonic_padding, sm_s0, sm_x0, key,
                     causal=True, minimum_phase=False):
    '''
    Constructs an operator encoding the Greens function of a linear homogeneous dynamic system.

    Parameters
    ----------
    domain : RGSpace
        The space under consideration
    harmonic_padding : None, int, list of int
        Amount of central padding in harmonic space in pixels. If None the field is not padded at all.
    sm_s0 : float
        Cutoff for dynamic smoothness prior
    sm_x0 : float, List of float
        Scaling of dynamic smoothness along each axis
    key : String
        key for dynamics encoding parameter.
    causal : boolean
        Whether or not the reconstructed dynamics should be causal in time
    minimum_phase: boolean
        Whether or not the reconstructed dynamics should be minimum phase

    Returns
    -------
    Operator
        The Operator encoding the dynamic Greens function in harmonic space.
    Dictionary of Operator
        A collection of sub-chains of Operators which can be used for plotting and evaluation.

    Notes
    -----
    Currently only supports RGSpaces.
    Note that the first axis of the space is interpreted as the time axis.
    '''
    return _make_dynamic_operator(domain, harmonic_padding, sm_s0, sm_x0,
                                  keys=[key],
                                  causal=causal, cone=False,
                                  minimum_phase=minimum_phase)

def dynamic_lightcone_operator(domain, harmonic_padding, sm_s0, sm_x0, key,
                               lightcone_key, sigc, quant,
                               causal=True, minimum_phase=False):
    '''
    Constructs an operator encoding the Greens function of a linear
    homogeneous dynamic system. The Greens function is constrained
    to be within a light cone.

    Parameters
    ----------
    domain : RGSpace
        The space under consideration. Must have dim > 1.
    harmonic_padding : None, int, list of int
        Amount of central padding in harmonic space in pixels. If None the
        field is not padded at all.
    sm_s0 : float
        Cutoff for dynamic smoothness prior.
    sm_x0 : float, List of float
        Scaling of dynamic smoothness along each axis.
    key : String
        key for dynamics encoding parameter.
    lightcone_key: String
        key for lightspeed paramteter.
    sigc : float, List of float
        variance of lightspeed parameter.
    quant : float
        Quantization of the light cone in pixels.
    causal : boolean
        Whether or not the reconstructed dynamics should be causal in time.
    minimum_phase: boolean
        Whether or not the reconstructed dynamics should be minimum phase.

    Returns
    -------
    Operator
        The Operator encoding the dynamic Greens function in harmonic space
        when evaluated.
    dict
        A collection of sub-chains of :class:`Operator` s which can be used
        for plotting and evaluation.

    Notes
    -----
    The first axis of the space is interpreted as the time axis.
    Supports only RGSpaces currently.
    '''
    if len(domain.shape) < 2:
        raise ValueError("Space must be at least 2 dimensional!")
    return _make_dynamic_operator(domain,harmonic_padding,sm_s0,sm_x0,
                                  keys=[key,lightcone_key],
                                  causal=causal, cone=True,
                                  minimum_phase = minimum_phase,
                                  sigc = sigc, quant = quant)
