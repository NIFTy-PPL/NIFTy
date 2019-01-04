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

from ..operators.scaling_operator import ScalingOperator
from ..operators.harmonic_operators import FFTOperator
from ..operators.field_zero_padder import FieldZeroPadder
from ..operators.simple_linear_operators import Realizer,FieldAdapter
from ..operators.diagonal_operator import DiagonalOperator
from ..sugar import makeOp
from ..field import Field
from ..domains.unstructured_domain import UnstructuredDomain

from .light_cone_operator import LightConeOperator,field_from_function

def make_dynamic_operator(FFT,harmonic_padding,sm_s0,sm_x0,
                       keys=['f', 'c'],
                       causal=True,
                       cone=True,
                       minimum_phase=False,
                       sigc=3.,
                       quant=5.):
    '''
    Constructs an operator for a dynamic field prior.

    Parameters
    ----------
    FFT : FFTOperator
        A Fourier Transformation Operator of the space under consideration
    harmonic_padding : None, list of float
        Amount of central padding in harmonic space in pixels. If None the field is not padded at all.
    sm_s0 : float
        Cutoff for dynamic smoothness prior
    sm_x0 : float, List of float
        Scaling of dynamic smoothness along each axis
    keys : List of String
        keys of input fields of operator.
    causal : boolean
        Whether or not the reconstructed dynamics should be causal in time
    cone : boolean
        Whether or not the reconstructed dynamics should be within a light cone
    minimum_phase: boolean
        Whether or not the reconstructed dynamics should be minimum phase
    sigc : float, List of float
        variance of light cone parameters.
        If cone is False this is ignored
    quant : float
        Quantization of the light cone in pixels.
        If cone is False this is ignored

    Returns
    -------
    Operator
        The Operator encoding the dynamic Greens function in harmonic space when evaluated.
    Dictionary of Operator
        A collection of sub-chains of Operators which can be used for plotting and evaluation.

    Notes
    -----
    Currently only supports RGSpaces.
    Note that the first axis of the space is interpreted as the time axis.
    '''
    ops = {}
    if harmonic_padding is None:
        CentralPadd = ScalingOperator(1.,FFT.target)
    else:
        shp = ()
        for i in range(len(FFT.target.shape)):
            shp += (FFT.target.shape[i] + harmonic_padding[i],)
        CentralPadd = FieldZeroPadder(FFT.target,shp,central=True)
    ops['CentralPadd'] = CentralPadd
    sdom = CentralPadd.target[0].get_default_codomain()
    FFTB = FFTOperator(sdom)(Realizer(sdom))

    m = FieldAdapter(sdom, keys[0])

    dists = m.target[0].distances
    if isinstance(sm_x0,float):
        sm_x0 = list((sm_x0,)*len(dists))
    elif len(sm_x0) != len(dists):
        raise(ValueError,"Shape mismatch!")
    def smoothness_prior_func(x):
        res = 1.
        for i in range(len(dists)):
            res = res + (x[i]/sm_x0[i]/dists[i])**2
        return sm_s0/res

    Sm = makeOp(field_from_function(m.target, smoothness_prior_func))
    m = CentralPadd.adjoint(FFTB(Sm(m)))
    ops[keys[0]+'_k'] = m

    m = -m.log()
    if not minimum_phase:
        m = m.exp()
    ops['Gncc'] = m
    if causal:
        m = FFT.inverse(Realizer(FFT.target).adjoint(m))
        kernel = makeOp(field_from_function(FFT.domain, (lambda x: 1.+np.sign(x[0]))))
        m = kernel(m)
    elif minimum_phase:
        raise(ValueError,"minimum phase and not causal not possible!")

    if cone and len(m.target.shape) > 1:
        if isinstance(sigc,float):
            sigc = list((sigc,)*(len(m.target.shape)-1))
        elif len(sigc) != len(m.target.shape)-1:
            raise(ValueError,"Shape mismatch!")
        c = FieldAdapter(UnstructuredDomain(len(sigc)), keys[1])
        c = makeOp(Field.from_global_data(c.target, np.array(sigc)))(c)

        lightspeed = ScalingOperator(-0.5,c.target)(c).exp()
        scaling = m.target[0].distances[1:]/m.target[0].distances[0]
        scaling = DiagonalOperator(Field.from_global_data(c.target,scaling))
        ops['lightspeed'] = scaling(lightspeed)

        c = LightConeOperator(c.target, m.target, quant)(c.exp())
        ops['a'] = c
        m = c*m

    ops['Gx'] = m
    if causal:
        m = FFT(m)
    if minimum_phase:
        m = m.exp()
    return m, ops
