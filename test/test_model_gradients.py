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
import pytest
from numpy.testing import assert_

import nifty5 as ift

from .common import list2fixture

pmp = pytest.mark.parametrize
space = list2fixture([
    ift.GLSpace(15),
    ift.RGSpace(64, distances=.789),
    ift.RGSpace([32, 32], distances=.789)
])
space1 = space
seed = list2fixture([4, 78, 23])


def _make_linearization(type, space, seed):
    np.random.seed(seed)
    S = ift.ScalingOperator(1., space)
    s = S.draw_sample()
    if type == "Constant":
        return ift.Linearization.make_const(s)
    elif type == "Variable":
        return ift.Linearization.make_var(s)
    raise ValueError('unknown type passed')


def testBasics(space, seed):
    var = _make_linearization("Variable", space, seed)
    model = ift.ScalingOperator(6., var.target)
    ift.extra.check_value_gradient_consistency(model, var.val)


@pmp('type1', ['Variable', 'Constant'])
@pmp('type2', ['Variable'])
def testBinary(type1, type2, space, seed):
    dom1 = ift.MultiDomain.make({'s1': space})
    dom2 = ift.MultiDomain.make({'s2': space})

    # FIXME Remove this?
    _make_linearization(type1, dom1, seed)
    _make_linearization(type2, dom2, seed)

    dom = ift.MultiDomain.union((dom1, dom2))
    select_s1 = ift.ducktape(None, dom, "s1")
    select_s2 = ift.ducktape(None, dom, "s2")
    model = select_s1*select_s2
    pos = ift.from_random("normal", dom)
    ift.extra.check_value_gradient_consistency(model, pos, ntries=20)
    model = select_s1 + select_s2
    pos = ift.from_random("normal", dom)
    ift.extra.check_value_gradient_consistency(model, pos, ntries=20)
    model = select_s1.scale(3.)
    pos = ift.from_random("normal", dom1)
    ift.extra.check_value_gradient_consistency(model, pos, ntries=20)
    model = ift.ScalingOperator(2.456, space)(select_s1*select_s2)
    pos = ift.from_random("normal", dom)
    ift.extra.check_value_gradient_consistency(model, pos, ntries=20)
    model = ift.sigmoid(2.456*(select_s1*select_s2))
    pos = ift.from_random("normal", dom)
    ift.extra.check_value_gradient_consistency(model, pos, ntries=20)
    pos = ift.from_random("normal", dom)
    model = ift.OuterProduct(pos['s1'], ift.makeDomain(space))
    ift.extra.check_value_gradient_consistency(model, pos['s2'], ntries=20)
    if isinstance(space, ift.RGSpace):
        model = ift.FFTOperator(space)(select_s1*select_s2)
        pos = ift.from_random("normal", dom)
        ift.extra.check_value_gradient_consistency(model, pos, ntries=20)


def testModelLibrary(space, seed):
    # Tests amplitude model and coorelated field model
    Npixdof, ceps_a, ceps_k, sm, sv, im, iv = 4, 0.5, 2., 3., 1.5, 1.75, 1.3
    np.random.seed(seed)
    domain = ift.PowerSpace(space.get_default_codomain())
    model = ift.SLAmplitude(domain, Npixdof, ceps_a, ceps_k, sm, sv, im, iv)
    assert_(isinstance(model, ift.Operator))
    S = ift.ScalingOperator(1., model.domain)
    pos = S.draw_sample()
    ift.extra.check_value_gradient_consistency(model, pos, ntries=20)

    model2 = ift.CorrelatedField(space, model)
    S = ift.ScalingOperator(1., model2.domain)
    pos = S.draw_sample()
    ift.extra.check_value_gradient_consistency(model2, pos, ntries=20)

    domtup = ift.DomainTuple.make((space, space))
    model3 = ift.MfCorrelatedField(domtup, [model, model])
    S = ift.ScalingOperator(1., model3.domain)
    pos = S.draw_sample()
    ift.extra.check_value_gradient_consistency(model3, pos, ntries=20)


def testPointModel(space, seed):
    S = ift.ScalingOperator(1., space)
    pos = S.draw_sample()
    alpha = 1.5
    q = 0.73
    model = ift.InverseGammaOperator(space, alpha, q)
    # FIXME All those cdfs and ppfs are not very accurate
    ift.extra.check_value_gradient_consistency(model, pos, tol=1e-2, ntries=20)


@pmp('target', [
    ift.RGSpace(64, distances=.789,harmonic=True),
    ift.RGSpace([32, 32], distances=.789,harmonic=True),
    ift.RGSpace([32, 32, 8], distances=.789,harmonic=True)
])
@pmp('causal', [True, False])
@pmp('minimum_phase', [True, False])
@pmp('seed', [4, 78, 23])
def testDynamicModel(target, causal, minimum_phase, seed):
    dct = {
            'target': target,
            'harmonic_padding': None,
            'sm_s0': 3.,
            'sm_x0': 1.,
            'key': 'f',
            'causal': causal,
            'minimum_phase': minimum_phase
            }
    model, _ = ift.dynamic_operator(**dct)
    S = ift.ScalingOperator(1., model.domain)
    pos = S.draw_sample()
    # FIXME I dont know why smaller tol fails for 3D example
    ift.extra.check_value_gradient_consistency(model, pos, tol=1e-5, ntries=20)
    if len(target.shape) > 1:
        dct = {
            'target': target,
            'harmonic_padding': None,
            'sm_s0': 3.,
            'sm_x0': 1.,
            'key': 'f',
            'lightcone_key': 'c',
            'sigc': 1.,
            'quant': 5,
            'causal': causal,
            'minimum_phase': minimum_phase
        }
        dct['lightcone_key'] = 'c'
        dct['sigc'] = 1.
        dct['quant'] = 5
        model, _ = ift.dynamic_lightcone_operator(**dct)
        S = ift.ScalingOperator(1., model.domain)
        pos = S.draw_sample()
        # FIXME I dont know why smaller tol fails for 3D example
        ift.extra.check_value_gradient_consistency(
            model, pos, tol=1e-5, ntries=20)
