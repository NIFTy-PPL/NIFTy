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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import pytest

import nifty6 as ift

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize
space = list2fixture([
    ift.GLSpace(5),
    ift.RGSpace(3, distances=.789),
    ift.RGSpace([4, 4], distances=.789)
])
_h_RG_spaces = [
    ift.RGSpace(7, distances=0.2, harmonic=True),
    ift.RGSpace((3, 4), distances=(.2, .3), harmonic=True)
]
_h_spaces = _h_RG_spaces + [ift.LMSpace(17)]
space1 = space
seed = list2fixture([4, 78])
ntries = 10


def testBasics(space, seed):
    with ift.random.Context(seed):
        s = ift.from_random('normal', space)
        var = ift.Linearization.make_var(s)
        model = ift.ScalingOperator(var.target, 6.)
        ift.extra.check_jacobian_consistency(model, var.val, ntries=ntries)


@pmp('type1', ['Variable', 'Constant'])
@pmp('type2', ['Variable'])
def testBinary(type1, type2, space, seed):
    with ift.random.Context(seed):
        dom1 = ift.MultiDomain.make({'s1': space})
        dom2 = ift.MultiDomain.make({'s2': space})
        dom = ift.MultiDomain.union((dom1, dom2))
        select_s1 = ift.ducktape(None, dom1, "s1")
        select_s2 = ift.ducktape(None, dom2, "s2")
        model = select_s1*select_s2
        pos = ift.from_random("normal", dom)
        ift.extra.check_jacobian_consistency(model, pos, ntries=ntries)
        model = select_s1 + select_s2
        pos = ift.from_random("normal", dom)
        ift.extra.check_jacobian_consistency(model, pos, ntries=ntries)
        model = select_s1.scale(3.)
        pos = ift.from_random("normal", dom1)
        ift.extra.check_jacobian_consistency(model, pos, ntries=ntries)
        model = ift.ScalingOperator(space, 2.456)(select_s1*select_s2)
        pos = ift.from_random("normal", dom)
        ift.extra.check_jacobian_consistency(model, pos, ntries=ntries)
        model = (2.456*(select_s1*select_s2)).ptw("sigmoid")
        pos = ift.from_random("normal", dom)
        ift.extra.check_jacobian_consistency(model, pos, ntries=ntries)
        pos = ift.from_random("normal", dom)
        model = ift.OuterProduct(pos['s1'], ift.makeDomain(space))
        ift.extra.check_jacobian_consistency(model, pos['s2'], ntries=ntries)
        model = select_s1**2
        pos = ift.from_random("normal", dom1)
        ift.extra.check_jacobian_consistency(model, pos, ntries=ntries)
        model = select_s1.clip(-1, 1)
        pos = ift.from_random("normal", dom1)
        ift.extra.check_jacobian_consistency(model, pos, ntries=ntries)
        f = ift.from_random("normal", space)
        model = select_s1.clip(f-0.1, f+1.)
        pos = ift.from_random("normal", dom1)
        ift.extra.check_jacobian_consistency(model, pos, ntries=ntries)
        if isinstance(space, ift.RGSpace):
            model = ift.FFTOperator(space)(select_s1*select_s2)
            pos = ift.from_random("normal", dom)
            ift.extra.check_jacobian_consistency(model, pos, ntries=ntries)


def testSpecialDistributionOps(space, seed):
    with ift.random.Context(seed):
        pos = ift.from_random('normal', space)
        alpha = 1.5
        q = 0.73
        model = ift.InverseGammaOperator(space, alpha, q)
        ift.extra.check_jacobian_consistency(model, pos, ntries=20)
        model = ift.UniformOperator(space, alpha, q)
        ift.extra.check_jacobian_consistency(model, pos, ntries=20)


@pmp('neg', [True, False])
def testAdder(space, seed, neg):
    with ift.random.Context(seed):
        f = ift.from_random('normal', space)
        f1 = ift.from_random('normal', space)
        op = ift.Adder(f1, neg)
        ift.extra.check_jacobian_consistency(op, f, ntries=ntries)
        op = ift.Adder(f1.val.ravel()[0], neg=neg, domain=space)
        ift.extra.check_jacobian_consistency(op, f, ntries=ntries)


@pmp('target', [ift.RGSpace(64, distances=.789, harmonic=True),
                ift.RGSpace([10, 10], distances=.789, harmonic=True)])
@pmp('causal', [True, False])
@pmp('minimum_phase', [True, False])
def testDynamicModel(target, causal, minimum_phase, seed):
    with ift.random.Context(seed):
        dct = {'target': target,
               'harmonic_padding': None,
               'sm_s0': 3.,
               'sm_x0': 1.,
               'key': 'f',
               'causal': causal,
               'minimum_phase': minimum_phase}
        model, _ = ift.dynamic_operator(**dct)
        pos = ift.from_random('normal', model.domain)
        ift.extra.check_jacobian_consistency(model, pos, tol=1e-7, ntries=ntries)
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
            pos = ift.from_random('normal', model.domain)
            ift.extra.check_jacobian_consistency(model, pos, tol=1e-7, ntries=ntries)


@pmp('h_space', _h_spaces)
@pmp('specialbinbounds', [True, False])
@pmp('logarithmic', [True, False])
@pmp('nbin', [3, None])
def testNormalization(h_space, specialbinbounds, logarithmic, nbin):
    if not specialbinbounds and (not logarithmic or nbin is not None):
        return
    binbounds = None
    if specialbinbounds:
        binbounds = ift.PowerSpace.useful_binbounds(h_space, logarithmic, nbin)
    dom = ift.PowerSpace(h_space, binbounds)
    op = ift.library.correlated_fields._Normalization(dom)
    pos = 0.1*ift.from_random('normal', op.domain)
    ift.extra.check_jacobian_consistency(op, pos, ntries=10)
