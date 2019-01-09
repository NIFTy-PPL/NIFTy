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

import unittest
from itertools import product
from test.common import expand

import nifty5 as ift
import numpy as np


class OperatorTests(unittest.TestCase):
    @staticmethod
    def make_linearization(type, space, seed):
        np.random.seed(seed)
        S = ift.ScalingOperator(1., space)
        s = S.draw_sample()
        if type == "Constant":
            return ift.Linearization.make_const(s)
        elif type == "Variable":
            return ift.Linearization.make_var(s)
        raise ValueError('unknown type passed')

    @expand(product(
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testBasics(self, space, seed):
        var = self.make_linearization("Variable", space, seed)
        op = ift.ScalingOperator(6., var.target)
        ift.extra.check_value_gradient_consistency(op, var.val)

    @expand(product(
        ['Variable', 'Constant'],
        ['Variable'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testBinary(self, type1, type2, space, seed):
        dom1 = ift.MultiDomain.make({'s1': space})
        lin1 = self.make_linearization(type1, dom1, seed)
        dom2 = ift.MultiDomain.make({'s2': space})
        lin2 = self.make_linearization(type2, dom2, seed)

        dom = ift.MultiDomain.union((dom1, dom2))
        select_s1 = ift.ducktape(None, dom, "s1")
        select_s2 = ift.ducktape(None, dom, "s2")
        op = select_s1*select_s2
        pos = ift.from_random("normal", dom)
        ift.extra.check_value_gradient_consistency(op, pos, ntries=20)
        op = select_s1+select_s2
        pos = ift.from_random("normal", dom)
        ift.extra.check_value_gradient_consistency(op, pos, ntries=20)
        op = select_s1.scale(3.)
        pos = ift.from_random("normal", dom1)
        ift.extra.check_value_gradient_consistency(op, pos, ntries=20)
        op = ift.ScalingOperator(2.456, space)(select_s1*select_s2)
        pos = ift.from_random("normal", dom)
        ift.extra.check_value_gradient_consistency(op, pos, ntries=20)
        op = ift.sigmoid(ift.ScalingOperator(2.456, space)(
            select_s1*select_s2))
        pos = ift.from_random("normal", dom)
        ift.extra.check_value_gradient_consistency(op, pos, ntries=20)
        pos = ift.from_random("normal", dom)
        op = ift.OuterProduct(pos['s1'], ift.makeDomain(space))
        ift.extra.check_value_gradient_consistency(op, pos['s2'], ntries=20)
        if isinstance(space, ift.RGSpace):
            op = ift.FFTOperator(space)(select_s1*select_s2)
            pos = ift.from_random("normal", dom)
            ift.extra.check_value_gradient_consistency(op, pos, ntries=20)

    @expand(product(
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4],
        [0.5],
        [2.],
        [3.],
        [1.5],
        [1.75],
        [1.3],
        [4, 78, 23],
        ))
    def testOperatorLibrary(self, space, Npixdof, ceps_a,
                            ceps_k, sm, sv, im, iv, seed):
        # tests amplitude operator and coorelated field operator
        np.random.seed(seed)
        op = ift.AmplitudeOperator(space, Npixdof, ceps_a, ceps_k, sm,
                                   sv, im, iv)
        S = ift.ScalingOperator(1., op.domain)
        pos = S.draw_sample()
        ift.extra.check_value_gradient_consistency(op, pos, ntries=20)

        op2 = ift.CorrelatedField(space, op)
        S = ift.ScalingOperator(1., op2.domain)
        pos = S.draw_sample()
        ift.extra.check_value_gradient_consistency(op2, pos, ntries=20)

    @expand(product(
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]))
    def testInvGammaOperator(self, space, seed):
        S = ift.ScalingOperator(1., space)
        pos = S.draw_sample()
        alpha = 1.5
        q = 0.73
        op = ift.InverseGammaOperator(space, alpha, q)
        # FIXME All those cdfs and ppfs are not very accurate
        ift.extra.check_value_gradient_consistency(op, pos, tol=1e-2,
                                                   ntries=20)

    @expand(product(
        [ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789),
         ift.RGSpace([32, 32, 8], distances=.789)],
        [True, False],
        [True, False],
        [4, 78, 23]))
    def testDynamicModel(self, domain, causal, minimum_phase, seed):
        model, _ = ift.dynamic_operator(domain,None,1.,1.,'f',
                                        causal = causal,
                                        minimum_phase = minimum_phase)
        S = ift.ScalingOperator(1., model.domain)
        pos = S.draw_sample()
        # FIXME I dont know why smaller tol fails for 3D example
        ift.extra.check_value_gradient_consistency(model, pos, tol=1e-5,
                                                   ntries=20)
        if len(domain.shape) > 1:
            model, _ = ift.dynamic_lightcone_operator(domain,None,3.,1.,
                                                      'f','c',1.,5,
                                                      causal = causal,
                                                      minimum_phase = minimum_phase)
            S = ift.ScalingOperator(1., model.domain)
            pos = S.draw_sample()
            # FIXME I dont know why smaller tol fails for 3D example
            ift.extra.check_value_gradient_consistency(model, pos, tol=1e-5,
                                                       ntries=20)
