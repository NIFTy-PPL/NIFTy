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

import unittest
from itertools import product
from test.common import expand

import nifty5 as ift
import numpy as np


class Model_Tests(unittest.TestCase):
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
        model = ift.ScalingOperator(6., var.target)
        ift.extra.check_value_gradient_consistency(model, var.val)

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
        model = ift.FieldAdapter(dom, "s1")*ift.FieldAdapter(dom, "s2")
        pos = ift.from_random("normal", dom)
        ift.extra.check_value_gradient_consistency(model, pos)
        model = ift.FieldAdapter(dom, "s1")+ift.FieldAdapter(dom, "s2")
        pos = ift.from_random("normal", dom)
        ift.extra.check_value_gradient_consistency(model, pos)
        model = ift.FieldAdapter(dom, "s1").scale(3.)
        pos = ift.from_random("normal", dom1)
        ift.extra.check_value_gradient_consistency(model, pos)
        model = ift.ScalingOperator(2.456, space)(
            ift.FieldAdapter(dom, "s1")*ift.FieldAdapter(dom, "s2"))
        pos = ift.from_random("normal", dom)
        ift.extra.check_value_gradient_consistency(model, pos)
        model = ift.positive_tanh(ift.ScalingOperator(2.456, space)(
            ift.FieldAdapter(dom, "s1")*ift.FieldAdapter(dom, "s2")))
        pos = ift.from_random("normal", dom)
        ift.extra.check_value_gradient_consistency(model, pos)
        if isinstance(space, ift.RGSpace):
            model = ift.FFTOperator(space)(
                ift.FieldAdapter(dom, "s1")*ift.FieldAdapter(dom, "s2"))
            pos = ift.from_random("normal", dom)
            ift.extra.check_value_gradient_consistency(model, pos)

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
    def testModelLibrary(self, space, Npixdof, ceps_a,
                         ceps_k, sm, sv, im, iv, seed):
        # tests amplitude model and coorelated field model
        np.random.seed(seed)
        model = ift.AmplitudeModel(space, Npixdof, ceps_a, ceps_k, sm,
                                   sv, im, iv)
        S = ift.ScalingOperator(1., model.domain)
        pos = S.draw_sample()
        ift.extra.check_value_gradient_consistency(model, pos)

        model2 = ift.CorrelatedField(space, model)
        S = ift.ScalingOperator(1., model2.domain)
        pos = S.draw_sample()
        ift.extra.check_value_gradient_consistency(model2, pos)

    @expand(product(
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]))
    def testPointModel(self, space, seed):
        S = ift.ScalingOperator(1., space)
        pos = S.draw_sample()
        alpha = 1.5
        q = 0.73
        model = ift.InverseGammaModel(space, alpha, q)
        # FIXME All those cdfs and ppfs are not very accurate
        ift.extra.check_value_gradient_consistency(model, pos, tol=1e-2)

#     @expand(product(
#         ['Variable', 'Constant'],
#         [ift.GLSpace(15),
#          ift.RGSpace(64, distances=.789),
#          ift.RGSpace([32, 32], distances=.789)],
#         [4, 78, 23]
#         ))
#     def testMultiModel(self, type, space, seed):
#         model = self.make_model(
#             type, space_key='s', space=space, seed=seed)['s']
#         mmodel = ift.MultiModel(model, 'g')
#         ift.extra.check_value_gradient_consistency(mmodel)
