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
    def make_model(self, type, **kwargs):
        if type == 'Constant':
            np.random.seed(kwargs['seed'])
            S = ift.ScalingOperator(1., kwargs['space'])
            s = S.draw_sample()
            return ift.Constant(
                ift.MultiField.from_dict({kwargs['space_key']: s}),
                ift.MultiField.from_dict({kwargs['space_key']: s}))
        elif type == 'Variable':
            np.random.seed(kwargs['seed'])
            S = ift.ScalingOperator(1., kwargs['space'])
            s = S.draw_sample()
            return ift.Variable(
                ift.MultiField.from_dict({kwargs['space_key']: s}))
        elif type == 'LinearModel':
            return ift.LinearModel(
                inp=kwargs['model'], lin_op=kwargs['lin_op'])
        else:
            raise ValueError('unknown type passed')

    def make_linear_operator(self, type, **kwargs):
        if type == 'ScalingOperator':
            lin_op = ift.ScalingOperator(1., kwargs['space'])
        else:
            raise ValueError('unknown type passed')
        return lin_op

    @expand(product(
        ['Variable', 'Constant'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testBasics(self, type1, space, seed):
        model1 = self.make_model(
            type1, space_key='s1', space=space, seed=seed)['s1']
        ift.extra.check_value_gradient_consistency(model1)

    @expand(product(
        ['Variable', 'Constant'],
        ['Variable'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testBinary(self, type1, type2, space, seed):
        model1 = self.make_model(
            type1, space_key='s1', space=space, seed=seed)['s1']
        model2 = self.make_model(
            type2, space_key='s2', space=space, seed=seed+1)['s2']
        ift.extra.check_value_gradient_consistency(model1*model2)
        ift.extra.check_value_gradient_consistency(model1+model2)
        ift.extra.check_value_gradient_consistency(model1*3.)

    @expand(product(
        ['Variable', 'Constant'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testLinModel(self, type1, space, seed):
        model1 = self.make_model(
            type1, space_key='s1', space=space, seed=seed)['s1']
        lin_op = self.make_linear_operator('ScalingOperator', space=space)
        model2 = self.make_model('LinearModel', model=model1, lin_op=lin_op)
        ift.extra.check_value_gradient_consistency(model1*model2)

    @expand(product(
        ['Variable', 'Constant'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testLocalModel(self, type, space, seed):
        model = self.make_model(
            type, space_key='s', space=space, seed=seed)['s']
        ift.extra.check_value_gradient_consistency(ift.PointwiseExponential(model))
        ift.extra.check_value_gradient_consistency(ift.PointwiseTanh(model))
        ift.extra.check_value_gradient_consistency(ift.PointwisePositiveTanh(model))


    @expand(product(
        ['Variable', 'Constant'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testMultiModel(self, type, space, seed):
        model = self.make_model(
            type, space_key='s', space=space, seed=seed)['s']
        mmodel = ift.MultiModel(model, 'g')
        ift.extra.check_value_gradient_consistency(mmodel)
