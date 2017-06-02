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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division

import unittest
import numpy as np

from numpy.testing import assert_, assert_equal, assert_almost_equal
from nifty import RGSpace
from test.common import expand

# [shape, zerocenter, distances, harmonic, expected]
CONSTRUCTOR_CONFIGS = [
        [(8,), False, None, False,
            {
                'shape': (8,),
                'zerocenter': (False,),
                'distances': (0.125,),
                'harmonic': False,
                'dim': 8,
                'total_volume': 1.0
            }],
        [(8,), True, None, False,
            {
                'shape': (8,),
                'zerocenter': (True,),
                'distances': (0.125,),
                'harmonic': False,
                'dim': 8,
                'total_volume': 1.0
            }],
        [(8,), False, None, True,
            {
                'shape': (8,),
                'zerocenter': (False,),
                'distances': (1.0,),
                'harmonic': True,
                'dim': 8,
                'total_volume': 8.0
            }],
        [(8,), False, (12,), True,
            {
                'shape': (8,),
                'zerocenter': (False,),
                'distances': (12.0,),
                'harmonic': True,
                'dim': 8,
                'total_volume': 96.0
            }],
        [(11, 11), (False, True), None, False,
            {
                'shape': (11, 11),
                'zerocenter': (False, True),
                'distances': (1/11, 1/11),
                'harmonic': False,
                'dim': 121,
                'total_volume': 1.0
            }],
        [(11, 11), True, (1.3, 1.3), True,
            {
                'shape': (11, 11),
                'zerocenter': (True, True),
                'distances': (1.3, 1.3),
                'harmonic': True,
                'dim': 121,
                'total_volume': 204.49
            }]

    ]


def get_distance_array_configs():
    # for RGSpace(shape=(4, 4), distances=None, zerocenter=[False, False])
    cords_0 = np.ogrid[0:4, 0:4]
    da_0 = ((cords_0[0] - 4 // 2) * 0.25)**2
    da_0 = np.fft.ifftshift(da_0)
    temp = ((cords_0[1] - 4 // 2) * 0.25)**2
    temp = np.fft.ifftshift(temp)
    da_0 = da_0 + temp
    da_0 = np.sqrt(da_0)
    # for RGSpace(shape=(4, 4), distances=None, zerocenter=[True, True])
    da_1 = ((cords_0[0] - 4 // 2) * 0.25)**2
    temp = ((cords_0[1] - 4 // 2) * 0.25)**2
    da_1 = da_1 + temp
    da_1 = np.sqrt(da_1)
    # for RGSpace(shape=(4, 4), distances=(12, 12), zerocenter=[True, True])
    da_2 = ((cords_0[0] - 4 // 2) * 12)**2
    temp = ((cords_0[1] - 4 // 2) * 12)**2
    da_2 = da_2 + temp
    da_2 = np.sqrt(da_2)
    return [
        [(4, 4),  None, [False, False], da_0],
        [(4, 4),  None, [True, True], da_1],
        [(4, 4),  (12, 12), [True, True], da_2]
        ]


def get_weight_configs():
    np.random.seed(42)
    # power 1
    w_0_x = np.random.rand(32, 12, 6)
    # for RGSpace(shape=(11,11), distances=None, harmonic=False)
    w_0_res = w_0_x * (1/11 * 1/11)
    # for RGSpace(shape=(11, 11), distances=(1.3,1.3), harmonic=False)
    w_1_res = w_0_x * (1.3 * 1.3)
    # for RGSpace(shape=(11,11), distances=None, harmonic=True)
    w_2_res = w_0_x * (1.0 * 1.0)
    # for RGSpace(shape=(11,11), distances=(1.3, 1,3), harmonic=True)
    w_3_res = w_0_x * (1.3 * 1.3)
    return [
        [(11, 11), None, False, w_0_x, 1, None, False, w_0_res],
        [(11, 11), None, False, w_0_x.copy(), 1, None,  True, w_0_res],
        [(11, 11), (1.3, 1.3), False, w_0_x, 1, None, False, w_1_res],
        [(11, 11), (1.3, 1.3), False, w_0_x.copy(), 1, None,  True, w_1_res],
        [(11, 11), None, True, w_0_x, 1, None, False, w_2_res],
        [(11, 11), None, True, w_0_x.copy(), 1, None,  True, w_2_res],
        [(11, 11), (1.3, 1.3), True, w_0_x, 1, None, False, w_3_res],
        [(11, 11), (1.3, 1.3), True, w_0_x.copy(), 1, None,  True, w_3_res]
        ]


def get_hermitian_configs():
    h_0_x = np.array([
        [0.88250339+0.12102381j,  0.54293435+0.7345584j, 0.87057998+0.20515315j,
            0.16602950+0.09396132j],
        [0.83853902+0.17974696j,  0.79735933+0.37104425j, 0.22057732+0.9498977j,
            0.14329183+0.47899678j],
        [0.96934284+0.3792878j, 0.13118669+0.45643055j, 0.16372149+0.48235714j,
            0.66141537+0.20383357j],
        [0.49168197+0.77572178j, 0.09570420+0.14219071j, 0.69735595+0.33017333j,
            0.83692452+0.18544449j]])
    h_0_res_real = np.array([
        [0.88250339+0.j, 0.35448193+0.32029854j, 0.87057998+0.j,
            0.35448193-0.32029854j],
        [0.66511049-0.29798741j, 0.81714193+0.09279988j, 0.45896664+0.30986218j,
            0.11949801+0.16840303j],
        [0.96934284+0.j, 0.39630103+0.12629849j, 0.16372149+0.j,
            0.39630103-0.12629849j],
        [0.66511049+0.29798741j, 0.11949801-0.16840303j, 0.45896664-0.30986218j,
            0.81714193-0.09279988j]])
    h_0_res_imag = np.array([
        [0.12102381+0.j, 0.41425986-0.18845242j, 0.20515315+0.j,
            0.41425986+0.18845242j],
        [0.47773437-0.17342852j, 0.27824437+0.0197826j, 0.64003551+0.23838932j,
            0.31059374-0.02379381j],
        [0.37928780+0.j, 0.33013206+0.26511434j, 0.48235714+0.j,
            0.33013206-0.26511434j],
        [0.47773437+0.17342852j, 0.31059374+0.02379381j, 0.64003551-0.23838932j,
            0.27824437-0.0197826j]])*1j

    h_1_x = np.array([
        [[0.23987021+0.41617749j, 0.34605012+0.55462234j, 0.07947035+0.73360723j,
            0.22853748+0.39275304j],
         [0.90254910+0.02107809j, 0.28195470+0.56031588j, 0.23004043+0.33873536j,
             0.56398377+0.68913034j],
         [0.81897406+0.2050369j, 0.88724852+0.8137488j, 0.84645004+0.0059284j,
             0.14950377+0.50013099j]],
        [[0.93491597+0.73251066j, 0.74764790+0.11539037j, 0.48090736+0.04352568j,
            0.49363732+0.97233093j],
         [0.72761881+0.74636216j, 0.46390134+0.4343401j, 0.88436859+0.79415269j,
             0.67027606+0.85498234j],
         [0.86318727+0.19076379j, 0.36859448+0.89842333j, 0.73407193+0.85091112j,
             0.44187657+0.08936409j]]
        ])
    h_1_res_real = np.array([
        [[0.23987021+0.j, 0.28729380+0.08093465j, 0.07947035+0.j,
            0.28729380-0.08093465j],
         [0.90254910+0.j, 0.42296924-0.06440723j, 0.23004043+0.j,
             0.42296924+0.06440723j],
         [0.81897406+0.j, 0.51837614+0.1568089j, 0.84645004+0.j,
             0.51837614-0.1568089j]],
        [[0.93491597+0.j, 0.62064261-0.42847028j, 0.48090736+0.j,
            0.62064261+0.42847028j],
         [0.72761881+0.j, 0.56708870-0.21032112j, 0.88436859+0.j,
             0.56708870+0.21032112j],
         [0.86318727+0.j, 0.40523552+0.40452962j, 0.73407193+0.j,
             0.40523552-0.40452962j]]
        ])
    h_1_res_imag = np.array([
        [[0.41617749+0.j, 0.47368769-0.05875632j, 0.73360723+0.j,
            0.47368769+0.05875632j],
         [0.02107809+0.j, 0.62472311+0.14101454j, 0.33873536+0.j,
             0.62472311-0.14101454j],
         [0.20503690+0.j, 0.65693990-0.36887238j, 0.00592840+0.j,
             0.65693990+0.36887238j]],
        [[0.73251066+0.j, 0.54386065-0.12700529j, 0.04352568+0.j,
            0.54386065+0.12700529j],
         [0.74636216+0.j, 0.64466122+0.10318736j, 0.79415269+0.j,
             0.64466122-0.10318736j],
         [0.19076379+0.j, 0.49389371+0.03664104j, 0.85091112+0.j,
             0.49389371-0.03664104j]]
        ])*1j
    return [
        [h_0_x, None, h_0_res_real, h_0_res_imag],
        [h_1_x, (2,), h_1_res_real, h_1_res_imag]
    ]


class RGSpaceInterfaceTests(unittest.TestCase):
    @expand([['distances', tuple],
            ['zerocenter', tuple]])
    def test_property_ret_type(self, attribute, expected_type):
        x = RGSpace(1)
        assert_(isinstance(getattr(x, attribute), expected_type))


class RGSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, shape, zerocenter, distances,
                         harmonic, expected):
        x = RGSpace(shape, zerocenter, distances, harmonic)
        for key, value in expected.iteritems():
            assert_equal(getattr(x, key), value)

    @expand(get_hermitian_configs())
    def test_hermitian_decomposition(self, x, axes, real, imag):
        r = RGSpace(5)
        assert_almost_equal(r.hermitian_decomposition(x, axes=axes)[0], real)
        assert_almost_equal(r.hermitian_decomposition(x, axes=axes)[1], imag)

    @expand(get_distance_array_configs())
    def test_distance_array(self, shape, distances, zerocenter, expected):
        r = RGSpace(shape=shape, distances=distances, zerocenter=zerocenter)
        assert_almost_equal(r.get_distance_array('not'), expected)

    @expand(get_weight_configs())
    def test_weight(self, shape, distances, harmonic, x, power, axes,
                    inplace, expected):
        r = RGSpace(shape=shape, distances=distances, harmonic=harmonic)
        res = r.weight(x, power, axes, inplace)
        assert_almost_equal(res, expected)
        if inplace:
            assert_(x is res)
