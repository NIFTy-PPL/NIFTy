from __future__ import division

import unittest
import numpy as np

from numpy.testing import assert_, assert_equal, assert_almost_equal
from nifty import RGSpace
from test.common import expand

# [shape, zerocenter, distances, harmonic, dtype, expected]
CONSTRUCTOR_CONFIGS = [
        [(8,), False, None, False, None,
            {
                'shape': (8,),
                'zerocenter': (False,),
                'distances': (0.125,),
                'harmonic': False,
                'dtype': np.dtype('float'),
                'dim': 8,
                'total_volume': 1.0
            }],
        [(8,), True, None, False, None,
            {
                'shape': (8,),
                'zerocenter': (True,),
                'distances': (0.125,),
                'harmonic': False,
                'dtype': np.dtype('float'),
                'dim': 8,
                'total_volume': 1.0
            }],
        [(8,), False, None, True, None,
            {
                'shape': (8,),
                'zerocenter': (False,),
                'distances': (1.0,),
                'harmonic': True,
                'dtype': np.dtype('complex'),
                'dim': 8,
                'total_volume': 8.0
            }],
        [(8,), False, (12,), True, None,
            {
                'shape': (8,),
                'zerocenter': (False,),
                'distances': (12.0,),
                'harmonic': True,
                'dtype': np.dtype('complex'),
                'dim': 8,
                'total_volume': 96.0
            }],
        [(11, 11), (False, True), None, False, None,
            {
                'shape': (11, 11),
                'zerocenter': (False, True),
                'distances': (1/11, 1/11),
                'harmonic': False,
                'dtype': np.dtype('float'),
                'dim': 121,
                'total_volume': 1.0
            }],
        [(11, 11), True, (1.3, 1.3), True, None,
            {
                'shape': (11, 11),
                'zerocenter': (True, True),
                'distances': (1.3, 1.3),
                'harmonic': True,
                'dtype': np.dtype('complex'),
                'dim': 121,
                'total_volume': 204.49
            }]

    ]


def get_distance_array_configs():
    npzfile = np.load('test/data/rg_space.npz')
    return [
        [(4, 4),  None, [False, False], npzfile['da_0']],
        [(4, 4),  None, [True, True], npzfile['da_1']],
        [(4, 4),  (12, 12), [True, True], npzfile['da_2']]
        ]


def get_weight_configs():
    npzfile = np.load('test/data/rg_space.npz')
    return [
        [(11, 11), None, False,
            npzfile['w_0_x'], 1, None, False, npzfile['w_0_res']],
        [(11, 11), None, False,
            npzfile['w_0_x'], 1, None, True, npzfile['w_0_res']],
        [(11, 11), (1.3, 1.3), False,
            npzfile['w_0_x'], 1, None, False, npzfile['w_1_res']],
        [(11, 11), (1.3, 1.3), False,
            npzfile['w_0_x'], 1, None, True, npzfile['w_1_res']],
        [(11, 11), None, True,
            npzfile['w_0_x'], 1, None, False, npzfile['w_2_res']],
        [(11, 11), None, True,
            npzfile['w_0_x'], 1, None, True, npzfile['w_2_res']],
        [(11, 11), (1.3, 1.3), True,
            npzfile['w_0_x'], 1, None, False, npzfile['w_3_res']],
        [(11, 11), (1.3, 1.3), True,
            npzfile['w_0_x'], 1, None, True, npzfile['w_3_res']]
        ]


def get_hermitian_configs():
    npzfile = np.load('test/data/rg_space.npz')
    return [
        [npzfile['h_0_x'], None,
            npzfile['h_0_res_real'], npzfile['h_0_res_imag']],
        [npzfile['h_1_x'], (2,),
            npzfile['h_1_res_real'], npzfile['h_1_res_imag']]
    ]


class RGSpaceInterfaceTests(unittest.TestCase):
    @expand([['distances', tuple],
            ['zerocenter', tuple]])
    def test_property_ret_type(self, attribute, expected_type):
        x = RGSpace()
        assert_(isinstance(getattr(x, attribute), expected_type))


class RGSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, shape, zerocenter, distances,
                         harmonic, dtype, expected):
        x = RGSpace(shape, zerocenter, distances, harmonic, dtype)
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
