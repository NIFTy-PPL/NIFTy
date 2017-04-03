import unittest
import numpy as np

from numpy.testing import assert_, assert_equal, assert_raises,\
        assert_almost_equal
from nose.plugins.skip import SkipTest
from nifty import GLSpace
from nifty.config import dependency_injector as di
from test.common import expand

# [nlat, nlon, dtype, expected]
CONSTRUCTOR_CONFIGS = [
        [2, None, None, {
            'nlat': 2,
            'nlon': 3,
            'harmonic': False,
            'shape': (6,),
            'dim': 6,
            'total_volume': 4 * np.pi,
            'dtype': np.dtype('float64')
            }],
        [0, None, None, {
            'error': ValueError
            }]
    ]


def get_distance_array_configs():
    npzfile = np.load('test/data/gl_space.npz')
    return [[2, None, None, npzfile['da_0']]]


def get_weight_configs():
    npzfile = np.load('test/data/gl_space.npz')
    return [
        [npzfile['w_0_x'], 1, None, False, npzfile['w_0_res']],
        [npzfile['w_0_x'], 1, None, True, npzfile['w_0_res']],
        [npzfile['w_1_x'], 1, (2,), True, npzfile['w_1_res']],
        ]


class GLSpaceInterfaceTests(unittest.TestCase):
    @expand([['nlat', int],
            ['nlon', int]])
    def test_property_ret_type(self, attribute, expected_type):
        try:
            g = GLSpace()
        except ImportError:
            raise SkipTest
        assert_(isinstance(getattr(g, attribute), expected_type))


class GLSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, nlat, nlon, dtype, expected):
        if 'libsharp_wrapper_gl' not in di:
            raise SkipTest
        else:
            if 'error' in expected:
                with assert_raises(expected['error']):
                    GLSpace(nlat, nlon, dtype)
            else:
                g = GLSpace(nlat, nlon, dtype)
                for key, value in expected.iteritems():
                    assert_equal(getattr(g, key), value)

    @expand(get_weight_configs())
    def test_weight(self, x, power, axes, inplace, expected):
        if 'libsharp_wrapper_gl' not in di:
            raise SkipTest
        else:
            g = GLSpace(2)
            res = g.weight(x, power, axes, inplace)
            assert_almost_equal(res, expected)
            if inplace:
                assert_(x is res)

    @expand(get_distance_array_configs())
    def test_distance_array(self, nlat, nlon, dtype, expected):
        if 'libsharp_wrapper_gl' not in di:
            raise SkipTest
        else:
            g = GLSpace(nlat, nlon, dtype)
            assert_almost_equal(g.get_distance_array('not').data, expected)
