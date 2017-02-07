import unittest
import numpy as np

from numpy.testing import assert_, assert_equal, assert_raises,\
                          assert_almost_equal
from nose.plugins.skip import SkipTest
from nifty import HPSpace
from nifty.config import dependency_injector as di
from test.common import expand

# [nside, dtype, expected]
CONSTRUCTOR_CONFIGS = [
        [2, None, {
            'nside': 2,
            'harmonic': False,
            'shape': (48,),
            'dim': 48,
            'total_volume': 4 * np.pi,
            'dtype': np.dtype('float64')
            }],
        [5, None, {
            'error': ValueError
            }],
        [1, None, {
            'error': ValueError
            }]
    ]


def get_distance_array_configs():
    npzfile = np.load('test/data/hp_space.npz')
    return [[2, None, npzfile['da_0']]]


def get_weight_configs():
    npzfile = np.load('test/data/hp_space.npz')
    return [
        [npzfile['w_0_x'], 1, None, False, npzfile['w_0_res']],
        [npzfile['w_0_x'], 1, None, True, npzfile['w_0_res']],
        [npzfile['w_1_x'], 2, None, False, npzfile['w_1_res']],
        ]


class HPSpaceInterfaceTests(unittest.TestCase):
    @expand([['nside', int]])
    def test_property_ret_type(self, attribute, expected_type):
        try:
            x = HPSpace()
        except ImportError:
            raise SkipTest
        assert_(isinstance(getattr(x, attribute), expected_type))


class HPSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, nside, dtype, expected):
        if 'healpy' not in di:
            raise SkipTest
        else:
            if 'error' in expected:
                with assert_raises(expected['error']):
                    HPSpace(nside, dtype)
            else:
                h = HPSpace(nside, dtype)
                for key, value in expected.iteritems():
                    assert_equal(getattr(h, key), value)

    @expand(get_weight_configs())
    def test_weight(self, x, power, axes, inplace, expected):
        if 'healpy' not in di:
            raise SkipTest
        else:
            h = HPSpace(2)
            res = h.weight(x, power, axes, inplace)
            assert_almost_equal(res, expected)
            if inplace:
                assert_(x is res)

    @expand(get_distance_array_configs())
    def test_distance_array(self, nside, dtype, expected):
        if 'healpy' not in di:
            raise SkipTest
        else:
            h = HPSpace(nside, dtype)
            assert_almost_equal(h.get_distance_array('not').data, expected)
