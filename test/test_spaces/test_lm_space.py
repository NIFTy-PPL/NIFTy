import unittest
import numpy as np

from numpy.testing import assert_, assert_equal, assert_raises
from nose.plugins.skip import SkipTest
from nifty import LMSpace
from nifty.config import dependency_injector as di
from test.common import expand

# [lmax, dtype, expected]
INIT_CONFIGS = [
        [5, None, {
            'lmax': 5,
            'mmax': 5,
            'shape': (36,),
            'harmonic': True,
            'dim': 36,
            'total_volume': 36.0,
            'dtype': np.dtype('float64')
            }],
        [7, np.dtype('float64'), {
            'lmax': 7,
            'mmax': 7,
            'shape': (64,),
            'harmonic': True,
            'dim': 64,
            'total_volume': 64.0,
            'dtype': np.dtype('float64')
            }],
        [-1, None, {
            'error': ValueError
            }]
    ]


class LMSpaceIntefaceTests(unittest.TestCase):
    @expand([['lmax', int],
            ['mmax', int],
            ['dim', int]])
    def test_properties(self, attribute, expected_type):
        try:
            x = LMSpace(7)
        except ImportError:
            raise SkipTest
        assert_(isinstance(getattr(x, attribute), expected_type))


class LMSpaceFunctionalityTests(unittest.TestCase):
    @expand(INIT_CONFIGS)
    def test_constructor(self, lmax, dtype, expected):
        if 'libsharp_wrapper_gl' not in di or 'healpy' not in di:
            raise SkipTest
        else:
            if 'error' in expected:
                with assert_raises(expected['error']):
                    LMSpace(lmax, dtype)
            else:
                l = LMSpace(lmax, dtype)
                for key, value in expected.iteritems():
                    assert_equal(getattr(l, key), value)

    def test_hermitian_decomposition(self):
        pass

    def test_weight(self):
        pass

    def test_distance_array(self):
        pass
