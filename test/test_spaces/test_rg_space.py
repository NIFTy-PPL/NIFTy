from __future__ import division

import unittest
import numpy as np

from numpy.testing import assert_, assert_equal
from nifty import RGSpace
from test.common import expand

# [shape, zerocenter, distances, harmonic, dtype, expected]
INIT_CONFIGS = [
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


class RGSpaceInterfaceTests(unittest.TestCase):
    @expand([['distances', tuple],
            ['zerocenter', tuple]])
    def test_properties(self, attribute, expected_type):
        x = RGSpace()
        assert_(isinstance(getattr(x, attribute), expected_type))


class RGSpaceFunctionalityTests(unittest.TestCase):
    @expand(INIT_CONFIGS)
    def test_constructor(self, shape, zerocenter, distances,
                         harmonic, dtype, expected):
        x = RGSpace(shape, zerocenter, distances, harmonic, dtype)
        for key, value in expected.iteritems():
            assert_equal(getattr(x, key), value)

    def test_hermitian_decomposition(self):
        pass

    def test_weight(self):
        pass

    def test_distance_array(self):
        pass
