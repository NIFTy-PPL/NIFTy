from __future__ import division

import unittest
import numpy as np

from d2o import distributed_data_object
from numpy.testing import assert_, assert_equal, assert_almost_equal,\
        assert_raises
from nifty import PowerSpace, RGSpace, LMSpace, Space
from types import NoneType
from test.common import expand

# [harmonic_domain, distribution_strategy,
#  log, nbin, binbounds, dtype, expected]
CONSTRUCTOR_CONFIGS = [
    [1, 'not', False, None, None, None, {'error': ValueError}],
    [RGSpace((8,)), 'not', False, None, None, None, {'error': ValueError}],
    [RGSpace((8,), harmonic=True), 'not', False, None, None, None, {
        'harmonic': True,
        'shape': (5,),
        'dim': 5,
        'total_volume': 8.0,
        'harmonic_domain': RGSpace((8,), harmonic=True),
        'log': False,
        'nbin': None,
        'binbounds': None,
        'pindex': distributed_data_object([0, 1, 2, 3, 4, 3, 2, 1]),
        'kindex': np.array([0., 1., 2., 3., 4.]),
        'rho': np.array([1, 2, 2, 2, 1]),
        'pundex': np.array([0, 1, 2, 3, 4]),
        'k_array': np.array([0., 1., 2., 3., 4., 3., 2., 1.]),
        'dtype': np.dtype('float64')
        }],
    [RGSpace((8,), harmonic=True), 'not', True, None, None, None, {
        'harmonic': True,
        'shape': (2,),
        'dim': 2,
        'total_volume': 8.0,
        'harmonic_domain': RGSpace((8,), harmonic=True),
        'log': True,
        'nbin': None,
        'binbounds': None,
        'pindex': distributed_data_object([0, 1, 1, 1, 1, 1, 1, 1]),
        'kindex': np.array([0., 2.28571429]),
        'rho': np.array([1, 7]),
        'pundex': np.array([0, 1]),
        'k_array': np.array([0., 2.28571429, 2.28571429, 2.28571429,
                             2.28571429, 2.28571429, 2.28571429, 2.28571429]),
        'dtype': np.dtype('float64')
        }],
    ]


def get_distance_array_configs():
    npzfile = np.load('test/data/power_space.npz')
    return [
        [RGSpace((4, 4), harmonic=True),  npzfile['da_0']],
        ]


def get_weight_configs():
    npzfile = np.load('test/data/power_space.npz')
    return [
        [RGSpace((4, 4), harmonic=True),
            npzfile['w_0_x'], 1, (2,), False, npzfile['w_0_res']],
        [RGSpace((4, 4), harmonic=True),
            npzfile['w_0_x'], 1, (2,), True, npzfile['w_0_res']],
        ]


class PowerSpaceInterfaceTest(unittest.TestCase):
    @expand([
        ['harmonic_domain', Space],
        ['log', bool],
        ['nbin', (int, NoneType)],
        ['binbounds', (list, NoneType)],
        ['pindex', distributed_data_object],
        ['kindex', np.ndarray],
        ['rho', np.ndarray],
        ['pundex', np.ndarray],
        ['k_array', distributed_data_object],
        ])
    def test_property_ret_type(self, attribute, expected_type):
        r = RGSpace((4, 4), harmonic=True)
        p = PowerSpace(r)
        assert_(isinstance(getattr(p, attribute), expected_type))


class PowerSpaceFunctionalityTest(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, harmonic_domain, distribution_strategy, log,
                         nbin, binbounds, dtype, expected):
        if 'error' in expected:
            with assert_raises(expected['error']):
                PowerSpace(harmonic_domain=harmonic_domain,
                           distribution_strategy=distribution_strategy,
                           log=log, nbin=nbin, binbounds=binbounds,
                           dtype=dtype)
        else:
            p = PowerSpace(harmonic_domain=harmonic_domain,
                           distribution_strategy=distribution_strategy,
                           log=log, nbin=nbin, binbounds=binbounds,
                           dtype=dtype)
            for key, value in expected.iteritems():
                if isinstance(value, np.ndarray):
                    assert_almost_equal(getattr(p, key), value)
                else:
                    assert_equal(getattr(p, key), value)

    @expand(get_distance_array_configs())
    def test_distance_array(self, harmonic_domain, expected):
        p = PowerSpace(harmonic_domain=harmonic_domain)
        assert_almost_equal(p.get_distance_array('not'), expected)

    @expand(get_weight_configs())
    def test_weight(self, harmonic_domain, x, power, axes,
                    inplace, expected):
        p = PowerSpace(harmonic_domain=harmonic_domain)
        res = p.weight(x, power, axes, inplace)
        assert_almost_equal(res, expected)
        if inplace:
            assert_(x is res)
