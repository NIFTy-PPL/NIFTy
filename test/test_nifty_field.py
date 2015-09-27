# -*- coding: utf-8 -*-

from numpy.testing import assert_equal,\
    assert_almost_equal,\
    assert_raises

from nose_parameterized import parameterized
import unittest
import itertools
import numpy as np

from nifty import space,\
    point_space,\
    rg_space,\
    lm_space,\
    hp_space,\
    gl_space,\
    field

from nifty.nifty_core import POINT_DISTRIBUTION_STRATEGIES
from nifty.rg.nifty_rg import RG_DISTRIBUTION_STRATEGIES,\
                              gc as RG_GC
from nifty.lm.nifty_lm import LM_DISTRIBUTION_STRATEGIES,\
                              GL_DISTRIBUTION_STRATEGIES,\
                              HP_DISTRIBUTION_STRATEGIES


###############################################################################

def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )

###############################################################################
###############################################################################

all_point_datatypes = [np.dtype('bool'),
                       np.dtype('int16'),
                       np.dtype('int32'),
                       np.dtype('int64'),
                       np.dtype('float32'),
                       np.dtype('float64'),
                       np.dtype('complex64'),
                       np.dtype('complex128')]

all_lm_datatypes = [np.dtype('complex64'),
                    np.dtype('complex128')]

all_gl_datatypes = [np.dtype('float64'),
                    np.dtype('float128')]

all_hp_datatypes = [np.dtype('float64')]

###############################################################################

DATAMODELS = {}
DATAMODELS['point_space'] = ['np'] + POINT_DISTRIBUTION_STRATEGIES
DATAMODELS['rg_space'] = ['np'] + RG_DISTRIBUTION_STRATEGIES
DATAMODELS['lm_space'] = ['np'] + LM_DISTRIBUTION_STRATEGIES
DATAMODELS['gl_space'] = ['np'] + GL_DISTRIBUTION_STRATEGIES
DATAMODELS['hp_space'] = ['np'] + HP_DISTRIBUTION_STRATEGIES

###############################################################################

fft_modules = []
for name in ['gfft', 'gfft_dummy', 'pyfftw']:
    if RG_GC.validQ('fft_module', name):
        fft_modules += [name]

###############################################################################

point_like_spaces = ['point_space', 'rg_space', 'lm_space', 'hp_space',
                     'gl_space']

###############################################################################

space_list = []

# Add point_spaces
for param in itertools.product([1, 10],
                               all_point_datatypes,
                               DATAMODELS['point_space']):
    space_list += [[point_space(num=param[0],
                               dtype=param[1],
                               datamodel=param[2])]]

# Add rg_spaces
for param in itertools.product([(1,), (4, 6), (5, 8)],
                               [False, True],
                               [0, 1, 2],
                               [None, 0.3],
                               [False, True],
                               DATAMODELS['rg_space'],
                               fft_modules):
    space_list += [[rg_space(shape=param[0],
                            zerocenter=param[1],
                            complexity=param[2],
                            distances=param[3],
                            harmonic=param[4],
                            datamodel=param[5],
                            fft_module=param[6])]]


###############################################################################
###############################################################################

class Test_field_init(unittest.TestCase):

    @parameterized.expand(space_list)
    def test_successfull_init_and_attributes(self, s):
        s = s[0]
        f = field(s)
        assert(f.domain is s)
        assert(s.check_codomain(f.codomain))









































