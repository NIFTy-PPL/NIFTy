# -*- coding: utf-8 -*-

from numpy.testing import assert_equal,\
    assert_almost_equal,\
    assert_raises

from nose_parameterized import parameterized
import unittest
import itertools
import numpy as np

from nifty import Space,\
                  RgSpace,\
                  Field,\
                  distributed_data_object

from nifty.operators import operator,\
                            diagonal_operator,\
                            power_operator,\
                            projection_operator,\
                            vecvec_operator,\
                            response_operator,\
                            invertible_operator,\
                            propagator_operator,\
                            identity_operator

from nifty.nifty_core import POINT_DISTRIBUTION_STRATEGIES

from nifty.rg.nifty_rg import RG_DISTRIBUTION_STRATEGIES,\
                              gc as RG_GC
from nifty.lm.nifty_lm import LM_DISTRIBUTION_STRATEGIES,\
                              GL_DISTRIBUTION_STRATEGIES,\
                              HP_DISTRIBUTION_STRATEGIES

available = []
try:
    from nifty import LmSpace
except ImportError:
    pass
else:
    available += ['lm_space']
try:
    from nifty import  GlSpace
except ImportError:
    pass
else:
    available += ['gl_space']
try:
    from nifty import  HpSpace
except ImportError:
    pass
else:
    available += ['hp_space']



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

all_spaces = ['space', 'point_space', 'rg_space']
if 'lm_space' in available:
    all_spaces += ['lm_space']
if 'gl_space' in available:
    all_spaces += ['gl_space']
if 'hp_space' in available:
    all_spaces += ['hp_space']


point_like_spaces = ['point_space', 'rg_space']
if 'lm_space' in available:
    point_like_spaces += ['lm_space']
if 'gl_space' in available:
    point_like_spaces += ['gl_space']
if 'hp_space' in available:
    point_like_spaces += ['hp_space']

###############################################################################

np_spaces = point_like_spaces
d2o_spaces = []
if POINT_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['point_space']
if RG_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['rg_space']
if LM_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['lm_space']
if GL_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['gl_space']
if HP_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['hp_space']
###############################################################################

all_operators = ['operator', 'diagonal_operator', 'power_operator',
                 'projection_operator', 'vecvec_operator', 'response_operator',
                 'invertible_operator', 'propagator_operator',
                 'identity_operator']


###############################################################################

def generate_operator(name):
    x = RgSpace((8, 8))
    k = x.get_codomain()
    operator_dict = {'operator': operator(domain=x),
                     'diagonal_operator': diagonal_operator(domain=x),
                     'identity_operator': identity_operator(domain=x),
                     'power_operator': power_operator(domain=k),
                     'projection_operator': projection_operator(domain=x),
                     'vecvec_operator': vecvec_operator(domain=x),
                     'response_operator': response_operator(domain=x),
                     'invertible_operator': invertible_operator(domain=x)}
    return operator_dict[name]


def generate_data(space):
    a = np.arange(space.dim).reshape(space.shape)
    data = space.cast(a)
    return data


def check_equality(space, data1, data2):
    return space.unary_operation(space.binary_operation(data1, data2, 'eq'),
                                 'all')


def check_almost_equality(space, data1, data2, integers=7):
    return space.unary_operation(
        space.binary_operation(
            space.unary_operation(
                space.binary_operation(data1, data2, 'sub'),
                'abs'),
            10.**(-1. * integers), 'le'),
        'all')


###############################################################################
###############################################################################

class Test_Operator_Base_Class(unittest.TestCase):

    @parameterized.expand(all_operators,
                          testcase_func_name=custom_name_func)
    def test_successfull_init_and_methods(self, name):
        op = generate_operator(name)
        assert(callable(op.set_val))
        assert(callable(op.get_val))
        assert(callable(op._multiply))
        assert(callable(op._adjoint_multiply))
        assert(callable(op._inverse_multiply))
        assert(callable(op._adjoint_inverse_multiply))
        assert(callable(op._inverse_adjoint_multiply))
        assert(callable(op._briefing))
        assert(callable(op._debriefing))
        assert(callable(op.times))
        assert(callable(op.__call__))
        assert(callable(op.adjoint_times))
        assert(callable(op.inverse_times))
        assert(callable(op.adjoint_inverse_times))
        assert(callable(op.inverse_adjoint_times))
        assert(callable(op.tr))
        assert(callable(op.inverse_tr))
        assert(callable(op.diag))
        assert(callable(op.det))
        assert(callable(op.inverse_det))
        assert(callable(op.log_det))
        assert(callable(op.tr_log))
        assert(callable(op.hat))
        assert(callable(op.inverse_hat))
        assert(callable(op.hathat))
        assert(callable(op.inverse_hathat))
        assert(callable(op.__repr__))

    @parameterized.expand(all_operators,
                          testcase_func_name=custom_name_func)
    def test_successfull_init_and_attributes(self, name):
        op = generate_operator(name)
        assert(isinstance(op.sym, bool))
        assert(isinstance(op.uni, bool))
        assert(isinstance(op.bare, bool))
        assert(isinstance(op.imp, bool))

        assert(isinstance(op.domain, space))
        assert(isinstance(op.codomain, space))
        assert(isinstance(op.target, space))
        assert(isinstance(op.cotarget, space))























































