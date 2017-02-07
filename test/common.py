import d2o
import numpy as np
from nose_parameterized import parameterized
from nifty import RGSpace, LMSpace, HPSpace, GLSpace
from string import strip


def pretty_str(obj):
    if type(obj) == list:
        return " ".join(pretty_str(x) for x in obj)
    if type(obj) == RGSpace:
        return type(obj).__name__
    elif type(obj) == LMSpace:
        return type(obj).__name__
    elif type(obj) == HPSpace:
        return type(obj).__name__
    elif type(obj) == GLSpace:
        return type(obj).__name__
    elif isinstance(obj, d2o.distributed_data_object):
        return 'd2o'
    elif type(obj) == dict:
        if 'error' in obj:
            return 'error_' + obj['error'].__name__
        else:
            return ''
    elif type(obj) == np.ndarray:
        return 'DATA'
    else:
        return str(obj)


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        strip(parameterized.to_safe_name(
            " ".join(pretty_str(x) for x in param.args)), '_')
    )


def expand(*args, **kwargs):
    return parameterized.expand(*args, testcase_func_name=custom_name_func,
                                **kwargs)
