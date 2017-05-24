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

import d2o
import numpy as np
from nose_parameterized import parameterized
from nifty import RGSpace, LMSpace, HPSpace, GLSpace, PowerSpace
from nifty.config import dependency_injector as di
from string import strip


def pretty_str(obj):
    if type(obj) == list:
        return " ".join(pretty_str(x) for x in obj)
    if type(obj) == tuple:
        return " ".join(pretty_str(x) for x in obj)
    if type(obj) == RGSpace:
        return type(obj).__name__
    elif type(obj) == LMSpace:
        return type(obj).__name__
    elif type(obj) == HPSpace:
        return type(obj).__name__
    elif type(obj) == GLSpace:
        return type(obj).__name__
    elif type(obj) == PowerSpace:
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


def generate_spaces():
    spaces = [RGSpace(4), PowerSpace(RGSpace((4, 4), harmonic=True)),
              LMSpace(5), HPSpace(4)]
    if 'pyHealpix' in di:
        spaces.append(GLSpace(4))
    return spaces


def generate_harmonic_spaces():
    spaces = [RGSpace(4, harmonic=True), LMSpace(5)]
    return spaces
