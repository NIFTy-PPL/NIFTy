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

from builtins import str
from parameterized import parameterized
from nifty import RGSpace, LMSpace, HPSpace, GLSpace, PowerSpace


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


def expand(*args, **kwargs):
    return parameterized.expand(*args, testcase_func_name=custom_name_func,
                                **kwargs)


def generate_spaces():
    spaces = [RGSpace(4), PowerSpace(RGSpace((4, 4), harmonic=True)),
              LMSpace(5), HPSpace(4), GLSpace(4)]
    return spaces


def generate_harmonic_spaces():
    spaces = [RGSpace(4, harmonic=True), LMSpace(5)]
    return spaces
