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
from nifty import Space, RGSpace, LMSpace, HPSpace, GLSpace, PowerSpace
from nifty.config import dependency_injector as gdi
import numpy as np


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
              LMSpace(5), HPSpace(4)]
    if 'pyHealpix' in gdi:
        spaces.append(GLSpace(4))
    return spaces


def generate_harmonic_spaces():
    spaces = [RGSpace(4, harmonic=True), LMSpace(5)]
    return spaces


def marco_binbounds(space, logarithmic, nbin=None):
    """Only for testing purposes. DO NOT USE IN REAL LIFE!"""
    if logarithmic is None and nbin is None:
        return None
    if not (isinstance(space, Space) and space.harmonic):
        raise ValueError("space must be a harmonic space.")
    logarithmic = bool(logarithmic)
    if nbin is not None:
        nbin = int(nbin)
        assert nbin >= 3, "nbin must be at least 3"
    # equidistant binning (linear or log)
    # MR FIXME: this needs to improve
    kindex = space.get_unique_distances()
    if (logarithmic):
        k = np.r_[0, np.log(kindex[1:])]
    else:
        k = kindex
    dk = np.max(k[2:] - k[1:-1])  # minimum dk to avoid empty bins
    if(nbin is None):
        nbin = int((k[-1] - 0.5 * (k[2] + k[1])) /
                   dk - 0.5)  # maximal nbin
    else:
        nbin = min(int(nbin), int(
            (k[-1] - 0.5 * (k[2] + k[1])) / dk + 2.5))
        dk = (k[-1] - 0.5 * (k[2] + k[1])) / (nbin - 2.5)
    bb = np.r_[0.5 * (3 * k[1] - k[2]),
               0.5 * (k[1] + k[2]) + dk * np.arange(nbin-2)]
    if(logarithmic):
        bb = np.exp(bb)
    return bb
