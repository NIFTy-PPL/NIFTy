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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function
from ..compat import *
import numpy as np
from ..sugar import from_random
from ..linearization import Linearization

__all__ = ["check_value_gradient_consistency",
           "check_value_gradient_metric_consistency"]


def _get_acceptable_location(op, loc, lin):
    val = lin.val
    if not np.isfinite(val.sum()):
        raise ValueError('Initial value must be finite')
    dir = from_random("normal", loc.domain)
    dirder = lin.jac(dir)
    if dirder.norm() == 0:
        dir = dir * val.norm() * 1e-5
    else:
        dir = dir * val.norm() * (1e-5/dirder.norm())
    # Find a step length that leads to a "reasonable" location
    for i in range(50):
        try:
            loc2 = loc+dir
            lin2 = op(Linearization.make_var(loc2))
            if np.isfinite(lin2.val.sum()) and abs(lin2.val.sum()) < 1e20:
                break
        except FloatingPointError:
            pass
        dir = dir*0.5
    else:
        raise ValueError("could not find a reasonable initial step")
    return loc2, lin2


def check_value_gradient_consistency(op, loc, tol=1e-8, ntries=100):
    for _ in range(ntries):
        lin = op(Linearization.make_var(loc))
        loc2, lin2 = _get_acceptable_location(op, loc, lin)
        val = lin.val
        dir = loc2 - loc
        locnext = loc2
        dirnorm = dir.norm()
        for i in range(50):
            locmid = loc + 0.5*dir
            linmid = op(Linearization.make_var(locmid))
            dirder = linmid.jac(dir)/dirnorm
            numgrad = (lin2.val-val)/dirnorm
            xtol = tol * dirder.norm() / np.sqrt(dirder.size)
            if (abs(numgrad-dirder) <= xtol).all():
                break
            dir = dir*0.5
            dirnorm *= 0.5
            loc2 = locmid
            lin2 = linmid
        else:
            raise ValueError("gradient and value seem inconsistent")
        loc = locnext


def check_value_gradient_metric_consistency(op, loc, tol=1e-8, ntries=100):
    for _ in range(ntries):
        lin = op(Linearization.make_var(loc))
        loc2, lin2 = _get_acceptable_location(op, loc, lin)
        val = lin.val
        dir = loc2 - loc
        locnext = loc2
        dirnorm = dir.norm()
        for i in range(50):
            locmid = loc + 0.5*dir
            linmid = op(Linearization.make_var(locmid))
            dirder = linmid.jac(dir)/dirnorm
            numgrad = (lin2.val-val)/dirnorm
            dgrad = linmid.metric(dir)/dirnorm
            dgrad2 = (lin2.gradient-lin.gradient)/dirnorm
            xtol = tol * dirder.norm() / np.sqrt(dirder.size)
            if ((abs(numgrad-dirder) <= xtol).all() and
                    (abs(dgrad-dgrad2) <= xtol).all()):
                break
            dir = dir*0.5
            dirnorm *= 0.5
            loc2 = locmid
            lin2 = linmid
        else:
            raise ValueError("gradient and value seem inconsistent")
        loc = locnext
