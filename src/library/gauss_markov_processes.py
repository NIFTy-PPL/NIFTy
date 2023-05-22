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
# Copyright(C) 2013-2023 Max-Planck-Society
# Authors: Philipp Frank, Vincent Eberle
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from functools import reduce
from ..operators.endomorphic_operator import EndomorphicOperator
from ..operators.scaling_operator import ScalingOperator
from ..domains.rg_space import RGSpace
from ..domains.irg_space import IRGSpace
from ..extra import is_fieldlike
from ..sugar import makeOp, makeDomain, makeField


class _CumsumOperator(EndomorphicOperator):
    def __init__(self, domain, space = None):
        self._domain = makeDomain(domain)
        if space is None:
            space  = 0
        intdom = self._domain[space]
        if isinstance(intdom, RGSpace):
            if not len(intdom.distances) == 1:
                raise ValueError("Integration domain must be 1D")
            self._wgts = np.ones(intdom.shape)*intdom.distances[0]
        elif isinstance(intdom, IRGSpace):
            self._wgts = intdom.dvol
        else:
            raise ValueError("Integration domain of incorrect type!")
        self._wgts = np.sqrt(self._wgts)

        self._axis = reduce(lambda a,b:a+b, (len(dd.shape) for dd in 
                                            self._domain[:space]))
        _back = reduce(lambda a,b:a+b, (len(dd.shape) for dd in 
                                       self._domain[(space+1):]))
        self._wgts = np.expand_dims(self._wgts, 
                                    axis=tuple(i for i in range(self._axis)) + 
                                    tuple(-(i+1) for i in range(_back)))

        self._capabilities = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = self._wgts*x.val
        if mode == self.ADJOINT_TIMES:
            x = np.flip(x, axis=self._axis)
        res = np.cumsum(x, axis=self._axis)
        if mode == self.ADJOINT_TIMES:
            res = np.flip(res, axis=self._axis)
        return makeField(self._domain, res)


def WPPrior(Amplitude, key = 'xi', space = None):
    if is_fieldlike(Amplitude):
        wp = makeOp(Amplitude).ducktape(key)
    else:
        wp = Amplitude * ScalingOperator(Amplitude.target, 1).ducktape(key)
    return _CumsumOperator(wp.target, space=space) @ wp