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

import numpy as np

from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.log_rg_space import LogRGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from .linear_operator import LinearOperator


class SlopeOperator(LinearOperator):
    """
    Creates a slope on target.

    This operator creates a field on a LogRGSpace, which is created
    according to a slope of given entries, (mean, y-intercept).
    The slope mean is the power law of the field in normal-space.

    Parameters
    ----------
    domain : domain or DomainTuple, shape=(2,)
        It has to be an UnstructuredDomain.
        The domain of the slope mean and the y-intercept mean.
    target : domain or DomainTuple
        The output domain has to a LogRGSpace
    sigmas : np.array, shape=(2,)
        The slope variance and the y-intercept variance.
    """
    def __init__(self, target):
        if not isinstance(target, LogRGSpace):
            raise TypeError
        self._domain = DomainTuple.make(UnstructuredDomain((2,)))
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        self.ndim = len(self.target[0].shape)
        

        if self.ndim != 1:
            raise ValueError("Slope Operator only works for ndim == 1")
        
        # Prepare pos
        self.pos = self.target[0].get_k_array()-self.target[0].t_0[0]
        
        

    def apply(self, x, mode):
        self._check_input(x, mode)

        # Times
        if mode == self.TIMES:
            inp = x.to_global_data()
            res = inp[-1]
            for i in range(self.ndim):
                res = res + inp[i] * self.pos[i]
            res[0] = 0.
            return Field.from_global_data(self.target, res)

        # Adjoint times
        res = np.zeros(self.domain[0].shape, dtype=x.dtype)
        xglob = x.to_global_data()
        res[-1] = np.sum(xglob[1:])
        for i in range(self.ndim):
            res[i] = np.sum((self.pos[i] * xglob)[1:])
        return Field.from_global_data(self.domain, res)
