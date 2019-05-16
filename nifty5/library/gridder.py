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
# Copyright(C) 2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..fft import hartley
from ..operators.linear_operator import LinearOperator
from ..sugar import from_global_data, makeDomain


class GridderMaker(object):
    def __init__(self, domain, eps=1e-15):
        domain = makeDomain(domain)
        if (len(domain) != 1 or not isinstance(domain[0], RGSpace) or
                not len(domain.shape) == 2):
            raise ValueError("need domain with exactly one 2D RGSpace")
        nu, nv = domain.shape
        if nu % 2 != 0 or nv % 2 != 0:
            raise ValueError("dimensions must be even")
        nu2, nv2 = 2*nu, 2*nv
        w = int(-np.log10(eps)+1.9999)
        nsafe = (w+1)//2
        nu2 = max([nu2, 2*nsafe])
        nv2 = max([nv2, 2*nsafe])

        oversampled_domain = RGSpace(
            [nu2, nv2], distances=[1, 1], harmonic=False)

        self._eps = eps
        self._rest = _RestOperator(domain, oversampled_domain, eps)

    def getReordering(self, uv):
        from nifty_gridder import peanoindex
        nu2, nv2 = self._rest._domain.shape
        return peanoindex(uv, nu2, nv2)

    def getGridder(self, uv):
        return RadioGridder(self._rest.domain, self._eps, uv)

    def getRest(self):
        return self._rest

    def getFull(self, uv):
        return self.getRest() @ self.getGridder(uv)


class _RestOperator(LinearOperator):
    def __init__(self, domain, oversampled_domain, eps):
        from nifty_gridder import correction_factors
        self._domain = makeDomain(oversampled_domain)
        self._target = domain
        nu, nv = domain.shape
        nu2, nv2 = oversampled_domain.shape

        fu = correction_factors(nu2, nu//2+1, eps)
        fv = correction_factors(nv2, nv//2+1, eps)
        # compute deconvolution operator
        rng = np.arange(nu)
        k = np.minimum(rng, nu-rng)
        self._deconv_u = np.roll(fu[k], -nu//2).reshape((-1, 1))
        rng = np.arange(nv)
        k = np.minimum(rng, nv-rng)
        self._deconv_v = np.roll(fv[k], -nv//2).reshape((1, -1))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        nu, nv = self._target.shape
        res = x.to_global_data()
        if mode == self.TIMES:
            res = hartley(res)
            res = np.roll(res, (nu//2, nv//2), axis=(0, 1))
            res = res[:nu, :nv]
            res *= self._deconv_u
            res *= self._deconv_v
        else:
            res = res*self._deconv_u
            res *= self._deconv_v
            nu2, nv2 = self._domain.shape
            res = np.pad(res, ((0, nu2-nu), (0, nv2-nv)), mode='constant',
                         constant_values=0)
            res = np.roll(res, (-nu//2, -nv//2), axis=(0, 1))
            res = hartley(res)
        return from_global_data(self._tgt(mode), res)


class RadioGridder(LinearOperator):
    def __init__(self, target, eps, uv):
        self._domain = DomainTuple.make(
            UnstructuredDomain((uv.shape[0],)))
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._eps = float(eps)
        self._uv = uv  # FIXME: should we write-protect this?

    def apply(self, x, mode):
        from nifty_gridder import (to_grid, to_grid_post,
                                   from_grid, from_grid_pre)
        self._check_input(x, mode)
        nu2, nv2 = self._target.shape
        x = x.to_global_data()
        if mode == self.TIMES:
            res = to_grid(self._uv, x, nu2, nv2, self._eps)
            res = to_grid_post(res)
        else:
            x = from_grid_pre(x)
            res = from_grid(self._uv, x, nu2, nv2, self._eps)
        return from_global_data(self._tgt(mode), res)
