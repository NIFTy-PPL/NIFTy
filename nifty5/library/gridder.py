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
    def __init__(self, dirty_domain, uv, eps=2e-13):
        import nifty_gridder
        dirty_domain = makeDomain(dirty_domain)
        if (len(dirty_domain) != 1 or
                not isinstance(dirty_domain[0], RGSpace) or
                not len(dirty_domain.shape) == 2):
            raise ValueError("need dirty_domain with exactly one 2D RGSpace")
        bl = nifty_gridder.Baselines(uv, np.array([1.]));
        nxdirty, nydirty = dirty_domain.shape
        gconf = nifty_gridder.GridderConfig(nxdirty, nydirty, eps, 1., 1.)
        nu = gconf.Nu()
        nv = gconf.Nv()
        idx = bl.getIndices()
        idx = gconf.reorderIndices(idx, bl)

        grid_domain = RGSpace([nu, nv], distances=[1, 1], harmonic=False)

        self._rest = _RestOperator(dirty_domain, grid_domain, gconf)
        self._gridder = RadioGridder(grid_domain, bl, gconf, idx)

    def getGridder(self):
        return self._gridder

    def getRest(self):
        return self._rest

    def getFull(self):
        return self.getRest() @ self._gridder


class _RestOperator(LinearOperator):
    def __init__(self, dirty_domain, grid_domain, gconf):
        import nifty_gridder
        self._domain = makeDomain(grid_domain)
        self._target = makeDomain(dirty_domain)
        self._gconf = gconf
        fu = gconf.U_corrections()
        fv = gconf.V_corrections()
        nu, nv = dirty_domain.shape
        # compute deconvolution operator
        rng = np.arange(nu)
        k = np.minimum(rng, nu-rng)
        self._deconv_u = np.roll(fu[k], -nu//2).reshape((-1, 1))
        rng = np.arange(nv)
        k = np.minimum(rng, nv-rng)
        self._deconv_v = np.roll(fv[k], -nv//2).reshape((1, -1))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        import nifty_gridder
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
    def __init__(self, grid_domain, bl, gconf, idx):
        self._domain = DomainTuple.make(UnstructuredDomain((idx.shape[0],)))
        self._target = DomainTuple.make(grid_domain)
        self._bl = bl
        self._gconf = gconf
        self._idx = idx
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        import nifty_gridder
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = nifty_gridder.ms2grid(
                self._bl, self._gconf, self._idx, x.to_global_data().reshape((-1,1)))
        else:
            res = nifty_gridder.grid2ms(
                self._bl, self._gconf, self._idx, x.to_global_data())
        return from_global_data(self._tgt(mode), res)
