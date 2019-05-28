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

from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..operators.linear_operator import LinearOperator
from ..sugar import from_global_data, makeDomain
import numpy as np


class GridderMaker(object):
    def __init__(self, dirty_domain, uvw, channel_fact, eps=2e-13):
        import nifty_gridder
        dirty_domain = makeDomain(dirty_domain)
        if (len(dirty_domain) != 1 or not isinstance(dirty_domain[0], RGSpace)
                or not len(dirty_domain.shape) == 2):
            raise ValueError("need dirty_domain with exactly one 2D RGSpace")
        if channel_fact.ndim != 1:
            raise ValueError("channel_fact must be a 1D array")
        bl = nifty_gridder.Baselines(
            uvw, channel_fact,
            np.zeros((uvw.shape[0], channel_fact.shape[0]), dtype=np.bool))
        nxdirty, nydirty = dirty_domain.shape
        gconf = nifty_gridder.GridderConfig(nxdirty, nydirty, eps, 1., 1.)
        nu = gconf.Nu()
        nv = gconf.Nv()
        idx = nifty_gridder.getIndices(bl, gconf)

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
        self._domain = makeDomain(grid_domain)
        self._target = makeDomain(dirty_domain)
        self._gconf = gconf
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = x.to_global_data()
        if mode == self.TIMES:
            res = self._gconf.grid2dirty(res)
        else:
            res = self._gconf.dirty2grid(res)
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
            x = x.to_global_data().reshape((-1, 1))
            x = self._bl.ms2vis(x, self._idx)
            res = nifty_gridder.vis2grid(self._bl, self._gconf, self._idx, x)
        else:
            res = nifty_gridder.grid2vis(self._bl, self._gconf, self._idx,
                                         x.to_global_data())
            res = self._bl.vis2ms(res, self._idx).reshape((-1,))
        return from_global_data(self._tgt(mode), res)
