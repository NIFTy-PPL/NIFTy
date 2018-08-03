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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

from .. import dobj, utilities
from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..sugar import from_global_data
from .linear_operator import LinearOperator


class RegriddingOperator(LinearOperator):
    def __init__(self, domain, target):
        super(RegriddingOperator, self).__init__()
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)

        # domain: fine domain
        # target: coarse domain
        distances_dom = tuple([domain[0].distances[0], domain[1].distances[0]])
        distances_tgt = tuple([target[0].distances[0], target[1].distances[0]])

        # index arrays
        dom_indices = np.arange(self.domain.size).reshape(self.domain.shape)
        tgt_indices = np.arange(self.target.size).reshape(self.target.shape)

        # Input for sparse matrix
        foo = self.domain.size*2**len(self.domain.shape)
        rs, cs, ws = np.zeros(foo), np.zeros(foo), np.zeros(foo)
        ind = 0

        print('Initializing...')
        # Loop through all points in fine grid (domain) and compute weights
        for xx in range(domain.shape[0]):
            for yy in range(domain.shape[1]):
                # Find neighbours
                xx_in_tgt = xx*distances_dom[0]/distances_tgt[0]
                yy_in_tgt = yy*distances_dom[1]/distances_tgt[1]
                p = np.array([xx, yy])
                p_in_tgt = np.array([xx_in_tgt, yy_in_tgt])
                xx_neigh = int(xx*distances_dom[0]/distances_tgt[0])
                yy_neigh = int(yy*distances_dom[1]/distances_tgt[1])
                neighbours = [
                    np.array([xx_neigh, yy_neigh]),
                    np.array([xx_neigh+1, yy_neigh]),
                    np.array([xx_neigh, yy_neigh+1]),
                    np.array([xx_neigh+1, yy_neigh+1])
                ]
                for n in neighbours:
                    ws[ind] = np.prod(1-np.abs(n-p_in_tgt))
                    if any(n == self.target.shape):
                        rs[ind], cs[ind] = -1, -1
                    else:
                        rs[ind] = tgt_indices[tuple(n)]
                        cs[ind] = dom_indices[tuple(p)]
                    ind += 1
            print('{}%'.format(np.round(xx/domain.shape[0]*100, 1)))

        # FIXME?
        mask = np.logical_and(rs != -1, ws != 0)
        rs, cs, ws = rs[mask], cs[mask], ws[mask]

        smat = csr_matrix(
            (ws, (rs, cs)), shape=(self.target.size, self.domain.size))
        self._smat = aslinearoperator(smat)

    def apply(self, x, mode):
        self._check_input(x, mode)
        inp = x.to_global_data()
        if mode == self.TIMES:
            res = self._smat.matvec(inp.reshape(-1))
        else:
            res = self._smat.rmatvec(inp.reshape(-1))
        res *= self.target.size/self.domain.size
        tgt = self._tgt(mode)
        return Field.from_global_data(tgt, res.reshape(tgt.shape))

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
