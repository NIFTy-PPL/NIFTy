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

from ..compat import *
from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class RegriddingOperator(LinearOperator):
    def __init__(self, domain, target):
        super(RegriddingOperator, self).__init__()
        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)
        if any(np.array(self.domain.shape) < np.array(self.target.shape)):
            print('Warning: The regridding operator is not intended to be used for upsampling.')

        # domain: fine domain
        # target: coarse domain
        distances_dom = sum([list(dom.distances) for dom in self.domain], [])
        distances_tgt = sum([list(dom.distances) for dom in self.target], [])
        dim = len(distances_tgt)

        # index arrays
        dom_indices = np.arange(self.domain.size).reshape(self.domain.shape)
        tgt_indices = np.arange(self.target.size).reshape(self.target.shape)

        # Input for sparse matrix
        foo = (self.domain.size, 2**len(self.domain.shape))
        rs, cs, ws = np.zeros(foo), np.zeros(foo), np.zeros(foo)

        print('Initializing...')
        # Calculate weights
        fac = np.array(distances_dom)/distances_tgt
        find_neighbours = np.array(
            np.meshgrid(*[[0, 1] for _ in range(dim)])).T.reshape(-1, dim)
        for ind, global_index in np.ndenumerate(dom_indices):
            p_in_tgt = np.outer(ind*fac, np.ones(2**dim)).T
            neighbours = p_in_tgt.astype(int)+find_neighbours
            ws[global_index] = np.prod(1-np.abs(neighbours-p_in_tgt), axis=1)
            cs[global_index] = dom_indices[tuple(
                np.array(ind) % self.domain.shape)]
            rs[global_index] = [
                tgt_indices[tuple(n % self.target.shape)] for n in neighbours
            ]
            if global_index % 10000 == 9999:
                print('{}%'.format(np.round(global_index/dom_indices.size*100), 3))
        print('Done')

        # Throw away zero weights and flatten at the same time
        mask = ws != 0
        rs, cs, ws = rs[mask], cs[mask], ws[mask]
        if np.sum(ws) != self.domain.size:
            raise RuntimeError

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
