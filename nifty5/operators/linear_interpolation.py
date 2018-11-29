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

from numpy import (abs, arange, array, int64, mgrid, prod, ravel,
                   ravel_multi_index, zeros)
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import aslinearoperator

from ..compat import *
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..sugar import makeDomain
from .linear_operator import LinearOperator


class LinearInterpolator(LinearOperator):
    def __init__(self, domain, positions):
        """

        :param domain:
            RGSpace
        :param positions:
            positions at which to interpolate, shape (dim, ndata)
        """
        self._domain = makeDomain(domain)
        N_points = positions.shape[1]
        self._target = makeDomain(UnstructuredDomain(N_points))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._build_mat(positions, N_points)

    def _build_mat(self, positions, N_points):
        ndim = positions.shape[0]
        mg = mgrid[(slice(0, 2),)*ndim]
        mg = array(list(map(ravel, mg)))
        dist = []
        for dom in self.domain:
            if isinstance(dom, UnstructuredDomain):
                dist.append([1]*len(dom.shape))
            elif isinstance(dom, RGSpace):
                dist.append(list(dom.distances))
            else:
                raise TypeError
        dist = array(dist).flatten().reshape((-1, 1))
        pos = positions/dist
        excess = pos-pos.astype(int64)
        pos = pos.astype(int64)
        data = zeros((len(mg[0]), N_points))
        ii = zeros((len(mg[0]), N_points), dtype=int64)
        jj = zeros((len(mg[0]), N_points), dtype=int64)
        for i in range(len(mg[0])):
            factor = prod(abs(1-mg[:, i].reshape((-1, 1))-excess), axis=0)
            data[i, :] = factor
            fromi = pos+mg[:, i].reshape((-1, 1))
            ii[i, :] = arange(N_points)
            jj[i, :] = ravel_multi_index(fromi, self.domain.shape)
        self._mat = coo_matrix((data.reshape(-1),
                               (ii.reshape(-1), jj.reshape(-1))),
                               (N_points, prod(self.domain.shape)))
        self._mat = aslinearoperator(self._mat)

    def apply(self, x, mode):
        self._check_input(x, mode)
        x_val = x.to_global_data()
        if mode == self.TIMES:
            res = self._mat.matvec(x_val.reshape((-1,)))
            return Field.from_global_data(self.target, res)
        res = self._mat.rmatvec(x_val).reshape(self.domain.shape)
        return Field.from_global_data(self.domain, res)


# import numpy as np
# from ..domains.rg_space import RGSpace
# import itertools
#
#
# class LinearInterpolator(LinearOperator):
#     def __init__(self, domain, positions):
#         """
#         :param domain:
#             RGSpace
#         :param target:
#             UnstructuredDomain, shape (ndata,)
#         :param positions:
#             positions at which to interpolate
#             Field with UnstructuredDomain, shape (dim, ndata)
#         """
#         if not isinstance(domain, RGSpace):
#             raise TypeError("RGSpace needed")
#         if np.any(domain.shape < 2):
#             raise ValueError("RGSpace shape too small")
#         if positions.ndim != 2:
#             raise ValueError("positions must be a 2D array")
#         ndim = len(domain.shape)
#         if positions.shape[0] != ndim:
#             raise ValueError("shape mismatch")
#         self._domain = makeDomain(domain)
#         N_points = positions.shape[1]
#         dist = np.array(domain.distances).reshape((ndim, -1))
#         self._pos = positions/dist
#         shp = np.array(domain.shape, dtype=int).reshape((ndim, -1))
#         self._idx = np.maximum(0, np.minimum(shp-2, self._pos.astype(int)))
#         self._pos -= self._idx
#         tmp = tuple([0, 1] for i in range(ndim))
#         self._corners = np.array(list(itertools.product(*tmp)))
#         self._target = makeDomain(UnstructuredDomain(N_points))
#         self._capability = self.TIMES | self.ADJOINT_TIMES
#
#     def apply(self, x, mode):
#         self._check_input(x, mode)
#         x = x.to_global_data()
#         ndim = len(self._domain.shape)
#
#         res = np.zeros(self._tgt(mode).shape, dtype=x.dtype)
#         for corner in self._corners:
#             corner = corner.reshape(ndim, -1)
#             idx = self._idx+corner
#             idx2 = tuple(idx[t, :] for t in range(idx.shape[0]))
#             wgt = np.prod(self._pos*corner+(1-self._pos)*(1-corner), axis=0)
#             if mode == self.TIMES:
#                 res += wgt*x[idx2]
#             else:
#                 np.add.at(res, idx2, wgt*x)
#         return Field.from_global_data(self._tgt(mode), res)
