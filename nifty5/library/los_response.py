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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.special import erfc

from .. import dobj
from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..operators.linear_operator import LinearOperator


def _gaussian_error_function(x):
    return 0.5*erfc(x*np.sqrt(2.))

def _comp_traverse(start, end, shp, dist, lo, mid, hi, erf):
    ndim = start.shape[0]
    nlos = start.shape[1]
    inc = np.full(len(shp), 1, dtype=np.int64)
    for i in range(-2, -len(shp)-1, -1):
        inc[i] = inc[i+1]*shp[i+1]

    pmax = np.array(shp)

    out = [None]*nlos
    for i in range(nlos):
        dir = end[:, i]-start[:, i]
        dirx = np.where(dir == 0., 1e-12, dir)
        d0 = np.where(dir == 0., ((start[:, i] > 0)-0.5)*1e12,
                      -start[:, i]/dirx)
        d1 = np.where(dir == 0., ((start[:, i] < pmax)-0.5)*-1e12,
                      (pmax-start[:, i])/dirx)
        (dmin, dmax) = (np.minimum(d0, d1), np.maximum(d0, d1))
        dmin = dmin.max()
        dmax = dmax.min()
        dmin = np.maximum(0., dmin)
        dmax = np.minimum(1., dmax)
        dmax = np.maximum(dmin, dmax)
        # hack: move away from potential grid crossings
        dmin += 1e-7
        dmax -= 1e-7
        if dmin >= dmax:  # no intersection
            out[i] = (np.full(0, 0, dtype=np.int64), np.full(0, 0.))
            continue
        # determine coordinates of first cell crossing
        c_first = np.ceil(start[:, i]+dir*dmin)
        c_first = np.where(dir > 0., c_first, c_first-1.)
        c_first = (c_first-start[:, i])/dirx
        pos1 = np.asarray((start[:, i]+dmin*dir), dtype=np.int)
        pos1 = np.sum(pos1*inc)
        cdist = np.empty(0, dtype=np.float64)
        add = np.empty(0, dtype=np.int)
        for j in range(ndim):
            if dir[j] != 0:
                step = inc[j] if dir[j] > 0 else -inc[j]
                tmp = np.arange(start=c_first[j], stop=dmax,
                                step=abs(1./dir[j]))
                cdist = np.append(cdist, tmp)
                add = np.append(add, np.full(len(tmp), step, dtype=np.int64))
        idx = np.argsort(cdist)
        cdist = cdist[idx]
        add = add[idx]
        cdist = np.append(np.full(1, dmin), cdist)
        cdist = np.append(cdist, np.full(1, dmax))
        corfac = np.linalg.norm(dir*dist)
        cdist *= corfac
        wgt = np.diff(cdist)
        mdist = 0.5*(cdist[:-1]+cdist[1:])
        wgt = apply_erf(wgt, mdist, lo[i], mid[i], hi[i], erf)
        add = np.append(pos1, add)
        add = np.cumsum(add)
        out[i] = (add, wgt)
    return out


def apply_erf(wgt, dist, lo, mid, hi, erf):
    wgt = wgt.copy()
    mask = dist > hi
    wgt[mask] = 0.
    mask = (dist > lo) & (dist <= hi)
    sig = (1/lo - 1/mid)/3
    wgt[mask] *= erf((-1/dist[mask]+1/mid)/sig/2)
    #wgt[mask] *= erf((dist[mask]-mid)/(hi-mid))
    #mask = (dist <= mid) & (dist > lo)
    #wgt[mask] *= erf((dist[mask]-mid)/(mid-lo))
    return wgt


class LOSResponse(LinearOperator):
    """Line-of-sight response operator

    This operator transforms from a single RGSpace to an unstructured domain
    with as many entries as there were lines of sight passed to the
    constructor. Adjoint application is also provided.

    Parameters
    ----------
    domain : RGSpace or DomainTuple
        The operator's input domain. This must be a single RGSpace.
    starts, ends : numpy.ndarray(float) with two dimensions
        Arrays containing the start and end points of the individual lines
        of sight. The first dimension must have as many entries as `domain`
        has dimensions. The second dimensions must be identical for both arrays
        and indicated the total number of lines of sight.
    sigmas_low, sigmas_up : numpy.ndarray(float) (optional)
        sigmas_low is 1/(parallax+3*parallax_error), where the parallax
        error is assumed to be Gaussian distributed.
        sigmas_up is the distance at which the weight is truncated.
        Should be at least 1/(parllax-3*parallax_error), but could be higher.
        If unsure, leave blank.

    Notes
    -----
    `starts, `ends`, `sigmas_low`, and `sigmas_up` have to be identical on
    every calling MPI task (i.e. the full LOS information has to be provided on
    every task).
    """

    def __init__(self, domain, starts, ends, sigmas_low=None, sigmas_up=None):
        self._domain = DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        if ((not isinstance(self.domain[0], RGSpace)) or
                (len(self._domain) != 1)):
            raise TypeError("The domain must be exactly one RGSpace instance.")

        ndim = len(self.domain[0].shape)
        starts = np.array(starts)
        nlos = starts.shape[1]
        ends = np.array(ends)
        if sigmas_low is None:
            sigmas_low = np.zeros(nlos, dtype=np.float32)
        if sigmas_up is None:
            sigmas_up = np.zeros(nlos, dtype=np.float32)
        sigmas_low = np.array(sigmas_low)
        sigmas_up = np.array(sigmas_up)
        if starts.shape[0] != ndim:
            raise TypeError("dimension mismatch")
        if nlos != sigmas_low.shape[0]:
            raise TypeError("dimension mismatch")
        if starts.shape != ends.shape:
            raise TypeError("dimension mismatch")
        if sigmas_low.shape != sigmas_up.shape:
            raise TypeError("dimension mismatch")

        self._local_shape = dobj.local_shape(self.domain[0].shape)
        local_zero_point = (np.array(
            dobj.ibegin_from_shape(self.domain[0].shape)) *
            np.array(self.domain[0].distances))

        diffs = ends-starts
        difflen = np.linalg.norm(diffs, axis=0)
        diffs /= difflen
        real_ends = ends + sigmas_up*diffs
        lzp = local_zero_point.reshape((-1, 1))
        dist = np.array(self.domain[0].distances).reshape((-1, 1))
        localized_pixel_starts = (starts-lzp)/dist + 0.5
        localized_pixel_ends = (real_ends-lzp)/dist + 0.5

        # get the shape of the local data slice
        w_i = _comp_traverse(localized_pixel_starts,
                             localized_pixel_ends,
                             self._local_shape,
                             np.array(self.domain[0].distances),
                             difflen-sigmas_low, difflen, difflen+sigmas_up,
                             _gaussian_error_function)

        boxsz = 16
        nlos = len(w_i)
        npix = np.prod(self._local_shape)
        ntot = 0
        for i in w_i:
            ntot += len(i[1])
        pri = np.empty(ntot, dtype=np.float64)
        ilos = np.empty(ntot, dtype=np.int32)
        iarr = np.empty(ntot, dtype=np.int32)
        xwgt = np.empty(ntot, dtype=np.float32)
        ofs = 0
        cnt = 0
        for i in w_i:
            nval = len(i[1])
            ilos[ofs:ofs+nval] = cnt
            iarr[ofs:ofs+nval] = i[0]
            xwgt[ofs:ofs+nval] = i[1]
            fullidx = np.unravel_index(i[0], self._local_shape)
            tmp = np.zeros(nval, dtype=np.float64)
            fct = 1.
            for j in range(ndim):
                tmp += (fullidx[j]//boxsz)*fct
                fct *= self._local_shape[j]
            tmp += cnt/float(nlos)
            tmp += iarr[ofs:ofs+nval]/(float(nlos)*float(npix))
            pri[ofs:ofs+nval] = tmp
            ofs += nval
            cnt += 1
        xtmp = np.argsort(pri)
        ilos = ilos[xtmp]
        iarr = iarr[xtmp]
        xwgt = xwgt[xtmp]
        self._smat = aslinearoperator(
            coo_matrix((xwgt, (ilos, iarr)),
                       shape=(nlos, np.prod(self._local_shape))))

        self._target = DomainTuple.make(UnstructuredDomain(nlos))

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            result_arr = self._smat.matvec(x.local_data.reshape(-1))
            return Field.from_global_data(self._target, result_arr,
                                          sum_up=True)
        local_input_data = x.to_global_data().reshape(-1)
        res = self._smat.rmatvec(local_input_data).reshape(self._local_shape)
        return Field.from_local_data(self._domain, res)
