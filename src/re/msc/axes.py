# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
import healpy as hp
import jax.numpy as jnp
from jax import vmap
from .utils import get_all_kneighbours


class RegularAxis:
    def __init__(self, base, size, binsize, kernel_size, fine_axis,
                 interpolation_method_in, interpolation_method_out):
        """An axis describing a regularly spaced, one-dimensional and periodic
        space on one level in a sparse charted multigrid.

        Parameters:
        -----------
        base: int
            Refinment base along this axis from this level to the next one.
        size: int
            Total number of pixels that can possibly exist along this axis.
        binsize: float
            Width of one bin.
        kernel_size: int
            The extend in number of pixels of a convolution kernel along this
            axis. Must be odd.
        is_linear: bool #TODO kernel_interpolation_method (str)
            Whether a convolution kernel is represented via its value and first
            integral for each bin. If True, the kernel is evaluated such that it
            matches the values at the boundaries of each bin, otherwise it is
            set to match the value in the center of the bin.
        fine_axis: RegularAxis or None
            The axis representing this dimension of the space on the next
            refinment level. If this axis has no next level, `fine_axis` must be
            `None`.
        """
        self._base = int(base)
        self._size = int(size)
        self._binsize = float(binsize)
        self._kernel_size = int(kernel_size)
        if self._kernel_size%2 == 0:
            raise ValueError("Kernel size must be odd!")
        self._imethod = str(interpolation_method_in)
        self._omethod = str(interpolation_method_out)
        methods = ['nearest', 'linear']
        if self._imethod not in methods:
            msg = f'Unknown input interpolation method: {self._imethod}'
            raise NotImplementedError(msg)
        if self._omethod not in methods:
            msg = f'Unknown input interpolation method: {self._omethod}'
            raise NotImplementedError(msg)
        self._fine_axis = fine_axis
        if self._fine_axis is not None:
            if not isinstance(fine_axis, RegularAxis):
                raise ValueError("Next refinment axis is of incorrect type!")
            if fine_axis.size != self.size*self.base:
                raise ValueError("Next refinment axis is of incorrect size!")
            if fine_axis.binsize != self.binsize / self.base:
                raise ValueError("Next refinment axis has incorrect binsize!")
            #FIXME test too small for linear case
            fine_ker_part = (fine_axis.kernel_size - 1) // 2 // self.base
            if fine_ker_part > (self.kernel_size - 1) // 2:
                raise ValueError("Fine kernel too big for this axis!")
            if ((fine_axis.method_in == 'linear') and
                (self.method_in != 'linear')):
                msg = "Fine kernel cannot be second order if this axis is not"
                raise ValueError(msg)

    @property
    def base(self):
        """Refinment base of this axis"""
        return self._base

    @property
    def kernel_size(self):
        """The extend in number of pixels of a convolution kernel along this
           axis."""
        return self._kernel_size

    @property
    def binsize(self):
        """Width of one bin."""
        return self._binsize

    @property
    def size(self):
        """Total number of pixels."""
        return self._size

    @property
    def in_size(self):
        if self.method_in == 'nearest':
            return 1
        if self.method_in == 'linear':
            return 2
        raise ValueError

    @property
    def method_in(self):
        return self._imethod

    @property
    def method_out(self):
        return self._omethod

    @property
    def fine_axis(self):
        """Axis of the dimension this axis belongs to on the next refinment
        level."""
        return self._fine_axis

    @property
    def axdim(self):
        """Number of spatial dimensions this axis represents"""
        return 1

    @property
    def regular(self):
        return True

    def copy(self):
        """Creates a copy of this axis."""
        fine = None if self._fine_axis is None else self._fine_axis.copy()
        return RegularAxis(self._base, self._size, self._binsize,
                           self._kernel_size, fine, self._imethod,
                           self._omethod)

    def get_fine_indices(self, index):
        """Indices on the next level corresponding to the bins of `indices`.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Indices of the bins for which the fine binindices are requested
        Returns:
        --------
        numpy.ndarray of int
            Fine indices on the next level for the requested bins. Is of shape
            `indices.shape + (self.base,)`
        """
        fine = np.arange(self.base, dtype=index.dtype)
        return np.add.outer(self.base*index, fine, dtype=index.dtype)

    def get_binid_of(self, fine_index):
        """The the binindex on this level which contains the bin of `fine_index`
        for an index of the next level.

        Parameters:
        -----------
        fine_index: int of numpy.ndarray of int
            Index on the next level.
        Returns:
        --------
        int or numpy.ndarray of int
            Binindex on this level along this axis
        """
        return fine_index // self.base

    def get_kernel_window_ids(self, index, ks = None):
        """The indices corresponding to the kernel window centered around each
        entry of `indices`.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index on the this level.
        Returns:
        --------
        numpy.ndarray of int
            Binindices on this level of the kernel window centered on `index`.
        """
        if ks is None:
            ks = self.kernel_size
        window = np.arange(ks, dtype=index.dtype) - ks // 2
        return np.add.outer(index, window, dtype=index.dtype)%self.size

    def get_inter_window_ids(self, index):
        """The indices corresponding to the kernel and interpolation window
        centered around each entry of `indices`.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index on the this level.
        Returns:
        --------
        numpy.ndarray of int
            Binindices on this level of the kernel window centered on `index`.
        """
        ks = self.kernel_size if self.kernel_size != 1 else self.kernel_size + 2
        return self.get_kernel_window_ids(index, ks=ks)

    def kernel_to_batchkernel(self, index):
        ks = self.fine_axis.kernel_size
        wsize = self.base + 2*(ks//2)
        trafo = np.zeros((self.base, wsize, self.base, ks), dtype=int)
        for i in range(self.base):
            trafo[i, i:ks+i, i] = np.eye(ks, dtype=int)
        return trafo[np.newaxis, ...]

    def get_batch_kernel_window(self, index):
        ks = self.fine_axis.kernel_size
        w = np.arange(-(ks//2), self.base + ks//2, dtype=index.dtype)
        window = np.add.outer(self.base*index, w, dtype=index.dtype)
        return window%self.fine_axis.size

    def flagged_kernel_window(self, index):
        """Potential flagging of entries in the kernel window.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index on the this level.
        Returns:
        --------
        numpy.ndarray of bool
            Whether the entries of the kernel window centered on `index` are
            flagged or not.

        Notes:
        ------
            For a direct instance of `RegularAxis` no entries of the kernel are
            flagged. For axes that inherit from `RegularAxis` this is not
            always the case. (See `HPAxis` for such an example)
        """
        return np.zeros_like(self.get_kernel_window_ids(index), dtype=bool)

    def _bincenter(self, index):
        """Coordinate of the bincenter corresponding to `index`.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index on this level.
        Returns:
        --------
        numpy.ndarray of float
            The corresponding coordinates. Unlike `coordinate` the extra axis of
            length 1 is not prepended.
        """
        return index*self.binsize + self.binsize/2

    def coordinate(self, index):
        """Coordinate of the bincenter corresponding to `index`.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index on this level.
        Returns:
        --------
        numpy.ndarray of float
            The corresponding coordinates. Although this axis is one
            dimensional, other axes in the chart may not. Therefore the
            returned array is of shape (1,...).
        """
        return self._bincenter(index)[np.newaxis, ...]

    def binbound_coords(self, index):
        """Coordinates of the edges of the bin corresponding to `index`.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index on this level.
        Returns:
        --------
        numpy.ndarray of float
            Coordinates of the bin edges.
        """
        index = index[np.newaxis, ...]
        bbs = np.arange(2, dtype=index.dtype)
        return self.binsize*np.add.outer(index, bbs, dtype=index.dtype)

    def binid_from_coord(self, coord):
        """Binindex a coordinate belongs to modulo the length of the space along
        this axis.

        Parameters:
        -----------
        coord: numpy.ndarray of float
            The coordinate along this axis. As the axis is periodic, out of
            bounds coordinates get wrapped around into the space.
        Returns:
        --------
        numpy.ndarray of int
            The bin the coordinate falls into on that axis.
        """
        loc = coord[0]
        res = (((loc - 0.5*self.binsize) / self.binsize)%self.size).astype(int)
        left = self._bincenter(res)
        d = np.abs(left-loc)
        res[d > 0.5*self.binsize] += 1
        return res%self.size

    def coarse_grain(self):
        """Builds the matrices to coarse grain values from the next fine level
        to this level along this axis.

        Returns:
        --------
        numpy.ndarray of float
            The matrix that can be used for coarse graining
        """
        if self.fine_axis is None:
            raise ValueError
        delta = self.binsize / self.base
        wgt = 0.5 * delta * (2 * np.arange(self.base) + 1 - self.base)
        one, zero = np.ones_like(wgt), np.zeros_like(wgt)
        mat = np.array([[one, zero],[wgt, one]])
        return mat[:self.in_size, :self.fine_axis.in_size]

    def refine_mat(self, fine_index):
        """Finds the bin and builds the matrix to refine the input from this
        level to the `fine_index` on the next level along this axis.

        Parameters:
        -----------
        fine_index: ndarray of int
            Indices on the next level which requires refined values.
        Returns:
        --------
        coarse: numpy.ndarray of int
            Binids of the bins on this level that contain the `fine_index`.
        mat: jax.numpy.ndarray of float
            Unique matrices that map the input on this level to the next fine 
            level along this axis.
        select: numpy.ndarray of int
            Indexing along the first axis of `mat` that maps from the unique
            array `mat` to an array of the shape `fine_index.shape[0] + 
            mat.shape[1:]`. Can be used to identify the entries in `mat` that
            correspond to one entry of `fine_index`
        Notes:
        ------
        The array `select` is only created to avoid computing and storing all
        refinment matrices of the input.
        """
        if self.fine_axis is None:
            raise ValueError
        coarse = self.get_binid_of(fine_index)
        dcoord = self._bincenter(coarse)
        dcoord -= self._bincenter(fine_index) / self.base
        dcoord, select = np.unique(dcoord, return_inverse=True)
        def get_ker(weight):
            return jnp.array([[1., -weight],[0., 1.]]) / self.base
        mat = vmap(get_ker, 0, 0)(dcoord)
        mat = mat[:, :self.fine_axis.in_size, :self.in_size]
        return coarse, mat, select

    def _get_normalized_dist(self, fine_index, index):
        """Distance between the bincenter of `fine_index` and `index`."""
        return (fine_index + 0.5) / self.base - (index + 0.5)

    def batch_interpolate(self, refine_index, kernel_index):
        #TODO add interpolation method (nearest & quadratic) to make use of full
        #     scope.
        #TODO unify with `batch_window_to_window`
        """Finds the neighbouring bins and builds the interpolation matrices to
        interpolate the output from this level to the `fine_index` along this
        axis.

        Parameters:
        -----------
        refine_index: numpy.ndarray of int
            Indices of bins that should get interpolated to the next level.
        Returns:
        --------
        coarse: numpy.ndarray of int
            The indices on this level that are used for bilinear interpolation.
            For each entry in `refine_index` three entries in `coarse` exist.
        weights: numpy.ndarray of float
            Unique interpolation weights
        select: numpy.ndarray of int
            Indexing of `weights` that yields the weights for all entries of
            `fine_index` (See `refine_mat` for further information)
        """
        if self.fine_axis is None:
            raise ValueError
        assert np.all(kernel_index == kernel_index[0])
        if self._omethod == 'nearest':
            mat = np.ones((1,2,1))
            coarse = refine_index[..., np.newaxis]
        elif self._omethod == 'linear':
            mat = np.zeros((self.base, 3))
            for b in range(self.base):
                d = np.abs(self._get_normalized_dist(b, 0))
                if b < self.base//2:
                    mat[b] = np.array([d, 1.-d, 0.])
                else:
                    mat[b] = np.array([0., 1.-d, d])
            mat = mat[np.newaxis, ...]
            coarse = np.add.outer(refine_index, 
                                np.array([-1,0,1], dtype=refine_index.dtype), 
                                dtype=refine_index.dtype)
            coarse = coarse%self.size
        else:
            msg = f'Unknown interpolation method: {self._omethod}'
            raise NotImplementedError(msg)
        return coarse, mat

    def batch_window_to_window(self):
        if self._omethod == 'nearest':
            mat = np.ones((self.base, 1, 1))
        elif self._omethod == 'linear':
            mat = np.zeros((self.base, 2, 3))
            for b in range(self.base):
                if b < self.base//2:
                    mat[b] = np.array([[1, 0, 0], [0, 1, 0]])
                else:
                    mat[b] = np.array([[0, 1, 0], [0, 0, 1]])
        else:
            msg = f'Unknown interpolation method: {self._omethod}'
            raise NotImplementedError(msg)
        return mat

    def get_batch_fine_kernel_selection(self, refine_index):
        """Returns the selection to get the subset along the integrand axis of
        the kernel that is covered by the kernel on the next refinment level.
        #TODO docstring
        """
        fine_index = self.get_fine_indices(refine_index)
        assert len(fine_index.shape) == 2
        shp = fine_index.shape
        fine_index = fine_index.flatten()

        if self._omethod == 'nearest':
            coarse = self.get_binid_of(fine_index)
            start = ((self.kernel_size+1)*self.base)//2 - self.base
            start += (fine_index - self.base * coarse)
            fine_size = self.fine_axis.kernel_size
            start -= fine_size//2
            select = np.add.outer(start, np.arange(fine_size), 
                                  dtype=fine_index.dtype)
            select = select[:, np.newaxis, ...]
            front = np.zeros_like(select)
        elif self._omethod == 'linear':
            coarse = self.get_binid_of((fine_index - self.base//2))
            start = ((self.kernel_size+1)*self.base)//2 - self.base
            start += (fine_index - self.base * coarse)
            fine_size = self.fine_axis.kernel_size
            start -= fine_size//2
            select = np.add.outer(start, 
                                  np.arange(fine_size, dtype=fine_index.dtype), 
                                  dtype=fine_index.dtype)
            front = np.multiply.outer(np.ones_like(select), 
                                    np.arange(2, dtype=fine_index.dtype),
                                    dtype=fine_index.dtype)
            front = np.moveaxis(front, 1, -1)
            select = np.stack((select, select - self.base), axis = 1)
        else:
            msg = f'Unknown interpolation method: {self._omethod}'
            raise NotImplementedError(msg)
        return (front.reshape(shp+front.shape[1:]), 
                select.reshape(shp+select.shape[1:]))

    def get_coords_and_distances(self, index):
        """Returns the coordinate of `index` and the distances to
        the locations of the kernel window surrounding `index` for this axis.

        Parameters:
        -----------
        index: numpy.ndarray
            Index of the bin the kernel is centered on
        Returns:
        --------
        tuple of len(2) of numpy.ndarray
            The coordinate corrresponding to `index` and the coordinates of all
            bins in the kernel window, centered on `index` along this axis.
        """
        locs = self._bincenter(index)
        if self._imethod == 'linear':
            window = np.arange(self.kernel_size+1) - self.kernel_size//2 - 0.5
        elif self._imethod == 'nearest':
            window = np.arange(self.kernel_size) - self.kernel_size//2
        else:
            raise NotImplementedError
        window = window*self.binsize
        window = np.outer(np.ones_like(locs), window)
        locs = locs[np.newaxis, ..., np.newaxis]
        window = window[np.newaxis, ...]
        return (locs, window)

    def refine_axis(self, base = None, kernel_size = None,
                    interpolation_method_in = None,
                    interpolation_method_out = None):
        """Creates a new instance of `RegularAxis` and sets it to be the
        `fine_axis` of this axis.

        Parameters:
        -----------
        base: int (optional)
            Refinment base of the new fine axis. If None, the refinment base of
            this axis is used.
        kernel_size: int (optional)
            Kernel size of the new fine axis. If None, the kernel size of
            this axis is used.
        is_linear: bool (optional)#TODO
            Whether the kernel is linearly interpolated or not. If None, the 
            `is_linear` of this axis is used.
        Returns:
        --------
        RegularAxis
            The new fine axis that has been set as the fine axis of this axis.
        """
        if self._fine_axis is not None:
            raise ValueError("Axis is already refined!")
        if base is None:
            base = self.base
        if kernel_size is None:
            kernel_size = self.kernel_size
        if interpolation_method_in is None:
            interpolation_method_in = self.method_in
        if interpolation_method_out is None:
            interpolation_method_out = self.method_out
        newax = RegularAxis(base, self.size*self.base, self.binsize/self.base,
                            kernel_size, None, interpolation_method_in,
                            interpolation_method_out)
        self._fine_axis = newax
        return newax

def _int_to_basis(nside, fine, coarse, missing_neighbours):
    """Bilinear interpolation in theta/phi on a healpix sphere. The
    interpolation is performed in a local basis around the missing point,"""
    r = hp.pix2vec(2*nside, fine, nest=True)
    r = np.stack((r[1],-r[0],np.zeros_like(r[0])), axis=0)[:,np.newaxis,:]
    r /= np.linalg.norm(r, axis=0)[np.newaxis,...]
    vc = hp.pix2vec(nside, coarse, nest=True)
    vc = np.stack(vc, axis=0)
    thq, phq = hp.pix2ang(2*nside, fine, nest=True)
    thq = thq[np.newaxis, np.newaxis] - 0.5 * np.pi
    phq = np.pi - phq[np.newaxis, np.newaxis]

    # Rotate vectors to move query to (pi/2, pi) and center on query
    s, c = np.sin(thq), np.cos(thq)
    vc = (c*vc + s*np.cross(r,vc,axis=0) 
            + (1-c)*r*((vc*r).sum(axis=0)[np.newaxis,...]))
    s, c = np.sin(phq), np.cos(phq)
    vc = c*vc + np.stack([-s[0]*vc[1], s[0]*vc[0],(1-c[0])*vc[2]], axis = 0)
    th = np.stack(hp.vec2ang(vc.reshape((3,-1)).T), axis=0)
    th = th.reshape((2,) + coarse.shape)
    th -= np.pi * np.array([0.5,1.])[:,np.newaxis,np.newaxis]
    th[:,-1,missing_neighbours] = 0.5*(th[:,1,missing_neighbours] + 
                                        th[:,2,missing_neighbours])

    # Construct local eigenbasis using pixel query belongs to and the two
    # left/right neighbours. Bilinear interplation is performed in this 
    # basis.
    e1 = th[:,1]-th[:,0]
    e1 /= np.linalg.norm(e1, axis=0)
    e2 = th[:,2]-th[:,0]
    e2 -= (e1*e2).sum(axis = 0)[np.newaxis,...] * e1
    e2 /= np.linalg.norm(e2, axis=0)
    e2 = e2[:,np.newaxis,:]
    e1 = e1[:,np.newaxis,:]
    return np.stack([(e1*th).sum(axis = 0),(e2*th).sum(axis = 0)], axis=0)

def _interpolation_weights(theta, missing_neighbours):
    """Interpolation weights in case of missing neighbours. If a neighbor is
    missing (N/E/S/W), triangular interpolation is performed instead."""
    M = tuple(np.stack((np.ones(theta.shape[-1]), theta[0,i], 
                theta[1,i], theta[0,i]*theta[1,i]), axis = 1) 
                for i in range(4))
    M = np.stack(M, axis=1)
    M = np.linalg.inv(M)[:,0]
    M[missing_neighbours,1] += 0.5*M[missing_neighbours,-1]
    M[missing_neighbours,2] += 0.5*M[missing_neighbours,-1]
    M[missing_neighbours,-1] = 0
    return M

class HPAxis(RegularAxis):
    def __init__(self, nside, knn_neighbours, fine_axis, 
                 interpolation_method_in, interpolation_method_out):
        """An axis describing a pixelization of a HEALPiX sphere.

        Parameters:
        -----------
        nside: int
            Nside of the sphere
        size: int
            Total number of pixels that can possibly exist along this axis.
        knn_neighbours: int
            Number of (k-)nearest neighbours that are used to define the kernel
            window on this axis.
        fine_axis: HPAxis
            The axis representing this subspace of the space on the next
            refinment level. If this axis has no next level, `fine_axis` must be
            `None`.
        Notes:
        ------
            The pixels on the HEALPiX sphere are indexed using the "NESTED"
            indexing scheme of healpy.
        """
        self._nside = int(nside)
        if not hp.isnsideok(self._nside, nest=True):
            raise ValueError("Incompatible nside")
        self._knn = int(knn_neighbours)
        if fine_axis is not None:
            if not isinstance(fine_axis, HPAxis):
                raise ValueError("Fine axis is not an instance of `HPAxis`")
            if fine_axis.nside != 2*self.nside:
                raise ValueError("Fine axis has incorrect nside")
        if interpolation_method_in != 'nearest':
            raise NotImplementedError
        sz = hp.nside2npix(self._nside)
        super().__init__(4, sz, 4*np.pi / sz, 1, fine_axis,
                         interpolation_method_in, interpolation_method_out)

    @property
    def nside(self):
        """Nside of the HEALPiX sphere"""
        return self._nside

    @property
    def kernel_size(self):
        """Size of the kernel in number of bins."""
        ksz = (1 + 2*self._knn)**2
        if (self._knn == -1) or (ksz >= hp.nside2npix(self.nside)):
            return hp.nside2npix(self.nside)
        return ksz

    @property
    def axdim(self):
        """Number of spatial dimensions this axis represents"""
        return 2

    @property
    def regular(self):
        return False

    def copy(self):
        """Creates a copy of this axis."""
        fine = None if self._fine_axis is None else self._fine_axis.copy()
        return HPAxis(self.nside, self._knn, fine)

    def _bincenter(self, index):
        # TODO?
        raise NotImplementedError

    def coordinate(self, index):
        """Coordinate of the bincenter corresponding to `indices`.

        Parameters:
        -----------
        indices: numpy.ndarray of int
            Index on this level.
        Returns:
        --------
        numpy.ndarray of float
            The corresponding coordinates. The returned array is of shape 
            (3,...).
        Notes:
        ------
        Returns the xyz coordinates of the unit vector pointing to the center
        of the bin indexed by `index`.
        """
        loc = hp.pix2vec(self.nside, index.flatten(), nest = True)
        return np.stack(loc, axis=0).reshape((3,) + index.shape)

    def binbound_coords(self, index):
        #TODO?
        raise NotImplementedError

    def binid_from_coord(self, loc):
        """Binindex a coordinate belongs to.

        Parameters:
        -----------
        coord: numpy.ndarray of float
            The xyz coordinate on the unit sphere.
        Returns:
        --------
        numpy.ndarray of int
            The binid of the pixel the coordinate falls into on the HEALPiX
            sphere.
        """
        return hp.vec2pix(self.nside, *(x for x in loc), nest=True)

    def get_kernel_window_ids(self, index, want_isbad = False, k_nn = None):
        """The indices corresponding to the kernel window centered around each
        entry of `indices`.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index on the this level.
        Returns:
        --------
        numpy.ndarray of int
            Binindices on this level of the kernel window centered on `index`.
        """
        # TODO One could improve the coverage using radial symmetries!
        if k_nn is None:
            k_nn = self._knn
        if (k_nn == -1) or (self.kernel_size >= hp.nside2npix(self.nside)):
            knn = np.arange(hp.nside2npix(self.nside), dtype=index.dtype)
            knn = np.outer(np.ones_like(index), knn).astype(index.dtype)
            if want_isbad:
                isbad = np.zeros_like(knn, dtype=bool)
                return knn, isbad
            return knn
        knn, isbad = get_all_kneighbours(self.nside, index, k_nn, True)
        knn = knn.astype(index.dtype)
        if want_isbad:
            return knn.T, isbad.T
        return knn.T

    def get_inter_window_ids(self, index):
        """The indices corresponding to the kernel and interpolation window 
        centered around each entry of `indices`.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index on the this level.
        Returns:
        --------
        numpy.ndarray of int
            Binindices on this level of the kernel window centered on `index`.
        """
        knn = self._knn if self._knn != 0 else 1
        return self.get_kernel_window_ids(index, k_nn=knn)

    def flagged_kernel_window(self, index):
        """Potential flagging of entries in the kernel window.

        Parameters:
        -----------
        index: numpy.ndarray of int
            Index on the this level.
        Returns:
        --------
        numpy.ndarray of bool
            Whether the entries of the kernel window centered on `index` are
            flagged or not.

        Notes:
        ------
            Due to the irregular base shape of HEALPiX, some pixels do not have
            the same number of pixels as others (most pixels have 8 nearest
            neighbors, some may have less). In such cases the corresponding
            kernel window still has the desired shape, with the center of the
            window serving as a dummy index for the nonexisting neighbours.
            These entries get flagged and ultimately the kernel is multiplied by
            0 for these entries. (See healpy.pixelfunc.get_all_neighbours for
            further information on missing neighbours)
        """
        return self.get_kernel_window_ids(index, True)[1]

    def _batch_kernel(self, index, want_kernel=False):
        fax = self.fine_axis
        fids = self.get_fine_indices(index)
        shp = fids.shape
        windows, bad = fax.get_kernel_window_ids(fids.flatten(),
                                                 want_isbad=True)
        windows = windows.reshape(shp+windows.shape[1:])
        bad = bad.reshape(shp+bad.shape[1:])

        ks = self.fine_axis.kernel_size
        ksz = (1 + 1 + 2*self.fine_axis._knn)**2
        if (self.fine_axis._knn == -1) or (ksz >= hp.nside2npix(self.nside)):
            ksz = hp.nside2npix(self.nside)

        ids = np.zeros(index.shape+(ksz,), dtype=index.dtype)
        if want_kernel:
            trafo = np.zeros((ids.shape[0], self.base, ksz, self.base, ks),
                             dtype=np.int8)

        good = np.prod(~bad.reshape((bad.shape[0],-1)), axis=1).astype(bool)
        if np.sum(good) > 0:
            goodw = windows[good]
            goodw = goodw.reshape((goodw.shape[0],-1))
            gs = np.argsort(goodw, axis=1)
            goodw = np.take_along_axis(goodw, gs, axis=1)
            sub = (goodw[:,1:] - goodw[:,:-1]) != 0
            u = np.concatenate((np.ones((sub.shape[0],1),dtype=bool), sub),
                               axis=1)
            assert np.all(np.sum(u, axis=1) == ksz)

            goodids = goodw[u].reshape((goodw.shape[0],-1))
            ids[good] = goodids
            if want_kernel:
                gtrafo = np.zeros((goodids.shape[0], self.base, ksz,
                                   self.base, ks), dtype=trafo.dtype)
                tsort = np.zeros((gs.shape[1], gtrafo.shape[0], gs.shape[1]),
                                 dtype=trafo.dtype)
                fill = np.arange(gs.shape[0], dtype=gs.dtype)
                for i in range(gs.shape[1]):
                    tsort[i, fill, gs[:,i]] = 1
                tsort = np.swapaxes(tsort, 0, 1)
                assert np.all(np.sum(tsort, axis=-1) == 1)
                tproj = np.zeros((gtrafo.shape[0],gs.shape[1],goodids.shape[1]),
                                 dtype=trafo.dtype)
                tm = np.cumsum(u, axis=1) - 1
                for i in range(goodids.shape[1]):
                    tproj[tm == i, i] = 1
                tproj = np.swapaxes(tproj, 1, 2)
                tr = (tproj[...,np.newaxis]*tsort[:,np.newaxis,:,:]).sum(axis=2)
                for bb in range(self.base):
                    m0 = np.zeros((self.base*ks, ks), dtype=trafo.dtype)
                    m0[bb*ks:(bb+1)*ks, :] = np.eye(ks, dtype=trafo.dtype)
                    gtrafo[:, bb, :, bb, :] = np.tensordot(tr, m0,
                                        axes=((2,),(0,))).astype(trafo.dtype)
                    assert np.all((gtrafo[:,bb,:,bb,:].sum(axis=-1) == 1) +
                                  (gtrafo[:,bb,:,bb,:].sum(axis=-1) == 0))
                trafo[good] = gtrafo

        if np.sum(~good) > 0:
            badw = windows[~good]
            badw = badw.reshape((badw.shape[0],-1))
            badids = np.zeros((badw.shape[0], ksz), dtype=index.dtype)
            if want_kernel:
                badtrafo = np.zeros((badw.shape[0],) + trafo.shape[1:],
                                    dtype=trafo.dtype)
            for ii in range(badw.shape[0]):
                u, inv = np.unique(badw[ii], return_inverse=True)
                extra = ksz-u.size
                tm = np.concatenate((u,np.ones(extra, dtype=index.dtype)*u[-1]))
                badids[ii] = tm
                if want_kernel:
                    um = np.zeros((ksz, self.base*ks), dtype=trafo.dtype)
                    for j in range(u.size):
                        um[j, inv == j] = 1
                    for bb in range(self.base):
                        m0 = np.zeros((self.base*ks, ks), dtype=trafo.dtype)
                        m0[bb*ks:(bb+1)*ks, :] = np.eye(ks, dtype=trafo.dtype)
                        badtrafo[ii, bb, :, bb, :] = np.tensordot(um, m0,
                                    axes=((1,),(0,)), ).astype(trafo.dtype)
            ids[~good] = badids
            if want_kernel:
                trafo[~good] = badtrafo

        #vpix = hp.pix2ang(self.nside, index, nest=True)
        #vpix = np.stack(vpix, axis=0)[..., np.newaxis]
        #shp = ids.shape
        #vnbr = hp.pix2ang(self.fine_axis.nside, ids.flatten(), nest=True)
        #vnbr = np.stack(vnbr, axis=0).reshape((2,) + shp)
        #dv = vnbr - vpix
        #dv = dv[1] + 1.j*dv[0]
        #s = np.argsort(dv, axis=1)
        #y = np.arange(s.shape[0], dtype=s.dtype)[...,np.newaxis]
        #ids = ids[y, s]

        if want_kernel:
            #trafo = np.moveaxis(trafo, 2, 1)
            #trafo = trafo[y, s]
            #trafo = np.moveaxis(trafo, 1, 2)

            assert np.all((trafo == 1) + (trafo == 0))
            return ids, trafo
        return ids

    def kernel_to_batchkernel(self, index):
        return self._batch_kernel(index, want_kernel=True)[1]

    def get_batch_kernel_window(self, index):
        return self._batch_kernel(index)

    def refine_mat(self, fine_index):
        """Finds the bin and builds the matrix to refine the input from this
        level to the `fine_index` on the next level along this axis.

        Parameters:
        -----------
        fine_index: ndarray of int
            Indices on the next level which requires refined values.
        Returns:
        --------
        coarse: numpy.ndarray of int
            Binids of the bins on this level that contain the `fine_index`.
        mat: jax.numpy.ndarray of float
            Unique matrices that map the input on this level to the next fine
            level along this axis.
        select: numpy.ndarray of int
            Indexing along the first axis of `mat` that maps from the unique
            array `mat` to an array of the shape `fine_index.shape[0] +
            mat.shape[1:]`. Can be used to identify the entries in `mat` that
            correspond to one entry of `fine_index`
        Notes:
        ------
        The array `select` is only created to avoid computing and storing all
        refinment matrices of the input.
        """
        if self.fine_axis is None:
            raise ValueError
        coarse = self.get_binid_of(fine_index)
        mat = np.array([[1./self.base,],])
        mat = mat[np.newaxis, ...]
        select = np.zeros_like(fine_index)
        return coarse, mat, select

    def _get_interpolation_window(self, index, want_bad = False):
        window = hp.get_all_neighbours(self.nside, index, nest=True).T
        window = window.astype(index.dtype)
        window = np.concatenate((index[:, np.newaxis], window), axis=1)
        bad = window == -1
        x, y = np.where(bad)
        window[x, y] = index[x]
        if want_bad:
            return window, bad
        return window

    def _select_pairs(self):
        return {0:np.array([0,1,7,8]), 1:np.array([0,7,5,6]),
                2:np.array([0,3,1,2]), 3:np.array([0,5,3,4])}

    def batch_window_to_window(self):
        if self._omethod == 'nearest':
            mat = np.ones((4,1,1))
        elif self._omethod == 'linear':
            mat = np.zeros((4, 4, 9), dtype=int)
            pairs = self._select_pairs()
            for i in range(4):
                m = np.zeros((4,9), dtype=int)
                m[np.arange(4, dtype=int), pairs[i]] = 1
                mat[i] = m
        else:
            msg = f'Unknown interpolation method: {self._omethod}'
            raise NotImplementedError(msg)
        return mat

    def batch_interpolate(self, refine_index, kernel_index,
                          want_coarse = False):
        #TODO add interpolation method (nearest & quadratic) to make use of full
        #     scope.
        """Finds the neighbouring bins and builds the interpolation matrices to
        interpolate the output from this level to the `fine_index` along this
        axis.

        Parameters:
        -----------
        refine_index: numpy.ndarray of int
            Indices of bins that should get interpolated to the next level.
        Returns:
        --------
        coarse: numpy.ndarray of int
            The indices on this level that are used for bilinear interpolation.
            For each entry in `refine_index` three entries in `coarse` exist.
        weights: numpy.ndarray of float
            Unique interpolation weights
        select: numpy.ndarray of int
            Indexing of `weights` that yields the weights for all entries of
            `fine_index` (See `refine_mat` for further information)
        """
        if self.fine_axis is None:
            raise ValueError
        if self._omethod == 'nearest':
            window = refine_index[..., np.newaxis]
            ker = np.ones(kernel_index.shape+(4,1))
            all_coarse = np.stack((kernel_index,)*4, axis=-1)
            all_coarse = all_coarse.flatten()[..., np.newaxis]
        elif self._omethod == 'linear':
            window = self._get_interpolation_window(refine_index)
            kid = np.multiply.outer(kernel_index,
                                    np.ones((4,), dtype=kernel_index.dtype),
                                    dtype=kernel_index.dtype).flatten()
            coarse, bad = self._get_interpolation_window(kid, True)
            fine_index = self.get_fine_indices(kernel_index).flatten()
            dm = fine_index - 4*kid

            select_pairs = self._select_pairs()
            all_coarse = np.zeros(fine_index.shape + (4,), dtype=coarse.dtype)
            all_bad = np.zeros(fine_index.shape + (4,), dtype=bad.dtype)
            for i in range(4):
                cond = dm == i
                all_coarse[cond] += coarse[cond][:,select_pairs[i]]
                all_bad[cond] = bad[cond][:,select_pairs[i]]
            assert np.all(np.sum(all_bad[:,:-1], axis=1) == 0)

            angles = _int_to_basis(self.nside, fine_index, all_coarse.T,
                                all_bad[:,-1])
            weights = _interpolation_weights(angles, all_bad[:,-1])
            weights = weights.reshape(kernel_index.shape + (4,4))
            ker = np.zeros(weights.shape[:-1] + (9,))
            mat = self.batch_window_to_window()
            for i in range(4):
                ker[:,i] = weights[:,i] @ mat[i]
        else:
            msg = f'Unknown interpolation method: {self._omethod}'
            raise NotImplementedError(msg)
        if want_coarse:
            return all_coarse
        return window, ker

    def get_coords_and_distances(self, index):
        """Returns the coordinate of `index` and the distances to
        the locations of the kernel window surrounding `index` for this axis.

        Parameters:
        -----------
        index: numpy.ndarray
            Index of the bin the kernel is centered on
        Returns:
        --------
        tuple of len(2) of numpy.ndarray
            The coordinate corrresponding to `index` and the coordinates of all
            bins in the kernel window, centered on `index` along this axis.
        """
        locp = self.get_kernel_window_ids(index)
        shp = index.shape
        shpp = locp.shape
        loc = hp.pix2vec(self.nside, index.flatten(), nest = True)
        loc = tuple(ll.reshape(shp) for ll in loc)
        loc = np.stack(loc, axis = 0)
        shpp = locp.shape
        locp = hp.pix2vec(self.nside, locp.flatten(), nest = True)
        locp = tuple(ll.reshape(shpp) for ll in locp)
        locp = np.stack(locp, axis = 0)
        loc = loc[..., np.newaxis]
        dloc = locp - loc
        return (loc, dloc)

    def get_batch_fine_kernel_selection(self, refine_index):
        """Returns the selection to get the subset along the integrand axis of
        the kernel that is covered by the kernel on the next refinment level.
        #TODO docstring
        """
        coarse_ids = self.batch_interpolate(refine_index, refine_index,
                                            want_coarse=True)
        fine_index = self.get_fine_indices(refine_index)
        assert len(fine_index.shape) == 2
        inshp = fine_index.shape
        fine_index = fine_index.flatten()

        coarse_ids = coarse_ids.T
        fine_kernel, fine_bad = self.fine_axis.get_kernel_window_ids(fine_index,
                                                            want_isbad = True)
        shp = coarse_ids.shape
        coarse_kernels, coarse_bad = self.get_kernel_window_ids(
                                                coarse_ids.flatten(),
                                                want_isbad = True)
        coarse_kernels = coarse_kernels.reshape(shp + (-1,))
        coarse_bad = coarse_bad.reshape(shp + (-1,))
        coarse_kernels = self.get_fine_indices(coarse_kernels)
        coarse_bad = np.multiply.outer(coarse_bad,
                                       np.ones(self.base, dtype=bool))
        coarse_kernels = coarse_kernels.reshape(coarse_kernels.shape[:2]+(-1,))
        coarse_bad = coarse_bad.reshape(coarse_bad.shape[:2]+(-1,))
        res = np.zeros(coarse_kernels.shape[:2] + (fine_kernel.shape[-1],),
                       dtype=coarse_kernels.dtype)
        cshp = coarse_kernels.shape[-1]
        fine_good = fine_bad.sum(axis = -1) == 0
        for i in range(res.shape[0]):
            # Vectorize in case there are no bad pixels in kernels
            coarse_good = coarse_bad[i].sum(axis = -1) == 0
            all_good = coarse_good * fine_good
            coarse = coarse_kernels[i][all_good]
            fine = fine_kernel[all_good]
            if coarse.shape[0] > 0:
                allid = np.concatenate((coarse, fine), axis = -1)
                sorting = np.argsort(allid, axis = -1)
                x = np.arange(allid.shape[0])[..., np.newaxis]
                allsort = allid[x, sorting]
                double = np.zeros_like(allid, dtype = bool)
                cond = (allsort[..., :-1] == allsort[..., 1:])
                double[x, sorting[...,:-1]] = cond
                double[x, sorting[..., 1:]] += cond
                sel = np.outer(np.ones_like(coarse[:,0]), np.arange(cshp))
                res[i][all_good] = sel[double[..., :cshp]].reshape(fine.shape)
            # Explicitly loop over all kernel pairs that contain bad pixels
            coarse = coarse_kernels[i][~all_good]
            fine = fine_kernel[~all_good]
            badres = np.zeros(fine.shape)
            for j in range(fine.shape[0]):
                rr = list(np.where(coarse[j] == bb)[0][0] for bb in fine[j])
                badres[j] = np.array(rr)
            res[i][~all_good] = badres
        front = np.multiply.outer(
            np.arange(res.shape[0], dtype=fine_index.dtype),
            np.ones_like(fine_kernel), dtype=fine_index.dtype)
        res = np.moveaxis(res, 1, 0)
        front = np.moveaxis(front, 1, 0)
        return (front.reshape(inshp+front.shape[1:]),
                res.reshape(inshp+res.shape[1:]))

    def refine_axis(self, knn_neighbours = None, interpolation_method_in = None,
                    interpolation_method_out = None):
        """Creates a new instance of `RegularAxis` and sets it to be the
        `fine_axis` of this axis.

        Parameters:
        -----------
        knn_neighbours: int (optional)
            Number of (k-)nearest neighbours fo the kernel window for the new
            fine axis. If None, `knn_neghbours` of this axis is used.
        Returns:
        --------
        HPAxis
            The new fine axis that has been set as the fine axis of this axis.
        """
        if self._fine_axis is not None:
            raise ValueError("Axis is already refined!")
        if knn_neighbours is None:
            knn_neighbours = self._knn
        if interpolation_method_in is None:
            interpolation_method_in = self.method_in
        if interpolation_method_out is None:
            interpolation_method_out = self.method_out
        newax = HPAxis(2*self.nside, knn_neighbours, None,
                       interpolation_method_in, interpolation_method_out)
        self._fine_axis = newax
        return newax