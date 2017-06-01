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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import numpy as np
from d2o import distributed_data_object

class PowerIndices(object):
    """Computes helpful quantities to deal with power spectra.

    Given the shape and the density of a underlying rectangular grid this
    class provides the user
    with the pindex, kindex and rho. The indices are binned
    according to the supplied parameter scheme.

    Parameters
    ----------
    domain : NIFTy harmonic space
        The space for which the power indices get computed
    distribution_strategy : str
        The distribution_strategy that will be used for the k_array and pindex
        distributed_data_object.
    """
    def __init__(self, domain, distribution_strategy):
        self.domain = domain
        self.distribution_strategy = distribution_strategy

        # Compute the global k_array
        self.k_array = self.domain.get_distance_array(distribution_strategy)


    def _cast_config(self, logarithmic, nbin, binbounds):
        """
            internal helper function which casts the various combinations of
            possible parameters into a properly defaulted dictionary
        """

        try:
            temp_logarithmic = bool(logarithmic)
        except(TypeError):
            temp_logarithmic = False

        try:
            temp_nbin = int(nbin)
        except(TypeError):
            temp_nbin = None

        try:
            temp_binbounds = tuple(np.array(binbounds))
        except(TypeError):
            temp_binbounds = None

        return temp_logarithmic, temp_nbin, temp_binbounds

    def get_index_dict(self, logarithmic, nbin, binbounds):
        """
            Returns a dictionary containing the pindex, kindex and rho
            binned according to the supplied parameter scheme and a
            configuration dict containing this scheme.

            Parameters
            ----------
            logarithmic : bool
                Flag specifying if the binning is performed on logarithmic
                scale.
            nbin : integer
                Number of used bins.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins.

            Returns
            -------
            index_dict : dict
                Contains the keys: 'config', 'pindex', 'kindex' and 'rho'
        """
        # Cast the input arguments
        loarithmic, nbin, binbounds = self._cast_config(logarithmic, nbin, binbounds)
        pindex, kindex, rho, k_array = self._compute_index_dict(logarithmic, nbin, binbounds)
        # Return the plain result.
        return pindex, kindex, rho, k_array

    def _compute_index_dict(self, logarithmic, nbin, binbounds):
        """
            Internal helper function which takes a config_dict, asks for the
            pindex/kindex/rho set, and bins them according to the config
        """
        # if no binning is requested, compute the indices, build the dict,
        # and return it straight.
        if not logarithmic and nbin is None and binbounds is None:
            (temp_pindex, temp_kindex, temp_rho) =\
                self._compute_indices(self.k_array)
            temp_k_array = self.k_array

        # if binning is required, make a recursive call to get the unbinned
        # indices, bin them, and then return everything.
        else:
            # Get the unbinned indices
            pindex, kindex, rho, dummy = self.get_index_dict(nbin=None,
                                                        binbounds=None,
                                                        logarithmic=False)
            # Bin them
            (temp_pindex, temp_kindex, temp_rho) = \
                self._bin_power_indices(
                    pindex, kindex, rho, logarithmic, nbin, binbounds)
            # Make a binned version of k_array
            temp_k_array = self._compute_k_array_from_pindex_kindex(
                               temp_pindex, temp_kindex)

        return temp_pindex, temp_kindex, temp_rho, temp_k_array

    def _compute_k_array_from_pindex_kindex(self, pindex, kindex):
        tempindex = pindex.copy(dtype=kindex.dtype)
        result = tempindex.apply_scalar_function(
                            lambda x: kindex[x.astype(np.dtype('int'))])
        return result

    def _compute_indices(self, k_array):
        """
        Internal helper function which computes pindex, kindex and rho
        from a given k_array
        """
        ##########
        # kindex #
        ##########
        global_kindex = k_array.unique()

        ##########
        # pindex #
        ##########
        # compute the local pindex slice on basis of the local k_array data
        local_pindex = np.searchsorted(global_kindex, k_array.get_local_data())
        # prepare the distributed_data_object
        global_pindex = distributed_data_object(
                            global_shape=k_array.shape,
                            dtype=local_pindex.dtype,
                            distribution_strategy=self.distribution_strategy)
        # store the local pindex data in the global_pindex d2o
        global_pindex.set_local_data(local_pindex)

        #######
        # rho #
        #######
        global_rho = global_pindex.bincount().get_full_data()

        return global_pindex, global_kindex, global_rho

    def _bin_power_indices(self, pindex, kindex, rho, logarithmic, nbin, binbounds):
        """
            Returns the binned power indices associated with the Fourier grid.

            Parameters
            ----------
            pindex : distributed_data_object
                Index of the Fourier grid points in a distributed_data_object.
            kindex : ndarray
                Array of all k-vector lengths.
            rho : ndarray
                Degeneracy factor of the individual k-vectors.
            logarithmic : bool
                Flag specifying if the binning is performed on logarithmic
                scale.
            nbin : integer
                Number of used bins.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins.

            Returns
            -------
            pindex : distributed_data_object
            kindex, rho : ndarrays
                The (re)binned power indices.

        """

        # boundaries
        if(binbounds is not None):
            binbounds = np.sort(binbounds)
        # equal binning
        else:
            if(logarithmic is None):
                logarithmic = False
            if(logarithmic):
                k = np.r_[0, np.log(kindex[1:])]
            else:
                k = kindex
            dk = np.max(k[2:] - k[1:-1])  # minimal dk
            if(nbin is None):
                nbin = int((k[-1] - 0.5 * (k[2] + k[1])) /
                           dk - 0.5)  # maximal nbin
            else:
                nbin = min(int(nbin), int(
                    (k[-1] - 0.5 * (k[2] + k[1])) / dk + 2.5))
                dk = (k[-1] - 0.5 * (k[2] + k[1])) / (nbin - 2.5)
            binbounds = np.r_[0.5 * (3 * k[1] - k[2]),
                              0.5 * (k[1] + k[2]) + dk * np.arange(nbin - 2)]
            if(logarithmic):
                binbounds = np.exp(binbounds)
        # reordering
        reorder = np.searchsorted(binbounds, kindex)
        rho_ = np.zeros(len(binbounds) + 1, dtype=rho.dtype)
        kindex_ = np.empty(len(binbounds) + 1, dtype=kindex.dtype)
        for ii in range(len(reorder)):
            if(rho_[reorder[ii]] == 0):
                kindex_[reorder[ii]] = kindex[ii]
                rho_[reorder[ii]] += rho[ii]
            else:
                kindex_[reorder[ii]] = ((kindex_[reorder[ii]] *
                                         rho_[reorder[ii]] +
                                         kindex[ii] * rho[ii]) /
                                        (rho_[reorder[ii]] + rho[ii]))
                rho_[reorder[ii]] += rho[ii]

        pindex_ = pindex.copy_empty()
        pindex_.set_local_data(reorder[pindex.get_local_data()])

        return pindex_, kindex_, rho_
