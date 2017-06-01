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

    Given the shape and the density of a underlying grid this class provides
    the user with the pindex, kindex and rho. The indices are binned
    according to the supplied parameter scheme.
    """
    @staticmethod
    def get_arrays(domain, distribution_strategy, logarithmic, nbin, binbounds):
        """
            Returns a dictionary containing the pindex, kindex and rho
            binned according to the supplied parameter scheme and a
            configuration dict containing this scheme.

            Parameters
            ----------
            domain : NIFTy harmonic space
                The space for which the power indices get computed
            distribution_strategy : str
                The distribution_strategy that will be used for the k_array and
                pindex distributed_data_object.
            logarithmic : bool
                Flag specifying if the binning is performed on logarithmic
                scale.
            nbin : integer
                Number of used bins.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins.

        """
        # if no binning is requested, compute the indices, build the dict,
        # and return it straight.
        if not logarithmic and nbin is None and binbounds is None:
            k_array = domain.get_distance_array(distribution_strategy)
            temp_kindex = k_array.unique()
            # compute the local pindex slice on basis of the local k_array data
            local_pindex = np.searchsorted(temp_kindex, k_array.get_local_data())
            # prepare the distributed_data_object
            temp_pindex = distributed_data_object(
                            global_shape=k_array.shape,
                            dtype=local_pindex.dtype,
                            distribution_strategy=distribution_strategy)
            # store the local pindex data in the global_pindex d2o
            temp_pindex.set_local_data(local_pindex)
            temp_rho = temp_pindex.bincount().get_full_data()
            temp_k_array = k_array

        # if binning is required, make a recursive call to get the unbinned
        # indices, bin them, and then return everything.
        else:
            # Get the unbinned indices
            pindex, kindex, rho, dummy = PowerIndices.get_arrays(domain,distribution_strategy,nbin=None,
                                                        binbounds=None,
                                                        logarithmic=False)
            # Bin them
            (temp_pindex, temp_kindex, temp_rho) = \
                PowerIndices._bin_power_indices(
                    pindex, kindex, rho, logarithmic, nbin, binbounds)
            # Make a binned version of k_array
            tempindex = temp_pindex.copy(dtype=temp_kindex.dtype)
            temp_k_array = tempindex.apply_scalar_function(
                            lambda x: temp_kindex[x.astype(np.dtype('int'))])

        return temp_pindex, temp_kindex, temp_rho, temp_k_array

    @staticmethod
    def _bin_power_indices(pindex, kindex, rho, logarithmic, nbin, binbounds):
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
            dk = np.max(k[2:] - k[1:-1])  # maximal dk
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
        rho_ = np.bincount(reorder,weights=rho).astype(rho.dtype)
        kindex_ = np.bincount(reorder,weights=kindex*rho)/rho_
        pindex_ = pindex.copy_empty()
        pindex_.set_local_data(reorder[pindex.get_local_data()])

        return pindex_, kindex_, rho_
