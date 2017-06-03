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

from nifty.spaces.space import Space


class PowerSpace(Space):
    """ NIFTY class for spaces of power spectra.

    Parameters
    ----------
    harmonic_partner : Space
        The harmonic Space of which this is the power space.
    distribution_strategy : str *optional*
        The distribution strategy used for the distributed_data_objects
        derived from this PowerSpace, e.g. the pindex.
        (default : 'not')
    logarithmic : bool *optional*
        True if logarithmic binning should be used (default : False).
    nbin : {int, None} *optional*
        The number of bins that should be used for power spectrum binning
        (default : None).
        if nbin == None, then nbin is set to the length of kindex.
    binbounds :  {list, array-like} *optional*
        Boundaries between the power spectrum bins.
        (If binbounds has n entries, there will be n+1 bins, the first bin
        starting at -inf, the last bin ending at +inf.)
        (default : None)
        if binbounds == None :
            Calculates the bounds from the kindex while applying the
            logarithmic and nbin keywords.

    Attributes
    ----------
    pindex : distributed_data_object
        This holds the information which pixel of the harmonic partner gets
        mapped to which power bin
    kindex : numpy.ndarray
        Sorted array of all k-modes.
    rho : numpy.ndarray
        The amount of k-modes that get mapped to one power bin is given by
        rho.
    dim : np.int
        Total number of dimensionality, i.e. the number of pixels.
    harmonic : bool
        Specifies whether the space is a signal or harmonic space.
    total_volume : np.float
        The total volume of the space.
    shape : tuple of np.ints
        The shape of the space's data array.
    binbounds :  {list, array-like}
        Boundaries between the power spectrum bins

    Notes
    -----
    A power space is the result of a projection of a harmonic space where
    k-modes of equal length get mapped to one power index.

    """

    # ---Overwritten properties and methods---

    def __init__(self, harmonic_partner,
                 distribution_strategy='not',
                 logarithmic=None, nbin=None, binbounds=None):
        super(PowerSpace, self).__init__()
        self._ignore_for_hash += ['_pindex', '_kindex', '_rho']

        if not (isinstance(harmonic_partner, Space) and \
                harmonic_partner.harmonic):
            raise ValueError("harmonic_partner must be a harmonic space.")
        self._harmonic_partner = harmonic_partner

        dists = self._harmonic_partner.get_distance_array(distribution_strategy)

        if logarithmic is None and nbin is None and binbounds is None:
            # compute the "natural" binning, i.e. there
            # is one bin for every distinct k-vector length
            tmp = dists.unique()
            tol = 1e-12*tmp[-1]
            # remove all points that are closer than tol to their right
            # neighbors.
            # I'm appending the last value*2 to the array to treat the
            # rightmost point correctly.
            tmp = tmp[np.diff(np.append(tmp,2*tmp[-1]))>tol]
            self._binbounds = tmp[0:-1]+0.5*np.diff(tmp)
        else:
            if binbounds is not None:
                self._binbounds = np.sort(tuple(np.array(binbounds)))
            else:
                if logarithmic is not None:
                    logarithmic = bool(logarithmic)
                if nbin is not None:
                    nbin = int(nbin)

                # equidistant binning (linear or log)
                # MR FIXME: this needs to improve
                kindex = dists.unique()
                if (logarithmic):
                    k = np.r_[0, np.log(kindex[1:])]
                else:
                    k = kindex
                dk = np.max(k[2:] - k[1:-1])  # minimum dk to avoid empty bins
                if(nbin is None):
                    nbin = int((k[-1] - 0.5 * (k[2] + k[1])) /
                               dk - 0.5)  # maximal nbin
                else:
                    nbin = min(int(nbin), int(
                        (k[-1] - 0.5 * (k[2] + k[1])) / dk + 2.5))
                    dk = (k[-1] - 0.5 * (k[2] + k[1])) / (nbin - 2.5)
                self._binbounds = np.r_[0.5 * (3 * k[1] - k[2]),
                                  0.5 * (k[1] + k[2]) + dk * np.arange(nbin - 2)]
                if(logarithmic):
                    self._binbounds = np.exp(self._binbounds)

        # Compute pindex, kindex and rho according to self._binbounds

        # prepare the pindex object
        self._pindex = distributed_data_object(
            global_shape=dists.shape,
            dtype=np.int,
            distribution_strategy=distribution_strategy)

        # bin according to the binbounds
        self._pindex.set_local_data(np.searchsorted(
            self._binbounds, dists.get_local_data()))
        self._rho = self._pindex.bincount().get_full_data()
        self._kindex = self._pindex.bincount(
            weights=dists).get_full_data()/self._rho

    def pre_cast(self, x, axes):
        """ Casts power spectrum functions to discretized power spectra.

        This function takes an array or a function. If it is an array it does
        nothing, otherwise it interpretes the function as power spectrum and
        evaluates it at every k-mode.

        Parameters
        ----------
        x : {array-like, function array-like -> array-like}
            power spectrum given either in discretized form or implicitly as a
            function
        axes : tuple of ints
            Specifies the axes of x which correspond to this space. For
            explicifying the power spectrum function, this is ignored.

        Returns
        -------
        array-like
            discretized power spectrum

        """

        return x(self.kindex) if callable(x) else x

    # ---Mandatory properties and methods---

    def __repr__(self):
        return ("PowerSpace(harmonic_partner=%r, distribution_strategy=%r, "
                "binbounds=%r)"
                % (self.harmonic_partner, self.pindex.distribution_strategy,
                   self._binbounds))

    @property
    def harmonic(self):
        return True

    @property
    def shape(self):
        return self.kindex.shape

    @property
    def dim(self):
        return self.shape[0]

    @property
    def total_volume(self):
        # every power-pixel has a volume of 1
        return float(reduce(lambda x, y: x*y, self.pindex.shape))

    def copy(self):
        distribution_strategy = self.pindex.distribution_strategy
        return self.__class__(harmonic_partner=self.harmonic_partner,
                              distribution_strategy=distribution_strategy,
                              binbounds=self._binbounds)

    def weight(self, x, power=1, axes=None, inplace=False):
        reshaper = [1, ] * len(x.shape)
        # we know len(axes) is always 1
        reshaper[axes[0]] = self.shape[0]

        weight = self.rho.reshape(reshaper)
        if power != 1:
            weight = weight ** np.float(power)

        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x*weight

        return result_x

    def get_distance_array(self, distribution_strategy):
        return distributed_data_object(
                                self.kindex, dtype=np.float64,
                                distribution_strategy=distribution_strategy)

    def get_fft_smoothing_kernel_function(self, sigma):
        raise NotImplementedError(
            "There is no fft smoothing function for PowerSpace.")

    # ---Added properties and methods---

    @property
    def harmonic_partner(self):
        """ Returns the Space of which this is the power space.
        """
        return self._harmonic_partner

    @property
    def binbounds(self):
        return self._binbounds

    @property
    def pindex(self):
        """ A distributed_data_object having the shape of the harmonic partner
        space containing the indices of the power bin a pixel belongs to.
        """
        return self._pindex

    @property
    def kindex(self):
        """ Sorted array of all k-modes.
        """
        return self._kindex

    @property
    def rho(self):
        """Degeneracy factor of the individual k-vectors.
        """
        return self._rho

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['kindex'] = self.kindex
        hdf5_group['rho'] = self.rho
        hdf5_group.attrs['binbounds'] = str(self._binbounds)

        #MR FIXME: why not "return None" as happens everywhere else?
        return {
            'harmonic_partner': self.harmonic_partner,
            'pindex': self.pindex,
        }

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        # make an empty PowerSpace object
        new_ps = EmptyPowerSpace()
        # reset class
        new_ps.__class__ = cls
        # call instructor so that classes are properly setup
        super(PowerSpace, new_ps).__init__()
        # set all values
        new_ps._harmonic_partner = repository.get('harmonic_partner',
                                                  hdf5_group)

        exec("new_ps._binbounds = " + hdf5_group.attrs['binbounds'])

        new_ps._pindex = repository.get('pindex', hdf5_group)
        new_ps._kindex = hdf5_group['kindex'][:]
        new_ps._rho = hdf5_group['rho'][:]
        new_ps._ignore_for_hash += ['_pindex', '_kindex', '_rho']

        return new_ps


class EmptyPowerSpace(PowerSpace):
    def __init__(self):
        pass
