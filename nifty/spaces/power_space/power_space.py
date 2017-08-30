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

#from builtins import str
import ast
import numpy as np

from d2o import distributed_data_object,\
    STRATEGIES as DISTRIBUTION_STRATEGIES

from ...spaces.space import Space
from functools import reduce
from ...config import nifty_configuration as gc


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
    binbounds: None, or tuple/array/list of float
        if None:
            There will be as many bins as there are distinct k-vector lengths
            in the harmonic partner space.
            The "binbounds" property of the PowerSpace will also be None.

        else:
            the bin bounds requested for this PowerSpace. The array
            must be sorted and strictly ascending. The first entry is the right
            boundary of the first bin, and the last entry is the left boundary
            of the last bin, i.e. thee will be len(binbounds)+1 bins in total,
            with the first and last bins reaching to -+infinity, respectively.
        (default : None)

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
        Always True for this space.
    total_volume : np.float
        The total volume of the space.
    shape : tuple of np.ints
        The shape of the space's data array.
    binbounds : tuple or None
        Boundaries between the power spectrum bins; None is used to indicate
        natural binning

    Notes
    -----
    A power space is the result of a projection of a harmonic space where
    k-modes of equal length get mapped to one power index.

    """

    _powerIndexCache = {}

    # ---Overwritten properties and methods---

    @staticmethod
    def linear_binbounds (nbin, first_bound, last_bound):
        """
        nbin: integer
            the number of bins
        first_bound, last_bound: float
            the k values for the right boundary of the first bin and the left
            boundary of the last bin, respectively. They are given in length
            units of the harmonic partner space.
        This will produce a binbounds array with nbin-1 entries with
        binbounds[0]=first_bound and binbounds[-1]=last_bound and the remaining
        values equidistantly spaced (in linear scale) between these two.
        """
        nbin = int(nbin)
        assert nbin>=3, "nbin must be at least 3"
        return np.linspace(float(first_bound),float(last_bound),nbin-1)

    @staticmethod
    def logarithmic_binbounds (nbin, first_bound, last_bound):
        """
        nbin: integer
            the number of bins
        first_bound, last_bound: float
            the k values for the right boundary of the first bin and the left
            boundary of the last bin, respectively. They are given in length
            units of the harmonic partner space.
        This will produce a binbounds array with nbin-1 entries with
        binbounds[0]=first_bound and binbounds[-1]=last_bound and the remaining
        values equidistantly spaced (in natural logarithmic scale)
        between these two.
        """
        nbin = int(nbin)
        assert nbin>=3, "nbin must be at least 3"
        return np.logspace(np.log(float(first_bound)),
                           np.log(float(last_bound)),
                           nbin-1, base=np.e)

    def __init__(self, harmonic_partner, distribution_strategy=None,
                 binbounds=None):
        super(PowerSpace, self).__init__()
        self._ignore_for_hash += ['_pindex', '_kindex', '_rho']

        if distribution_strategy is None:
            distribution_strategy = gc['default_distribution_strategy']
        elif distribution_strategy not in DISTRIBUTION_STRATEGIES['global']:
            raise ValueError(
                    "distribution_strategy must be a global-type "
                    "strategy.")

        if not (isinstance(harmonic_partner, Space) and
                harmonic_partner.harmonic):
            raise ValueError("harmonic_partner must be a harmonic space.")
        self._harmonic_partner = harmonic_partner

        if binbounds is not None:
            binbounds = tuple(binbounds)

        key = (harmonic_partner, distribution_strategy, binbounds)
        if self._powerIndexCache.get(key) is None:
            distance_array = \
                self.harmonic_partner.get_distance_array(distribution_strategy)
            temp_pindex = self._compute_pindex(
                                harmonic_partner=self.harmonic_partner,
                                distance_array=distance_array,
                                binbounds=binbounds,
                                distribution_strategy=distribution_strategy)
            temp_rho = temp_pindex.bincount().get_full_data()
            assert not np.any(temp_rho==0), "empty bins detected"
            temp_kindex = \
                (temp_pindex.bincount(weights=distance_array).get_full_data() /
                 temp_rho)
            self._powerIndexCache[key] = (binbounds,
                                          temp_pindex,
                                          temp_kindex,
                                          temp_rho)

        (self._binbounds, self._pindex, self._kindex, self._rho) = \
            self._powerIndexCache[key]

    @staticmethod
    def _compute_pindex(harmonic_partner, distance_array, binbounds,
                        distribution_strategy):

        # Compute pindex, kindex and rho according to bb
        pindex = distributed_data_object(
                                global_shape=distance_array.shape,
                                dtype=np.int,
                                distribution_strategy=distribution_strategy)
        if binbounds is None:
            binbounds = harmonic_partner.get_natural_binbounds()
        pindex.set_local_data(
                np.searchsorted(binbounds, distance_array.get_local_data()))
        return pindex

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

    def weight(self, x, power, axes, inplace=False):
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
        hdf5_group.attrs['binbounds'] = str(self._binbounds)
        hdf5_group.attrs['distribution_strategy'] = \
            self._pindex.distribution_strategy

        return {
            'harmonic_partner': self.harmonic_partner,
        }

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        hp = repository.get('harmonic_partner', hdf5_group)
        bb = ast.literal_eval(hdf5_group.attrs['binbounds'])
        ds = hdf5_group.attrs['distribution_strategy']
        return PowerSpace(hp, ds, binbounds=bb)
