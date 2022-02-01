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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .structured_domain import StructuredDomain


class PowerSpace(StructuredDomain):
    """Represents non-equidistantly binned spaces for power spectra.

    A power space is the result of a projection of a harmonic domain where
    k-modes of equal length get mapped to one power index.

    Parameters
    ----------
    harmonic_partner : StructuredDomain
        The harmonic domain of which this is the power space.
    binbounds : None, or tuple of float
        By default (binbounds=None):
            There are as many bins as there are distinct k-vector lengths in
            the harmonic partner space.
            The `binbounds` property of the PowerSpace will be None.
        else:
            The bin bounds requested for this PowerSpace. The array
            must be sorted and strictly ascending. The first entry is the right
            boundary of the first bin, and the last entry is the left boundary
            of the last bin, i.e. there will be `len(binbounds)+1` bins in
            total, with the first and last bins reaching to -+infinity,
            respectively.
    """

    _powerIndexCache = {}
    _needed_for_hash = ["_harmonic_partner", "_binbounds"]

    @staticmethod
    def linear_binbounds(nbin, first_bound, last_bound):
        """Produces linearly spaced bin bounds.

        Parameters
        ----------
        nbin : int
            the number of bins
        first_bound, last_bound : float
            the k values for the right boundary of the first bin and the left
            boundary of the last bin, respectively. They are given in length
            units of the harmonic partner space.

        Returns
        -------
        numpy.ndarray(numpy.float64)
            binbounds array with nbin-1 entries with
            binbounds[0]=first_bound and binbounds[-1]=last_bound and the
            remaining values equidistantly spaced (in linear scale) between
            these two.
        """
        nbin = int(nbin)
        if nbin < 3:
            raise ValueError("nbin must be at least 3")
        return np.linspace(float(first_bound), float(last_bound), nbin-1)

    @staticmethod
    def logarithmic_binbounds(nbin, first_bound, last_bound):
        """Produces logarithmically spaced bin bounds.

        Parameters
        ----------
        nbin : int
            the number of bins
        first_bound, last_bound : float
            the k values for the right boundary of the first bin and the left
            boundary of the last bin, respectively. They are given in length
            units of the harmonic partner space.

        Returns
        -------
        numpy.ndarray(numpy.float64)
            binbounds array with nbin-1 entries with
            binbounds[0]=first_bound and binbounds[-1]=last_bound and the
            remaining values equidistantly spaced (in natural logarithmic
            scale) between these two.
        """
        nbin = int(nbin)
        if nbin < 3:
            raise ValueError("nbin must be at least 3")
        return np.logspace(np.log(float(first_bound)),
                           np.log(float(last_bound)),
                           nbin-1, base=np.e)

    @staticmethod
    def useful_binbounds(space, logarithmic, nbin=None):
        """Produces bin bounds suitable for a given domain.

        Parameters
        ----------
        space : StructuredDomain
            the domain for which the binbounds will be computed.
        logarithmic : bool
            If True bins will have equal size in linear space; otherwise they
            will have equal size in logarithmic space.
        nbin : int, optional
            the number of bins
            If None, the highest possible number of bins will be used

        Returns
        -------
        numpy.ndarray(numpy.float64)
            Binbounds array with `nbin-1` entries, if `nbin` is
            supplied, or the maximum number of entries that does not produce
            empty bins, if `nbin` is not supplied.
            The first and last bin boundary are inferred from `space`.
        """
        if not (isinstance(space, StructuredDomain) and space.harmonic):
            raise ValueError("first argument must be a harmonic space.")
        if logarithmic is None and nbin is None:
            return None
        nbin = None if nbin is None else int(nbin)
        logarithmic = bool(logarithmic)
        dists = space.get_unique_k_lengths()
        if len(dists) < 3:
            raise ValueError("Space does not have enough unique k lengths")
        lbound = 0.5*(dists[0]+dists[1])
        rbound = 0.5*(dists[-2]+dists[-1])
        dists[0] = lbound
        dists[-1] = rbound
        if logarithmic:
            dists = np.log(dists)
        binsz_min = np.max(np.diff(dists))
        nbin_max = int((dists[-1]-dists[0])/binsz_min)+2
        if nbin is None:
            nbin = nbin_max
        if nbin < 3:
            raise ValueError("nbin must be at least 3")
        if nbin > nbin_max:
            raise ValueError("nbin is too large")
        if logarithmic:
            return PowerSpace.logarithmic_binbounds(nbin, lbound, rbound)
        else:
            return PowerSpace.linear_binbounds(nbin, lbound, rbound)

    def __init__(self, harmonic_partner, binbounds=None):
        if not (isinstance(harmonic_partner, StructuredDomain) and
                harmonic_partner.harmonic):
            raise ValueError("harmonic_partner must be a harmonic space.")
        if harmonic_partner.scalar_dvol is None:
            raise ValueError("harmonic partner must have "
                             "scalar volume factors")
        self._harmonic_partner = harmonic_partner
        pdvol = harmonic_partner.scalar_dvol

        if binbounds is not None:
            binbounds = tuple(binbounds)
            if min(binbounds) < 0:
                raise ValueError('Negative binbounds encountered')

        key = (harmonic_partner, binbounds)
        if self._powerIndexCache.get(key) is None:
            k_length_array = self.harmonic_partner.get_k_length_array()
            if binbounds is None:
                tmp = harmonic_partner.get_unique_k_lengths()
                tbb = 0.5*(tmp[:-1]+tmp[1:])
            else:
                tbb = binbounds
            temp_pindex = np.searchsorted(tbb, k_length_array.val)
            nbin = len(tbb)+1
            temp_rho = np.bincount(temp_pindex.ravel(), minlength=nbin)
            if (temp_rho == 0).any():
                raise ValueError("empty bins detected")
            # The explicit conversion to float64 is necessary because bincount
            # sometimes returns its result as an integer array, even when
            # floating-point weights are present ...
            temp_k_lengths = np.bincount(temp_pindex.ravel(),
                weights=k_length_array.val.ravel(),
                minlength=nbin).astype(np.float64, copy=False)
            temp_k_lengths = temp_k_lengths / temp_rho
            temp_k_lengths.flags.writeable = False
            temp_pindex.flags.writeable = False
            temp_dvol = temp_rho*pdvol
            temp_dvol.flags.writeable = False
            self._powerIndexCache[key] = (binbounds, temp_pindex,
                                          temp_k_lengths, temp_dvol)

        (self._binbounds, self._pindex, self._k_lengths, self._dvol) = \
            self._powerIndexCache[key]

    def __repr__(self):
        return ("PowerSpace(harmonic_partner={}, binbounds={})"
                .format(self.harmonic_partner, self._binbounds))

    @property
    def harmonic(self):
        """bool : Always False for this class."""
        return False

    @property
    def shape(self):
        return self.k_lengths.shape

    @property
    def size(self):
        return self.shape[0]

    @property
    def scalar_dvol(self):
        return None

    @property
    def dvol(self):
        return self._dvol

    @property
    def harmonic_partner(self):
        """StructuredDomain : the harmonic domain associated with `self`."""
        return self._harmonic_partner

    @property
    def binbounds(self):
        """None or tuple of float : inner bin boundaries

        The boundaries between bins, starting with the right boundary of the
        first bin, up to the left boundary of the last bin.

        `None` is used to indicate natural binning.
        """
        return self._binbounds

    @property
    def pindex(self):
        """numpy.ndarray : bin indices

        Bin index for every pixel in the harmonic partner.
        """
        return self._pindex

    @property
    def k_lengths(self):
        """numpy.ndarray(float) : k-vector length for each bin."""
        return self._k_lengths

    def __reduce__(self):
        return (_unpicklePowerSpace, (self._harmonic_partner, self._binbounds))


def _unpicklePowerSpace(*args):
    return PowerSpace(*args)
