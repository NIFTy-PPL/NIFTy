# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

import abc

from keepers import Loggable,\
                    Versionable


class DomainObject(Versionable, Loggable, object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # _global_id is used in the Versioning module from keepers
        self._ignore_for_hash = ['_global_id']

    def __hash__(self):
        # Extract the identifying parts from the vars(self) dict.
        result_hash = 0
        for key in sorted(vars(self).keys()):
            item = vars(self)[key]
            if key in self._ignore_for_hash or key == '_ignore_for_hash':
                continue
            result_hash ^= item.__hash__() ^ int(hash(key)/117)
        return result_hash

    def __eq__(self, x):
        """ Checks if two domain_objects are equal.

        Parameters
        ----------
        x: domain_object
            The domain_object `self` is compared to.

        Returns
        -------
        bool
            True if `self` and x describe the same manifold.

        """

        if isinstance(x, type(self)):
            for key in vars(self).keys():
                item1 = vars(self)[key]
                if key in self._ignore_for_hash or key == '_ignore_for_hash':
                    continue
                item2 = vars(x)[key]
                if item1 != item2:
                    return False
            return True
        else:
            return False

    def __ne__(self, x):
        return not self.__eq__(x)

    @abc.abstractproperty
    def shape(self):
        """ Returns the shape of the underlying array-like object.

        Returns
        -------
        tuple of ints
            The shape of the underlying array-like object.

        Raises
        ------
        NotImplementedError
            If called for this abstract class.

        """

        raise NotImplementedError(
            "There is no generic shape for DomainObject.")

    @abc.abstractproperty
    def dim(self):
        """ Returns the number of pixel-dimensions the object has.

        Returns
        -------
        int
            An Integer representing the number of pixels the discretized
            manifold has.

        Raises
        ------
        NotImplementedError
            If called for this abstract class.

        """

        raise NotImplementedError(
            "There is no generic dim for DomainObject.")

    @abc.abstractmethod
    def weight(self, x, power=1, axes=None, inplace=False):
        """ Weights the field on this domain with the space's volume-weights.

        Weights hereby refer to integration weights, as they appear in
        discretized integrals. Per default, this function mutliplies each bin
        of the field x by its volume, which lets it behave like a density
        (top form). However, different powers of the volume can be applied
        with the power parameter. The axes parameter specifies which of the
        field array's indices correspond to this domain.

        Parameters
        ----------
        x : distributed_data_object
            The fields data array.
        power : int, *optional*
            The power to which the volume-weight is raised (default: 1).
        axes : {int, tuple}, *optional*
            Specifies the axes of x which represent this domain
            (default: None).
            If axes==None:
                weighting is applied with respect to all axes
        inplace : bool, *optional*
            If this is True, the weighting is done on the values of x,
            if it is False, x is not modified and this method returns a
            weighted copy of x (default: False).

        Returns
        -------
        distributed_data_object
            A weighted version of x, with volume-weights raised to the
            given power.

        Raises
        ------
        NotImplementedError
            If called for this abstract class.

        """
        raise NotImplementedError(
            "There is no generic weight-method for DomainObject.")

    def pre_cast(self, x, axes):
        """ Casts input for Field.val before Field performs the cast.

        Parameters
        ----------
        x : {array-like, castable}
            an array-like object or anything that can be cast to arrays.
        axes : tuple of ints
            Specifies the axes of x which correspond to this domain.

        Returns
        -------
        {array-like, castable}
            Processed input where casting that needs Space-specific knowledge
            (for example location of pixels on the manifold) was performed.


        See Also
        --------
        post_cast

        Notes
        -----
            Usually returns x, except if a power spectrum is given to a
            PowerSpace, where this spectrum is evaluated at the power indices.

        """

        return x

    def post_cast(self, x, axes):
        """ Performs casting operations that are done after Field's cast.

        Parameters
        ----------
        x : {array-like, castable}
            an array-like object or anything that can be cast to arrays.
        axes : tuple of ints
            Specifies the axes of x which correspond to this domain.

        See Also
        --------
        pre_cast

        Returns
        -------
        distributed_data_object
            Processed input where casting that needs Space-specific knowledge
            (for example location of pixels on the manifold) was performed.

        """

        return x

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        result = cls()
        return result
