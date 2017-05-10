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
        """Checks if this domain_object represents the same thing as another domain_object.
        Parameters
        ----------
        x: domain_object
            The domain_object it is compared to.
        Returns
        -------
        bool : True if they this and x represent the same thing.
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
        """Returns the shape of the underlying array-like object.
        Returns
        -------
        (int, tuple) : A tuple representing the shape of the underlying array-like object
        Raises
        ------
        NotImplementedError : If it is called for an abstract class, all non-abstract child-classes should
        implement this.
        """
        raise NotImplementedError(
            "There is no generic shape for DomainObject.")

    @abc.abstractproperty
    def dim(self):
        """Returns the number of pixel-dimensions the object has.
        Returns
        -------
        int : An Integer representing the number of pixels the discretized space has.
        Raises
        ------
        NotImplementedError : If it is called for an abstract class, all non-abstract child-classes should
        implement this.
        """
        raise NotImplementedError(
            "There is no generic dim for DomainObject.")

    @abc.abstractmethod
    def weight(self, x, power=1, axes=None, inplace=False):
        """ Weights a field living on this domain with a specified amount of volume-weights.

        Weights hereby refer to integration weights, as they appear in discretized integrals.
        Per default, this function mutliplies each bin of the field x by its volume, which lets
        it behave like a density (top form). However, different powers of the volume can be applied
        with the power parameter. The axes parameter specifies which of the field indices represent this
        domain.
        Parameters
        ----------
        x : Field
            A field with this space as domain to be weighted.
        power : int, *optional*
            The power to which the volume-weight is raised.
            (default: 1).
        axes : {int, tuple}, *optional*
            Specifies the axes of x which represent this domain.
            (default: None).
            If axes==None:
                weighting is applied with respect to all axes
        inplace : bool, *optional*
            If this is True, the weighting is done on the values of x,
            if it is False, x is not modified and this method returns a 
            weighted copy of x
            (default: False).
        Returns
        -------
        Field
            A weighted version of x, with volume-weights raised to power.
        Raises
        ------
        NotImplementedError : If it is called for an abstract class, all non-abstract child-classes should
        implement this.
        """
        raise NotImplementedError(
            "There is no generic weight-method for DomainObject.")

    def pre_cast(self, x, axes=None):
        # FIXME This does nothing and non of the children override this. Why does this exist?!
        return x

    def post_cast(self, x, axes=None):
        # FIXME This does nothing and non of the children override this. Why does this exist?!
        return x

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        result = cls()
        return result
