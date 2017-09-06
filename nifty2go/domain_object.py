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

from __future__ import division
import abc
from .nifty_meta import NiftyMeta

from future.utils import with_metaclass


class DomainObject(with_metaclass(
        NiftyMeta, type('NewBase', (object,), {}))):
    """The abstract class that can be used as a domain for a field.

    This holds all the information and functionality a field needs to know
    about its domain and how the data of the field are stored.

    Attributes
    ----------
    dim : int
        Number of pixel-dimensions of the underlying data object.
    shape : tuple
        Shape of the array that stores the degrees of freedom for any field
        on this domain.

    """

    def __init__(self):
        self._ignore_for_hash = []

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        # Extract the identifying parts from the vars(self) dict.
        result_hash = 0
        for key in sorted(vars(self).keys()):
            item = vars(self)[key]
            if key in self._ignore_for_hash or key == '_ignore_for_hash':
                continue
            result_hash ^= item.__hash__() ^ int(hash(key)//117)
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
            for key in list(vars(self).keys()):
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
        """ The domain-object's shape contribution to the underlying array.

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
    def scalar_weight(self):
        """ Returns the volume factors of this domain as a floating
        point scalar, if the volume factors are all identical, otherwise
        returns None.

        Returns
        -------
        float or None
            Volume factor

        Raises
        ------
        NotImplementedError
            If called for this abstract class.

        """
        raise NotImplementedError(
            "There is no generic scalar_weight method for DomainObject.")

    @abc.abstractmethod
    def weight(self):
        """ Returns the volume factors of this domain, either as a floating
        point scalar (if the volume factors are all identical) or as a
        floating point array with a shape of `self.shape`.


        Returns
        -------
        float or numpy.ndarray(dtype=float)
            Volume factors

        Raises
        ------
        NotImplementedError
            If called for this abstract class.

        """
        raise NotImplementedError(
            "There is no generic weight method for DomainObject.")
