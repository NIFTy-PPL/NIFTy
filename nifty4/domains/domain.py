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

import abc
from ..utilities import NiftyMeta
from future.utils import with_metaclass
import numpy as np


class Domain(with_metaclass(
        NiftyMeta, type('NewBase', (object,), {}))):
    """The abstract class repesenting a (structured or unstructured) domain.
    """

    def __init__(self):
        self._needed_for_hash = []

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        """Returns a hash value for the object.

        Notes
        -----
            Only members that are explicitly added to
            :py:attr:`._needed_for_hash` will be used for hashing.
        """
        result_hash = 0
        for key in self._needed_for_hash:
            result_hash ^= hash(vars(self)[key])
        return result_hash

    def __eq__(self, x):
        """Checks whether two domains are equal.

        Parameters
        ----------
        x : Domain
            The domain `self` is compared to.

        Returns
        -------
        bool : True iff `self` and x describe the same domain.
        """
        if self is x:  # shortcut for simple case
            return True
        if not isinstance(x, type(self)):
            return False
        for key in self._needed_for_hash:
            if vars(self)[key] != vars(x)[key]:
                return False
        return True

    def __ne__(self, x):
        """Returns the opposite of :py:meth:`.__eq__()`"""
        return not self.__eq__(x)

    @abc.abstractproperty
    def shape(self):
        """tuple of ints: number of pixels along each axis

        The shape of the array-like object required to store information
        living on the domain.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def size(self):
        """int: total number of pixels.

        Equivalent to the products over all entries in the domain's shape.
        """
        raise NotImplementedError
