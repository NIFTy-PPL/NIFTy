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
        result_hash = 0
        for key in self._needed_for_hash:
            result_hash ^= hash(vars(self)[key])
        return result_hash

    def __eq__(self, x):
        """Checks if two domains are equal.

        Parameters
        ----------
        x: Domain
            The domain `self` is compared to.

        Returns
        -------
        bool: True iff `self` and x describe the same domain.
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
        return not self.__eq__(x)

    @abc.abstractproperty
    def shape(self):
        """The shape of the array-like object required to store information
        living on the domain.

        Returns
        -------
        tuple of ints: shape of the required array-like object
        """
        raise NotImplementedError

    @abc.abstractproperty
    def size(self):
        """Number of data elements associated with this domain.
        Equivalent to the products over all entries in the domain's shape.

        Returns
        -------
        int: number of data elements
        """
        raise NotImplementedError