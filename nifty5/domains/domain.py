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

from __future__ import absolute_import, division, print_function
from ..compat import *
import abc
from ..utilities import NiftyMetaBase


class Domain(NiftyMetaBase()):
    """The abstract class repesenting a (structured or unstructured) domain.
    """
    def __init__(self):
        self._hash = None

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        """Returns a hash value for the object.

        Notes
        -----
        Only members that are explicitly added to
        :attr:`._needed_for_hash` will be used for hashing.
        """
        if self._hash is None:
            h = 0
            for key in self._needed_for_hash:
                h ^= hash(vars(self)[key])
            self._hash = h
        return self._hash

    def __eq__(self, x):
        """Checks whether two domains are equal.

        Parameters
        ----------
        x : Domain
            The domain `self` is compared to.

        Returns
        -------
        bool : True iff `self` and x describe the same domain.

        Notes
        -----
        Only members that are explicitly added to
        :attr:`._needed_for_hash` will be used for comparison.

        Subclasses of Domain should not re-define :meth:`__eq__`,
        :meth:`__ne__`, or :meth:`__hash__`; they should instead add their
        relevant attributes' names to :attr:`._needed_for_hash`.
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
        """Returns the opposite of :meth:`.__eq__()`"""
        return not self.__eq__(x)

    @abc.abstractproperty
    def shape(self):
        """tuple of int: number of pixels along each axis

        The shape of the array-like object required to store information
        living on the domain.
        """
        raise NotImplementedError

    @property
    def local_shape(self):
        """tuple of int: number of pixels along each axis on the local task

        The shape of the array-like object required to store information
        living on part of the domain which is stored on the local MPI task.
        """
        from ..dobj import local_shape
        return local_shape(self.shape)

    @abc.abstractproperty
    def size(self):
        """int: total number of pixels.

        Equivalent to the products over all entries in the domain's shape.
        """
        raise NotImplementedError
