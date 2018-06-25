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

from functools import reduce
from .domains.domain import Domain


class DomainTuple(object):
    """Ordered sequence of Domain objects.

    This class holds a set of :class:`Domain` objects, which together form the
    space on which a :class:`Field` is defined.

    Notes
    -----

    DomainTuples should never be created using the constructor, but rather
    via the factory function :attr:`make`!
    """
    _tupleCache = {}

    def __init__(self, domain, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        self._dom = self._parse_domain(domain)
        self._axtuple = self._get_axes_tuple()
        shape_tuple = tuple(sp.shape for sp in self._dom)
        self._shape = reduce(lambda x, y: x + y, shape_tuple, ())
        self._size = reduce(lambda x, y: x * y, self._shape, 1)

    def _get_axes_tuple(self):
        i = 0
        res = [None]*len(self._dom)
        for idx, thing in enumerate(self._dom):
            nax = len(thing.shape)
            res[idx] = tuple(range(i, i+nax))
            i += nax
        return res

    @staticmethod
    def make(domain):
        """Returns a DomainTuple matching `domain`.

        This function checks whether a matching DomainTuple already exists.
        If yes, this object is returned, otherwise a new DomainTuple object
        is created and returned.

        Parameters
        ----------
        domain : Domain or tuple of Domain or DomainTuple
            The geometrical structure for which the DomainTuple shall be
            obtained.
        """
        if isinstance(domain, DomainTuple):
            return domain
        domain = DomainTuple._parse_domain(domain)
        obj = DomainTuple._tupleCache.get(domain)
        if obj is not None:
            return obj
        obj = DomainTuple(domain, _callingfrommake=True)
        DomainTuple._tupleCache[domain] = obj
        return obj

    @staticmethod
    def _parse_domain(domain):
        if domain is None:
            return ()
        if isinstance(domain, Domain):
            return (domain,)

        if not isinstance(domain, tuple):
            domain = tuple(domain)
        for d in domain:
            if not isinstance(d, Domain):
                raise TypeError(
                    "Given object contains something that is not an "
                    "instance of Domain class.")
        return domain

    def __getitem__(self, i):
        return self._dom[i]

    @property
    def shape(self):
        """tuple of int: number of pixels along each axis

        The shape of the array-like object required to store information
        living on the DomainTuple.
        """
        return self._shape

    @property
    def size(self):
        """int : total number of pixels.

        Equivalent to the products over all entries in the object's shape.
        """
        return self._size

    @property
    def axes(self):
        """tuple of tuple of int : axis indices of the underlying domains"""
        return self._axtuple

    def __len__(self):
        return len(self._dom)

    def __hash__(self):
        return self._dom.__hash__()

    def __eq__(self, x):
        if self is x:
            return True
        x = DomainTuple.make(x)
        return self is x

    def __ne__(self, x):
        return not self.__eq__(x)

    def compatibleTo(self, x):
        return self.__eq__(x)

    def subsetOf(self, x):
        return self.__eq__(x)

    def unitedWith(self, x):
        if self is x:
            return self
        x = DomainTuple.make(x)
        if self is not x:
            raise ValueError("domain mismatch")
        return self

    def __str__(self):
        res = "DomainTuple, len: " + str(len(self))
        for i in self:
            res += "\n" + str(i)
        return res
