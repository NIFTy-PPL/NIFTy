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

from functools import reduce

import numpy as np

from . import utilities
from .domains.domain import Domain


class DomainTuple:
    """Ordered sequence of Domain objects.

    This class holds a tuple of :class:`Domain` objects, which together form
    the space on which a :class:`Field` is defined.
    This corresponds to a tensor product of the corresponding vector
    fields.

    Notes
    -----

    DomainTuples should never be created using the constructor, but rather
    via the factory function :attr:`make`!
    """
    _tupleCache = {}
    _scalarDomain = None

    def __init__(self, domain, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        self._dom = self._parse_domain(domain)
        self._axtuple = self._get_axes_tuple()
        self._shape = reduce(lambda x, y: x+y, (sp.shape for sp in self._dom),
                             ())
        self._size = reduce(lambda x, y: x*y, self._shape, 1)

    def _get_axes_tuple(self):
        i = 0
        res = [None]*len(self._dom)
        for idx, thing in enumerate(self._dom):
            nax = len(thing.shape)
            res[idx] = tuple(range(i, i+nax))
            i += nax
        return tuple(res)

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
        from .multi_domain import MultiDomain
        if isinstance(domain, MultiDomain):
            raise TypeError("Cannot create DomainTuple from MultiDomain")
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
        defined on the DomainTuple.
        """
        return self._shape

    @property
    def size(self):
        """int : total number of pixels.

        Equivalent to the products over all entries in the object's shape.
        """
        return self._size

    def scalar_weight(self, spaces=None):
        """Returns the uniform volume element for a sub-domain of `self`.

        Parameters
        ----------
        spaces : int, tuple of int or None
            Indices of the sub-domains to be considered.
            If `None`, the entire domain is used.

        Returns
        -------
        float or None
            If the requested sub-domain has a uniform volume element, it is
            returned. Otherwise, `None` is returned.
        """
        if np.isscalar(spaces):
            return self._dom[spaces].scalar_dvol

        if spaces is None:
            spaces = range(len(self._dom))
        res = 1.
        for i in spaces:
            tmp = self._dom[i].scalar_dvol
            if tmp is None:
                return None
            res *= tmp
        return res

    def total_volume(self, spaces=None):
        """Returns the total volume of `self` or of a subspace of it.

        Parameters
        ----------
        spaces : int, tuple of int or None
            Indices of the sub-domains of the domain to be considered.
            If `None`, the total volume of the whole domain is returned.

        Returns
        -------
        float
            the total volume of the requested (sub-)domain.
        """
        if np.isscalar(spaces):
            return self._dom[spaces].total_volume

        if spaces is None:
            spaces = range(len(self._dom))
        res = 1.
        for i in spaces:
            res *= self._dom[i].total_volume
        return res

    @property
    def axes(self):
        """tuple of tuple of int : axis indices of the underlying domains"""
        return self._axtuple

    def __len__(self):
        return len(self._dom)

    def __hash__(self):
        return self._dom.__hash__()

    def __eq__(self, x):
        return (self is x) or (isinstance(x, DomainTuple) and self._dom == x._dom)

    def __ne__(self, x):
        return not self.__eq__(x)

    def __str__(self):
        if len(self) == 0:
            return "DomainTuple, len: 0"
        return ("DomainTuple, len: {}\n".format(len(self)) +
                "\n".join("* " + str(i) for i in self))

    def __reduce__(self):
        return (_unpickleDomainTuple, (self._dom,))

    @staticmethod
    def scalar_domain():
        if DomainTuple._scalarDomain is None:
            DomainTuple._scalarDomain = DomainTuple.make(())
        return DomainTuple._scalarDomain

    def __repr__(self):
        return f"DomainTuple.make({tuple(self)})"


def _unpickleDomainTuple(*args):
    return DomainTuple.make(*args)
