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

from functools import reduce
from .domain_object import DomainObject

class DomainTuple(object):
    _tupleCache = {}

    def __init__(self, domain):
        self._dom = self._parse_domain(domain)
        self._axtuple = self._get_axes_tuple()
        shape_tuple = tuple(sp.shape for sp in self._dom)
        self._shape = reduce(lambda x, y: x + y, shape_tuple, ())
        self._dim = reduce(lambda x, y: x * y, self._shape, 1)

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
        if isinstance(domain, DomainTuple):
            return domain
        domain = DomainTuple._parse_domain(domain)
        obj = DomainTuple._tupleCache.get(domain)
        if obj is not None:
            return obj
        obj = DomainTuple(domain)
        DomainTuple._tupleCache[domain] = obj
        return obj

    @staticmethod
    def _parse_domain(domain):
        if domain is None:
            return ()
        if isinstance(domain, DomainObject):
            return (domain,)

        if not isinstance(domain, tuple):
            domain = tuple(domain)
        for d in domain:
            if not isinstance(d, DomainObject):
                raise TypeError(
                    "Given object contains something that is not an "
                    "instance of DomainObject class.")
        return domain

    def __getitem__(self, i):
        return self._dom[i]

    @property
    def domains(self):
        return self._dom

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return self._dim

    @property
    def axes(self):
        return self._axtuple

    def __len__(self):
        return len(self._dom)

    def __hash__(self):
        return self._dom.__hash__()

    def __eq__(self, x):
        if not isinstance(x, DomainTuple):
            x = DomainTuple(x)
        return self._dom == x._dom

    def __ne__(self, x):
        if not isinstance(x, DomainTuple):
            x = DomainTuple(x)
        return self._dom != x._dom