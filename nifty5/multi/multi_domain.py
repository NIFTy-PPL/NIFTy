from __future__ import absolute_import, division, print_function
from ..compat import *
from ..domain_tuple import DomainTuple
from ..utilities import frozendict


class MultiDomain(object):
    _domainCache = {}
    _subsetCache = set()
    _compatCache = set()

    def __init__(self, dict, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError(
                'To create a MultiDomain call `MultiDomain.make()`.')
        self._keys = tuple(sorted(dict.keys()))
        self._domains = tuple(dict[key] for key in self._keys)
        self._dict = frozendict({key: i for i, key in enumerate(self._keys)})

    @staticmethod
    def make(inp):
        if isinstance(inp, MultiDomain):
            return inp
        if not isinstance(inp, dict):
            raise TypeError("dict expected")
        tmp = {}
        for key, value in inp.items():
            if not isinstance(key, str):
                raise TypeError("keys must be strings")
            tmp[key] = DomainTuple.make(value)
        tmp = frozendict(tmp)
        obj = MultiDomain._domainCache.get(tmp)
        if obj is not None:
            return obj
        obj = MultiDomain(tmp, _callingfrommake=True)
        MultiDomain._domainCache[tmp] = obj
        return obj

    def keys(self):
        return self._keys

    def domains(self):
        return self._domains

    def items(self):
        return zip(self._keys, self._domains)

    def __getitem__(self, key):
        return self._domains[self._dict[key]]

    def __len__(self):
        return len(self._keys)

    def __hash__(self):
        return self._keys.__hash__() ^ self._domains.__hash__()

    def __eq__(self, x):
        if self is x:
            return True
        return self is MultiDomain.make(x)

    def __ne__(self, x):
        return not self.__eq__(x)

    def compatibleTo(self, x):
        if self is x:
            return True
        x = MultiDomain.make(x)
        if self is x:
            return True
        if (self, x) in MultiDomain._compatCache:
            return True
        commonKeys = set(self.keys()) & set(x.keys())
        for key in commonKeys:
            if self[key] is not x[key]:
                return False
        MultiDomain._compatCache.add((self, x))
        MultiDomain._compatCache.add((x, self))
        return True

    def subsetOf(self, x):
        if self is x:
            return True
        x = MultiDomain.make(x)
        if self is x:
            return True
        if len(x) == 0:
            return True
        if (self, x) in MultiDomain._subsetCache:
            return True
        for key in self.keys():
            if key not in x:
                return False
            if self[key] is not x[key]:
                return False
        MultiDomain._subsetCache.add((self, x))
        return True

    def unitedWith(self, x):
        if self is x:
            return self
        x = MultiDomain.make(x)
        if self is x:
            return self
        if not self.compatibleTo(x):
            raise ValueError("domain mismatch")
        res = {}
        for key, val in self.items():
            res[key] = val
        for key, val in x.items():
            res[key] = val
        return MultiDomain.make(res)
