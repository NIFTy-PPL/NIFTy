import collections
from ..domain_tuple import DomainTuple

__all = ["MultiDomain"]


class frozendict(collections.Mapping):
    """
    An immutable wrapper around dictionaries that implements the complete
    :py:class:`collections.Mapping` interface. It can be used as a drop-in
    replacement for dictionaries where immutability is desired.
    """

    dict_cls = dict

    def __init__(self, *args, **kwargs):
        self._dict = self.dict_cls(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self._dict)

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self._dict.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash


class MultiDomain(frozendict):
    _domainCache = {}
    _subsetCache = set()
    _compatCache = set()

    def __init__(self, domain, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError(
                'To create a MultiDomain call `MultiDomain.make()`.')
        super(MultiDomain, self).__init__(domain)

    @staticmethod
    def make(domain):
        if isinstance(domain, MultiDomain):
            return domain
        if not isinstance(domain, dict):
            raise TypeError("dict expected")
        tmp = {}
        for key, value in domain.items():
            if not isinstance(key, str):
                raise TypeError("keys must be strings")
            tmp[key] = DomainTuple.make(value)
        domain = frozendict(tmp)
        obj = MultiDomain._domainCache.get(domain)
        if obj is not None:
            return obj
        obj = MultiDomain(domain, _callingfrommake=True)
        MultiDomain._domainCache[domain] = obj
        return obj

    def __eq__(self, x):
        if not isinstance(x, MultiDomain):
            x = MultiDomain.make(x)
        return self is x

    def __ne__(self, x):
        return not self.__eq__(x)

    def compatibleTo(self, x):
        if not isinstance(x, MultiDomain):
            x = MultiDomain.make(x)
        if (self, x) in MultiDomain._compatCache:
            return True
        commonKeys = set(self.keys()) & set(x.keys())
        for key in commonKeys:
            if self[key] != x[key]:
                return False
        MultiDomain._compatCache.add((self, x))
        MultiDomain._compatCache.add((x, self))
        return True

    def subsetOf(self, x):
        if not isinstance(x, MultiDomain):
            x = MultiDomain.make(x)
        if (self, x) in MultiDomain._subsetCache:
            return True
        if len(x) == 0:
            MultiDomain._subsetCache.add((self, x))
            return True
        for key in self.keys():
            if key not in x:
                return False
            if self[key] != x[key]:
                return False
        MultiDomain._subsetCache.add((self, x))
        return True

    def unitedWith(self, x):
        if not isinstance(x, MultiDomain):
            x = MultiDomain.make(x)
        if self == x:
            return self
        if not self.compatibleTo(x):
            raise ValueError("domain mismatch")
        res = {}
        for key, val in self.items():
            res[key] = val
        for key, val in x.items():
            res[key] = val
        return MultiDomain.make(res)
