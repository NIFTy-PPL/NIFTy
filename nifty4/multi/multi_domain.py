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

    def __init__(self, domain, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
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
