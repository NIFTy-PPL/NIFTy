from ..domain_tuple import DomainTuple
from ..utilities import frozendict


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
        if self is x:
            return True
        x = MultiDomain.make(x)
        return self is x

    def __ne__(self, x):
        return not self.__eq__(x)

    def __hash__(self):
        return super(MultiDomain, self).__hash__()

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
