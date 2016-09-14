
class _TransformationCache(object):
    def __init__(self):
        self.cache = {}

    def create(self, transformation_class, domain, codomain, module):
        key = domain.__hash__() ^ ((codomain.__hash__()/111) ^
                                   (module.__hash__())/179)
        if key not in self.cache:
            self.cache[key] = transformation_class(domain, codomain, module)

        return self.cache[key]

TransformationCache = _TransformationCache()
