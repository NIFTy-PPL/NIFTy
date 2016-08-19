# -*- coding: utf-8 -*-

class Paradict(object):

    def __init__(self, **kwargs):
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        for key in kwargs:
            self[key] = kwargs[key]

    def __iter__(self):
        return self.parameters.__iter__()

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.parameters.__repr__()

    def __setitem__(self, key, arg):
        raise NotImplementedError

    def __getitem__(self, key):
        return self.parameters.__getitem__(key)

    def __hash__(self):
        result_hash = 0
        for (key, item) in self.parameters.items():
            try:
                temp_hash = hash(item)
            except TypeError:
                temp_hash = hash(tuple(item))
            result_hash ^= temp_hash ^ int(hash(key)/131)
        return result_hash
