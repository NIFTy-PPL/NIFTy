# -*- coding: utf-8 -*-


class Energy(object):
    def __init__(self, position):
        self._cache = {}
        try:
            position = position.copy()
        except AttributeError:
            pass
        self.position = position

    def at(self, position):
        return self.__class__(position)

    @property
    def value(self):
        raise NotImplementedError

    @property
    def gradient(self):
        raise NotImplementedError

    @property
    def curvature(self):
        raise NotImplementedError

    def memo(f):
        name = id(f)

        def wrapped_f(self):
            try:
                return self._cache[name]
            except KeyError:
                self._cache[name] = f(self)
                return self._cache[name]
        return wrapped_f

