# -*- coding: utf-8 -*-


def memo(f):
    name = id(f)

    def wrapped_f(self):
        try:
            return self._cache[name]
        except KeyError:
            self._cache[name] = f(self)
            return self._cache[name]
    return wrapped_f
