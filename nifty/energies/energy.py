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
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position

    @property
    def value(self):
        raise NotImplementedError

    @property
    def gradient(self):
        raise NotImplementedError

    @property
    def curvature(self):
        raise NotImplementedError
