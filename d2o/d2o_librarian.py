# -*- coding: utf-8 -*-

from weakref import WeakValueDictionary as weakdict


class _d2o_librarian(object):

    def __init__(self):
        self.library = weakdict()
        self.counter = 0

    def register(self, d2o):
        self.counter += 1
        self.library[self.counter] = d2o
        return self.counter

    def __getitem__(self, key):
        return self.library[key]

d2o_librarian = _d2o_librarian()
