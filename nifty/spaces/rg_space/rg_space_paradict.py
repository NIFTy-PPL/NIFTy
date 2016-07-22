# -*- coding: utf-8 -*-

import numpy as np
from nifty.config import about
from nifty.spaces.space import SpaceParadict


class RGSpaceParadict(SpaceParadict):

    def __init__(self, shape, zerocenter, distances, harmonic):
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        self.parameters.__setitem__('shape', shape)
        self.parameters.__setitem__('harmonic', harmonic)
        SpaceParadict.__init__(
            self, zerocenter=zerocenter, distances=distances)

    def __setitem__(self, key, arg):
        if key not in ['shape', 'zerocenter', 'distances']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported RGSpace parameter:" + key))

        if key == 'shape':
            temp = np.empty(len(self['shape']), dtype=np.int)
            temp[:] = arg
            temp = tuple(temp)
        elif key == 'zerocenter':
            temp = np.empty(len(self['shape']), dtype=bool)
            temp[:] = arg
            temp = tuple(temp)
        elif key == 'distances':
            if arg is None:
                if self['harmonic']:
                    temp = np.ones_like(self['shape'], dtype=np.float)
                else:
                    temp = 1 / np.array(self['shape'], dtype=np.float)
            else:
                temp = np.empty(len(self['shape']), dtype=np.float)
                temp[:] = arg
            temp = tuple(temp)
        elif key == 'harmonic':
            temp = bool(arg)

        self.parameters.__setitem__(key, temp)
