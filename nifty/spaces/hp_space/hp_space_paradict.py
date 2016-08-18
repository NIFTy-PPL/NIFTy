# -*- coding: utf-8 -*-

from nifty.config import about
from nifty.spaces.space import SpaceParadict


class HPSpaceParadict(SpaceParadict):

    def __init__(self, nside, distances):
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        SpaceParadict.__init__(self, nside=nside, distances=distances)

    def __setitem__(self, key, arg):
        if key not in ['nside', 'distances']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported hp_space parameter"))

        if key == 'nside':
            temp = int(arg)
            # if(not hp.isnsideok(nside)):
            if ((temp & (temp - 1)) != 0) or (temp < 2):
                raise ValueError("ERROR: invalid parameter ( nside <> 2**n ).")
        elif key == 'distances':
            temp = (arg,)

        self.parameters.__setitem__(key, temp)
