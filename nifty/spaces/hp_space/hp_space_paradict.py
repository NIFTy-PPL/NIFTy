# -*- coding: utf-8 -*-

from nifty.config import about
from nifty.spaces.space import SpaceParadict


class HPSpaceParadict(SpaceParadict):

    def __init__(self, nside):
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        SpaceParadict.__init__(self, nside=nside)

    def __setitem__(self, key, arg):
        if key not in ['nside', 'distances']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported hp_space parameter"))

        if key == 'nside':
            temp = int(arg)
            if ((temp & (temp - 1)) != 0) or (temp < 2):
                raise ValueError(
                    about._errors.cstring(
                        "ERROR: invalid parameter ( nside <> 2**n ).")
                )

        self.parameters.__setitem__(key, temp)
