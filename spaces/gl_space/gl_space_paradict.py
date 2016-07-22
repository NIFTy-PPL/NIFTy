# -*- coding: utf-8 -*-

from nifty.config import about
from nifty.spaces.space import SpaceParadict


class GLSpaceParadict(SpaceParadict):

    def __init__(self, nlat, nlon):
        SpaceParadict.__init__(self, nlat=nlat)
        if nlon is None:
            nlon = -1
        self['nlon'] = nlon

    def __setitem__(self, key, arg):
        if key not in ['nlat', 'nlon']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported GLSpace parameter: " + key))

        if key == 'nlat':
            temp = int(arg)
            if(temp < 1):
                raise ValueError(about._errors.cstring(
                    "ERROR: nlat: nonpositive number."))
            if (temp % 2 != 0):
                raise ValueError(about._errors.cstring(
                    "ERROR: invalid parameter (nlat <> 2n)."))
            try:
                if temp < self['mmax']:
                    about.warnings.cprint(
                        "WARNING: mmax parameter set to lmax.")
                    self['mmax'] = temp
                if (temp != self['mmax']):
                    about.warnings.cprint(
                        "WARNING: unrecommended parameter set (mmax <> lmax).")
            except:
                pass
        elif key == 'nlon':
            temp = int(arg)
            if (temp < 1):
                about.warnings.cprint(
                    "WARNING: nlon parameter set to default.")
                temp = 2 * self['nlat'] - 1
            if(temp != 2 * self['nlat'] - 1):
                about.warnings.cprint(
                    "WARNING: unrecommended parameter set (nlon <> 2*nlat-1).")
        self.parameters.__setitem__(key, temp)
