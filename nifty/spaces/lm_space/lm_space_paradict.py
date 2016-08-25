# -*- coding: utf-8 -*-

#import numpy as np
#from nifty.config import about
#from nifty.spaces.space import SpaceParadict
#
#
#class LMSpaceParadict(SpaceParadict):
#
#    def __init__(self, lmax, mmax):
#        SpaceParadict.__init__(self, lmax=lmax)
#        if mmax is None:
#            mmax = -1
#        self['mmax'] = mmax
#
#    def __setitem__(self, key, arg):
#        if key not in ['lmax', 'mmax']:
#            raise ValueError(about._errors.cstring(
#                "ERROR: Unsupported LMSpace parameter: " + key))
#
#        if key == 'lmax':
#            temp = np.int(arg)
#            if temp < 1:
#                raise ValueError(about._errors.cstring(
#                    "ERROR: lmax: nonpositive number."))
#            # exception lmax == 2 (nside == 1)
#            if (temp % 2 == 0) and (temp > 2):
#                about.warnings.cprint(
#                    "WARNING: unrecommended parameter (lmax <> 2*n+1).")
#            try:
#                if temp < self['mmax']:
#                    about.warnings.cprint(
#                        "WARNING: mmax parameter set to lmax.")
#                    self['mmax'] = temp
#                if (temp != self['mmax']):
#                    about.warnings.cprint(
#                        "WARNING: unrecommended parameter set (mmax <> lmax).")
#            except:
#                pass
#        elif key == 'mmax':
#            temp = int(arg)
#            if (temp < 1) or(temp > self['lmax']):
#                about.warnings.cprint(
#                    "WARNING: mmax parameter set to default.")
#                temp = self['lmax']
#            if(temp != self['lmax']):
#                about.warnings.cprint(
#                    "WARNING: unrecommended parameter set (mmax <> lmax).")
#
#        self.parameters.__setitem__(key, temp)
