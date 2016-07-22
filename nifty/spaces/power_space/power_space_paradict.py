# -*- coding: utf-8 -*-

from nifty.config import about
from nifty.spaces.space import SpaceParadict


class PowerSpaceParadict(SpaceParadict):
    def __init__(self, pindex, kindex, rho, config, harmonic_domain):
        SpaceParadict.__init__(self,
                               pindex=pindex,
                               kindex=kindex,
                               rho=rho,
                               config=config,
                               harmonic_domain=harmonic_domain)

    def __setitem__(self, key, arg):
        if key not in ['pindex', 'kindex', 'rho', 'config', 'harmonic_domain']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported PowerSpace parameter: " + key))

        if key == 'harmonic_domain':
            if not arg.harmonic:
                raise ValueError(about._errors.cstring(
                    "ERROR: harmonic_domain must be harmonic."))
            temp = arg
        else:
            temp = arg

        self.parameters.__setitem__(key, temp)

    def __hash__(self):
        return (hash(frozenset(self.parameters['config'].items())) ^
                (hash(self.parameters['harmonic_domain'])/131))
