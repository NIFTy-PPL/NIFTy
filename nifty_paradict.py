# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:29:30 2015

@author: steininger
"""

import numpy as np
from nifty.config import about


class space_paradict(object):

    def __init__(self, **kwargs):
        self.parameters = {}
        for key in kwargs:
            self[key] = kwargs[key]

    def __iter__(self):
        return self.parameters.__iter__()

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.parameters.__repr__()

    def __setitem__(self, key, arg):
        raise NotImplementedError

    def __getitem__(self, key):
        return self.parameters.__getitem__(key)

    def __hash__(self):
        result_hash = 0
        for (key, item) in self.parameters.items():
            try:
                temp_hash = hash(item)
            except TypeError:
                temp_hash = hash(tuple(item))
            result_hash ^= temp_hash * hash(key)
        return result_hash


class rg_space_paradict(space_paradict):

    def __init__(self, shape, complexity, zerocenter):
        self.ndim = len(np.array(shape).flatten())
        space_paradict.__init__(
            self, shape=shape, complexity=complexity, zerocenter=zerocenter)

    def __setitem__(self, key, arg):
        if key not in ['shape', 'complexity', 'zerocenter']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported rg_space parameter"))

        if key == 'shape':
            temp = np.array(arg, dtype=np.int).flatten()
            if np.any(temp < 0):
                raise ValueError("ERROR: negative number in shape.")
            temp = tuple(temp)
            if len(temp) != self.ndim:
                raise ValueError(about._errors.cstring(
                    "ERROR: Number of dimensions does not match the init " +
                    "value."))
        elif key == 'complexity':
            temp = int(arg)
            if temp not in [0, 1, 2]:
                raise ValueError(about._errors.cstring(
                    "ERROR: Unsupported complexity parameter: " + str(temp)))
        elif key == 'zerocenter':
            temp = np.empty(self.ndim, dtype=bool)
            temp[:] = arg
            temp = tuple(temp)
        self.parameters.__setitem__(key, temp)


class lm_space_paradict(space_paradict):

    def __init__(self, lmax, mmax):
        space_paradict.__init__(self, lmax=lmax)
        if mmax is None:
            mmax = -1
        self['mmax'] = mmax

    def __setitem__(self, key, arg):
        if key not in ['lmax', 'mmax']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported rg_space parameter"))

        if key == 'lmax':
            temp = np.int(arg)
            if temp < 1:
                raise ValueError(about._errors.cstring(
                    "ERROR: lmax: nonpositive number."))
            # exception lmax == 2 (nside == 1)
            if (temp % 2 == 0) and (temp > 2):
                about.warnings.cprint(
                    "WARNING: unrecommended parameter (lmax <> 2*n+1).")
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
        elif key == 'mmax':
            temp = int(arg)
            if (temp < 1) or(temp > self['lmax']):
                about.warnings.cprint(
                    "WARNING: mmax parameter set to default.")
                temp = self['lmax']
            if(temp != self['lmax']):
                about.warnings.cprint(
                    "WARNING: unrecommended parameter set (mmax <> lmax).")

        self.parameters.__setitem__(key, temp)


class gl_space_paradict(space_paradict):

    def __init__(self, nlat, nlon):
        space_paradict.__init__(self, nlat=nlat)
        if nlon is None:
            nlon = -1
        self['nlon'] = nlon

    def __setitem__(self, key, arg):
        if key not in ['nlat', 'nlon']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported rg_space parameter"))

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


class hp_space_paradict(space_paradict):

    def __init__(self, nside):
        space_paradict.__init__(self, nside=nside)

    def __setitem__(self, key, arg):
        if key not in ['nside']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported hp_space parameter"))

        temp = int(arg)
        # if(not hp.isnsideok(nside)):
        if ((temp & (temp - 1)) != 0) or (temp < 2):
            raise ValueError(about._errors.cstring(
                "ERROR: invalid parameter ( nside <> 2**n )."))
        self.parameters.__setitem__(key, temp)


class power_space_paradict(space_paradict):
    def __init__(self, distribution_strategy, log, nbin, binbounds):
        super(power_space_paradict, self).__init___(
                                distribution_strategy=distribution_strategy,
                                log=log,
                                nbin=nbin,
                                binbounds=binbounds)

    def __setitem__(self, key, arg):
        if key not in ['distribution_strategy', 'log', 'nbin', 'binbounds']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported PowerSpace parameter"))

        if key == 'log':
            try:
                temp = bool(arg)
            except(TypeError):
                temp = False

        elif key == 'nbin':
            try:
                temp = int(arg)
            except(TypeError):
                temp = None

        elif key == 'binbounds':
            try:
                temp = tuple(np.array(arg))
            except(TypeError):
                temp = None

        elif key == 'distribution_strategy':
            temp = str(arg)

        self.parameters.__setitem__(key, temp)


class rg_power_space_paradict(power_space_paradict, rg_space_paradict):
    def __init__(self, shape, dgrid, zerocentered, log, nbin, binbounds):
        rg_space_paradict.__init___(shape=shape,
                                    dgrid=dgrid,
                                    zerocentered=zerocentered,
                                    log=log,
                                    nbin=nbin,
                                    binbounds=binbounds)

    def __setitem__(self, key, arg):
        if key not in ['shape', 'zerocentered', 'distribution_strategy',
                       'log', 'nbin', 'binbounds']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported RGPowerSpace parameter"))

        if key in ['distribution_strategy', 'log', 'nbin', 'binbounds']:
            power_space_paradict.__setitem__(key, arg)
        elif key == 'dgrid':
            temp = np.array(arg, dtype=np.float).flatten()
            temp = tuple(temp)
            if len(temp) != self.ndim:
                raise ValueError(about._errors.cstring(
                    "ERROR: Number of dimensions does not match the init "
                    "value."))
            self.parameters.__setitem__(key, temp)
        else:
            rg_space_paradict.__setitem__(key, arg)










