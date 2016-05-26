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

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.parameters.__repr__()

    def __setitem__(self, key, arg):
        if(np.isscalar(arg)):
            arg = np.array([arg], dtype=np.int)
        else:
            arg = np.array(arg, dtype=np.int)

        self.parameters.__setitem__(key, arg)

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


class point_space_paradict(space_paradict):

    def __setitem__(self, key, arg):
        if key is not 'num':
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported point_space parameter"))
        if not np.isscalar(arg):
            raise ValueError(about._errors.cstring(
                "ERROR: 'num' parameter must be scalar. Got: " + str(arg)))
        if abs(arg) != arg:
            raise ValueError(about._errors.cstring(
                "ERROR: 'num' parameter must be positive. Got: " + str(arg)))
        temp = np.int(arg)
        self.parameters.__setitem__(key, temp)


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
            temp = list(temp)
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
            temp = list(temp)
        self.parameters.__setitem__(key, temp)


class nested_space_paradict(space_paradict):

    def __init__(self, ndim):
        self.ndim = np.int(ndim)
        space_paradict.__init__(self)

    def __setitem__(self, key, arg):
        if not isinstance(key, int):
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported point_space parameter"))
        if key >= self.ndim or key < 0:
            raise ValueError(about._errors.cstring(
                "ERROR: Nestindex out of bounds"))
        temp = list(np.array(arg, dtype=int).flatten())
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

###################
#
#
# class _space(object):
#    def __init__(self):
#        self.paradict = space_paradict(default=123)
#        #self.para = [1,2,3]
#
#    @property
#    def para(self):
#        return self.paradict['default']
#        #return self.distributed_val
#
#    @para.setter
#    def para(self, x):
#        self.paradict['default'] = x
#
###################
###################
#
#
#
# class _point_space(object):
#    def __init__(self):
#        self.paradict = point_space_paradict()
#        self.para = [10]
#
#    @property
#    def para(self):
#        temp = np.array([self.paradict['num']], dtype=int)
#        return temp
#        #return self.distributed_val
#
#    @para.setter
#    def para(self, x):
#        self.paradict['num'] = x
#
###################
###################
#
#
# class _rg_space(object):
#    def __init__(self):
#        self.paradict = rg_space_paradict(num=[10,100,200])
#
#    @property
#    def para(self):
#        temp = np.array(self.paradict['num'] + \
#                         [self.paradict['hermitian']] + \
#                         self.paradict['zerocenter'], dtype=int)
#        return temp
#
#
#    @para.setter
#    def para(self, x):
#        self.paradict['num'] = x[:(np.size(x)-1)//2]
#        self.paradict['zerocenter'] = x[(np.size(x)+1)//2:]
#        self.paradict['complexity'] = x[(np.size(x)-1)//2]
#
###################
###################
#
# class _nested_space(object):
#    def __init__(self):
#        self.paradict = nested_space_paradict(ndim=10)
#        for i in range(10):
#            self.paradict[i] = [1+i, 2+i, 3+i]
#
#    @property
#    def para(self):
#        temp = []
#        for i in range(self.paradict.ndim):
#            temp = np.append(temp, self.paradict[i])
#        return temp
#
#    @para.setter
#    def para(self, x):
#        dict_iter = 0
#        x_iter = 0
#        while dict_iter < self.paradict.ndim:
#            temp = x[x_iter:x_iter+len(self.paradict[dict_iter])]
#            self.paradict[dict_iter] = temp
#            x_iter = x_iter+len(self.paradict[dict_iter])
#            dict_iter += 1
#
###################
###################
#
# class _lm_space(object):
#    def __init__(self):
#        self.paradict = lm_space_paradict(lmax = 10)
#
#    @property
#    def para(self):
#        temp = np.array([self.paradict['lmax'],
#                         self.paradict['mmax']], dtype=int)
#        return temp
#
#
#    @para.setter
#    def para(self, x):
#        self.paradict['lmax'] = x[0]
#        self.paradict['mmax'] = x[1]
#
#
###################
###################
#
# class _gl_space(object):
#    def __init__(self):
#        self.paradict = gl_space_paradict(nlat = 10)
#
#    @property
#    def para(self):
#        temp = np.array([self.paradict['nlat'],
#                         self.paradict['nlon']], dtype=int)
#        return temp
#
#
#    @para.setter
#    def para(self, x):
#        self.paradict['nlat'] = x[0]
#        self.paradict['nlon'] = x[1]
#
#
###################
###################
#
#
# class _hp_space(object):
#    def __init__(self):
#        self.paradict = hp_space_paradict(nside=16)
#
#    @property
#    def para(self):
#        temp = np.array([self.paradict['nside']], dtype=int)
#        return temp
#
#
#    @para.setter
#    def para(self, x):
#        self.paradict['nside'] = x[0]
#
#
#
###################
###################
#
#
#
# if __name__ == '__main__':
#    myspace = _space()
#    print myspace.para
#    print myspace.paradict.parameters.items()
#    myspace.para = [4,5,6]
#    print myspace.para
#    print myspace.paradict.parameters.items()
#
#    myspace.paradict.parameters['default'] = [1,4,7]
#    print myspace.para
#    print myspace.paradict.parameters.items()
