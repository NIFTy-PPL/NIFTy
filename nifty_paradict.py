# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:29:30 2015

@author: steininger
"""

import numpy as np
from nifty import *
#from nifty import about

def paradict_getter(space_instance):
    paradict_dictionary = {
        str(space().__class__) : _space_paradict,
        str(point_space((2)).__class__) : _point_space_paradict,           
        str(rg_space((2)).__class__) : _rg_space_paradict,
        str(nested_space([point_space(2), point_space(2)]).__class__) : _nested_space_paradict,
        str(lm_space(1).__class__) : _lm_space_paradict,
        str(gl_space(2).__class__) : _gl_space_paradict,
        str(hp_space(1).__class__) : _hp_space_paradict,
    }
    return paradict_dictionary[str(space_instance.__class__)]()


class _space_paradict(object):
    def __init__(self, **kwargs):
        self.parameters = {}
        for key in kwargs:
            self[key] = kwargs[key]
            
    def __setitem__(self, key, arg):
        self.parameters.__setitem__(key, arg)
    
    def __getitem__(self, key):
        return self.parameters.__getitem__(key)        
      

class _point_space_paradict(_space_paradict):
    def __setitem__(self, key, arg):
        if key is not 'num':
            raise ValueError(about._errors.cstring("ERROR: Unsupported point_space parameter"))
        temp = np.array(arg, dtype=int).flatten()[0]
        self.parameters.__setitem__(key, temp)
    
      
class _rg_space_paradict(_space_paradict):
    def __init__(self, num, hermitian=False, zerocenter=False):
        self.ndim = len(np.array(num).flatten())        
        _space_paradict.__init__(self, num=num, hermitian=hermitian, zerocenter=zerocenter)
        
    def __setitem__(self, key, arg):
        if key not in ['num', 'hermitian', 'zerocenter']:
            raise ValueError(about._errors.cstring("ERROR: Unsupported rg_space parameter"))
        
        if key == 'num':
            temp = list(np.array(arg, dtype=int).flatten())
            if len(temp) != self.ndim:
                raise ValueError(about._errors.cstring("ERROR: Number of dimensions does not match the init value."))
        elif key == 'hermitian':
            temp = int(arg)
        elif key == 'zerocenter':
            temp = np.empty(self.ndim, dtype=bool)
            temp[:] = arg
            temp = list(temp)
            #if len(temp) != self.ndim:
            #    raise ValueError(about._errors.cstring("ERROR: Number of dimensions does not match the init value."))            
        self.parameters.__setitem__(key, temp)
        
class _nested_space_paradict(_space_paradict):
    def __init__(self, ndim):
        self.ndim = np.int(ndim)
        _space_paradict.__init__(self)
    def __setitem__(self, key, arg):
        if not isinstance(key, int):
            raise ValueError(about._errors.cstring("ERROR: Unsupported point_space parameter"))
        if key >= self.ndim or key < 0:
            raise ValueError(about._errors.cstring("ERROR: Nestindex out of bounds"))
        temp = list(np.array(arg, dtype=int).flatten())   
        self.parameters.__setitem__(key, temp)
    
    
class _lm_space_paradict(_space_paradict):
    def __init__(self, lmax, mmax=None):
        _space_paradict.__init__(self, lmax=lmax)
        if mmax == None:        
           mmax = -1 
        self['mmax'] = mmax      
        
    def __setitem__(self, key, arg):
        if key not in ['lmax', 'mmax']:
            raise ValueError(about._errors.cstring("ERROR: Unsupported rg_space parameter"))

        if key == 'lmax':
            temp = int(arg)
            if(temp<1):
                raise ValueError(about._errors.cstring("ERROR: lmax: nonpositive number."))
            if (temp%2 == 0) and (temp > 2): ## exception lmax == 2 (nside == 1)
                about.warnings.cprint("WARNING: unrecommended parameter (lmax <> 2*n+1).")
            try:
                if temp < self['mmax']:
                    about.warnings.cprint("WARNING: mmax parameter set to lmax.")
                    self['mmax'] = temp
                if (temp != self['mmax']):
                    about.warnings.cprint("WARNING: unrecommended parameter set (mmax <> lmax).")
            except:
                pass
        elif key == 'mmax':
            temp = int(arg)            
            if (temp < 1) or(temp > self['lmax']):
                about.warnings.cprint("WARNING: mmax parameter set to default.")
                temp = self['lmax']            
            if(temp != self['lmax']):
                about.warnings.cprint("WARNING: unrecommended parameter set (mmax <> lmax).")
          
        self.parameters.__setitem__(key, temp)

class _gl_space_paradict(_space_paradict):
    def __init__(self, nlat, nlon=None):
        _space_paradict.__init__(self, nlat=nlat)
        if nlon == None:        
           nlon = -1
        self['nlon'] = nlon
        
    def __setitem__(self, key, arg):
        if key not in ['nlat', 'nlon']:
            raise ValueError(about._errors.cstring("ERROR: Unsupported rg_space parameter"))

        if key == 'nlat':
            temp = int(arg)
            if(temp<1):
                raise ValueError(about._errors.cstring("ERROR: nlat: nonpositive number."))
            if (temp%2 != 0):
                raise ValueError(about._errors.cstring("ERROR: invalid parameter (nlat <> 2n)."))
            try:
                if temp < self['mmax']:
                    about.warnings.cprint("WARNING: mmax parameter set to lmax.")
                    self['mmax'] = temp
                if (temp != self['mmax']):
                    about.warnings.cprint("WARNING: unrecommended parameter set (mmax <> lmax).")
            except:
                pass
        elif key == 'nlon':
            temp = int(arg)            
            if (temp < 1):
                about.warnings.cprint("WARNING: nlon parameter set to default.")
                temp = 2*self['nlat']-1
            if(temp != 2*self['nlat']-1):
                about.warnings.cprint("WARNING: unrecommended parameter set (nlon <> 2*nlat-1).")
        self.parameters.__setitem__(key, temp)


        
class _hp_space_paradict(_space_paradict):
    def __init__(self, nside):
        _space_paradict.__init__(self, nside=nside)
    def __setitem__(self, key, arg):
        if key not in ['nside']:
            raise ValueError(about._errors.cstring("ERROR: Unsupported hp_space parameter"))        
        
        temp = int(arg)
        #if(not hp.isnsideok(nside)):
        if ((temp & (temp-1)) != 0) or (temp < 2):
            raise ValueError(about._errors.cstring("ERROR: invalid parameter ( nside <> 2**n )."))
        self.parameters.__setitem__(key, temp)        

##################


class _space(object):
    def __init__(self):
        self.paradict = _space_paradict(default=123)        
        #self.para = [1,2,3]
        
    @property
    def para(self):
        return self.paradict['default']
        #return self.distributed_val
    
    @para.setter
    def para(self, x):
        self.paradict['default'] = x

##################
##################
        

    
class _point_space(object):
    def __init__(self):
        self.paradict = _point_space_paradict()        
        self.para = [10]
        
    @property
    def para(self):
        temp = np.array([self.paradict['num']], dtype=int)
        return temp
        #return self.distributed_val
    
    @para.setter
    def para(self, x):
        self.paradict['num'] = x

##################
##################
        

class _rg_space(object):
    def __init__(self):
        self.paradict = _rg_space_paradict(num=[10,100,200])        
        
    @property
    def para(self):
        temp = np.array(self.paradict['num'] + \
                         [self.paradict['hermitian']] + \
                         self.paradict['zerocenter'], dtype=int)
        return temp
        
    
    @para.setter
    def para(self, x):
        self.paradict['num'] = x[:(np.size(x)-1)//2]
        self.paradict['zerocenter'] = x[(np.size(x)+1)//2:]
        self.paradict['hermitian'] = x[(np.size(x)-1)//2]
        
##################
##################
        
class _nested_space(object):
    def __init__(self):
        self.paradict = _nested_space_paradict(ndim=10)
        for i in range(10):
            self.paradict[i] = [1+i, 2+i, 3+i]
        
    @property
    def para(self):
        temp = []
        for i in range(self.paradict.ndim):
            temp = np.append(temp, self.paradict[i])
        return temp
        
    @para.setter
    def para(self, x):
        dict_iter = 0
        x_iter = 0
        while dict_iter < self.paradict.ndim:
            temp = x[x_iter:x_iter+len(self.paradict[dict_iter])]
            self.paradict[dict_iter] = temp
            x_iter = x_iter+len(self.paradict[dict_iter])
            dict_iter += 1
                
##################
##################

class _lm_space(object):
    def __init__(self):
        self.paradict = _lm_space_paradict(lmax = 10)        
        
    @property
    def para(self):
        temp = np.array([self.paradict['lmax'], 
                         self.paradict['mmax']], dtype=int)
        return temp
        
    
    @para.setter
    def para(self, x):
        self.paradict['lmax'] = x[0]
        self.paradict['mmax'] = x[1]
    
        
##################
##################

class _gl_space(object):
    def __init__(self):
        self.paradict = _gl_space_paradict(nlat = 10)        
        
    @property
    def para(self):
        temp = np.array([self.paradict['nlat'], 
                         self.paradict['nlon']], dtype=int)
        return temp
        
    
    @para.setter
    def para(self, x):
        self.paradict['nlat'] = x[0]
        self.paradict['nlon'] = x[1]
    
        
##################
##################


class _hp_space(object):
    def __init__(self):
        self.paradict = _hp_space_paradict(nside=16)        
        
    @property
    def para(self):
        temp = np.array([self.paradict['nside']], dtype=int)
        return temp
        
    
    @para.setter
    def para(self, x):
        self.paradict['nside'] = x[0]
        
    
        
##################
##################


        
if __name__ == '__main__':
    myspace = _space()
    print myspace.para
    print myspace.paradict.parameters.items()
    myspace.para = [4,5,6]
    print myspace.para
    print myspace.paradict.parameters.items()
    
    myspace.paradict.parameters['default'] = [1,4,7] 
    print myspace.para
    print myspace.paradict.parameters.items()
    