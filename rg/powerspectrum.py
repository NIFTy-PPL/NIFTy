## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
import numpy as np
from mpi4py import MPI

from nifty.nifty_core import about
from nifty.nifty_mpi_data import distributed_data_object

class power_indices(object):
    def __init__(self, shape, dgrid, zerocentered=False, log=False, nbin=None, 
                 binbounds=None, comm=MPI.COMM_WORLD):
        """
            Returns an instance of the power_indices class. Given the shape and
            the density of a underlying rectangular grid it provides the user
            with the pindex, kindex, rho and pundex. The indices are bined 
            according to the supplied parameter scheme. If wanted, computed 
            results are stored for future reuse.
    
            Parameters
            ----------
            shape : tuple, list, ndarray
                Array-like object which specifies the shape of the underlying 
                rectangular grid
            dgrid : tuple, list, ndarray
                Array-like object which specifies the step-width of the 
                underlying grid
            zerocentered : boolean, tuple/list/ndarray of boolean *optional*
                Specifies which dimensions are zerocentered. (default:False)
            log : bool *optional*
                Flag specifying if the binning of the default indices is 
                performed on logarithmic scale.
            nbin : integer *optional*
                Number of used bins for the binning of the default indices.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins of the default 
                indices.
        """ 
        ## Basic inits and consistency checks
        self.comm = comm
        self.shape = np.array(shape).astype(int)
        self.dgrid = np.array(dgrid)
        if self.shape.shape != self.dgrid.shape:
            raise ValueError(about._errors.cstring("ERROR: The supplied shape\
                and dgrid have not the same dimensionality"))         
        self.zerocentered = self.__cast_zerocentered__(zerocentered)
        
        ## Initialize the dictonary which stores all individual index-dicts
        self.global_dict={}
    
        ## Calculate the default dictonory according to the kwargs and set it 
        ## as default
        self.get_index_dict(log=log, nbin=nbin, binbounds=binbounds, 
                            store=True)
        self.set_default(log=log, nbin=nbin, binbounds=binbounds)
        
    ## Redirect the direct calls approaching a power_index instance to the 
    ## default_indices dict
    def __getitem__(self, x):
        return self.default_indices.get(x)
    def __getattr__(self, x):
        return self.default_indices.__getattribute__(x)
        
    def __cast_zerocentered__(self, zerocentered=False):
        """        
            internal helper function which brings the zerocentered input in 
            the form of a boolean-tuple
        """
        zc = np.array(zerocentered).astype(bool)
        if zc.shape == self.shape.shape:
            return tuple(zc)
        else:
            temp = np.empty(shape=self.shape.shape, dtype=bool)
            temp[:] = zc
            return tuple(temp)
        
    def __cast_config__(self, *args, **kwargs):
        """
            internal helper function which casts the various combinations of 
            possible parameters into a properly defaulted dictionary
        """
        temp_config_dict = kwargs.get('config_dict', None)        
        if temp_config_dict != None:
            return self.__cast_config_helper__(**temp_config_dict)
        else:
            temp_log = kwargs.get("log", None)
            temp_nbin = kwargs.get("nbin", None)
            temp_binbounds = kwargs.get("binbounds", None)            
            
            return self.__cast_config_helper__(log=temp_log, 
                                               nbin=temp_nbin, 
                                               binbounds=temp_binbounds)
    
    def __cast_config_helper__(self, log, nbin, binbounds):
        """
            internal helper function which sets the defaults for the 
            __cast_config__ function
        """
        
        try:
            temp_log = bool(log)
        except(TypeError):
            temp_log = False
        
        try:
            temp_nbin = int(nbin)
        except(TypeError):
            temp_nbin = None
        
        try:
            temp_binbounds = tuple(np.array(binbounds))
        except(TypeError):
            temp_binbounds = None
        
        temp_dict = {"log":temp_log, 
                     "nbin":temp_nbin, 
                     "binbounds":temp_binbounds}
        return temp_dict
    
    def __freeze_config__(self, config_dict):
        """
            a helper function which forms a hashable identifying object from 
            a config dictionary which can be used as key of a dict
        """        
        return frozenset(config_dict.items())
        
    def set_default(self, *args, **kwargs):
        """
            Sets the index-set which is specified by the parameters as the 
            default for the power_index instance. 
            
            Parameters
            ----------
            log : bool
                Flag specifying if the binning is performed on logarithmic 
                scale.
            nbin : integer
                Number of used bins.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins.
    
            Returns
            -------
            None    
        """ 
        ## This shortcut relies on the fact, that get_index_dict returns a
        ## reference on the default dict and not a copy!!
        self.default_indices = self.get_index_dict(*args, **kwargs)         
        
    
    def get_index_dict(self, *args, **kwargs):
        """
            Returns a dictionary containing the pindex, kindex, rho and pundex
            binned according to the supplied parameter scheme and a 
            configuration dict containing this scheme.
    
            Parameters
            ----------
            store : bool
                Flag specifying if  the calculated index dictionary should be 
                stored in the global_dict for future use.
            log : bool
                Flag specifying if the binning is performed on logarithmic 
                scale.
            nbin : integer
                Number of used bins.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins.
    
            Returns
            -------
            index_dict : dict
                Contains the keys: 'config', 'pindex', 'kindex', 'rho' and 
                'pundex'    
        """        
        ## Cast the input arguments        
        temp_config_dict = self.__cast_config__(*args, **kwargs)
        ## Compute a hashable identifier from the config which will be used 
        ## as dict key
        temp_key = self.__freeze_config__(temp_config_dict)
        ## Check if the result should be stored for future use.
        storeQ = kwargs.get("store", True)
        ## Try to find the requested index dict in the global_dict
        try:
            return self.global_dict[temp_key]
        except(KeyError):
            ## If it is not found, calculate it.
            temp_index_dict = self.__compute_index_dict__(temp_config_dict)
            ## Store it, if required
            if storeQ == True:
                self.global_dict[temp_key] = temp_index_dict
                ## Important: If the result is stored, return a reference to 
                ## the dictionary entry, not anly a plain copy. Otherwise, 
                ## set_default breaks!
                return self.global_dict[temp_key]
            else:
                ## Return the plain result.
                return temp_index_dict
        
    
    def compute_nkdict(self):
        """
            Calculates an n-dimensional array with its entries being the 
            lengths of the k-vectors from the zero point of the grid.    
            
            Parameters
            ----------
            None : All information is taken from the parent object.
    
            Returns
            -------
            nkdict : distributed_data_object
        """
        
        
        ##if(fourier):
        ##   dk = dgrid
        ##else:
        ##    dk = np.array([1/dgrid[i]/axes[i] for i in range(len(axes))])
        
        dk = self.dgrid
        shape = self.shape
        
        ## prepare the distributed_data_object        
        nkdict = distributed_data_object(global_shape=shape, 
                                         dtype=np.float128, 
                                         distribution_strategy="fftw")
        ## get the node's individual slice of the first dimension 
        slice_of_first_dimension = slice(*nkdict.distributor.local_slice[0:2])
        
        inds = []
        for a in shape:
            inds += [slice(0,a)]
        
        cords = np.ogrid[inds]
    
        dists = ((cords[0]-shape[0]//2)*dk[0])**2
        ## apply zerocenteredQ shift
        if self.zerocentered[0] == False:
            dists = np.fft.fftshift(dists)
        ## only save the individual slice
        dists = dists[slice_of_first_dimension]
        for ii in range(1,len(shape)):
            temp = ((cords[ii]-shape[ii]//2)*dk[ii])**2
            if self.zerocentered[ii] == False:
                temp = np.fft.fftshift(temp)
            dists = dists + temp
        dists = np.sqrt(dists)
        nkdict.set_local_data(dists)
        return nkdict

    def __compute_indices__(self, nkdict):
        """
        Internal helper function which computes pindex, kindex, rho and pundex
        from a given nkdict
        """
        ##########
        # kindex #        
        ##########
        ## compute the local kindex array        
        local_kindex = np.unique(nkdict.get_local_data())
        ## unify the local_kindex arrays
        global_kindex = self.comm.allgather(local_kindex)
        ## flatten the gathered lists        
        global_kindex = np.hstack(global_kindex)
        ## remove duplicates        
        global_kindex = np.unique(global_kindex)
        
        ##########
        # pindex #        
        ##########
        ## compute the local pindex slice on basis of the local nkdict data
        local_pindex = np.searchsorted(global_kindex, nkdict.get_local_data())
        ## prepare the distributed_data_object
        global_pindex = distributed_data_object(global_shape=nkdict.shape, 
                                                dtype=local_pindex.dtype.type,
                                                distribution_strategy='fftw')  
        ## store the local pindex data in the global_pindex d2o
        global_pindex.set_local_data(local_pindex)
        
        #######
        # rho #        
        #######
        ## Prepare the local pindex data in order to conut the degeneracy 
        ## factors
        temp = local_pindex.flatten()
        ## Remember: np.array.sort is an inplace function        
        temp.sort()        
        ## In local_count we save how many of the indvidual values in 
        ## local_value occured. Therefore we use np.unique to calculate the 
        ## offset...
        local_value, local_count = np.unique(temp, return_index=True)
        ## ...and then build the offset differences
        if local_count.shape != (0,):
            local_count = np.append(local_count[1:]-local_count[:-1],
                                    [temp.shape[0]-local_count[-1]])
        ## Prepare the global rho array, and store the individual counts in it
        ## rho has the same length as the kindex array
        local_rho = np.zeros(shape=global_kindex.shape, dtype=np.int)
        global_rho = np.empty_like(local_rho)
        ## Store the individual counts
        local_rho[local_value] = local_count
        ## Use Allreduce to spread the information
        self.comm.Allreduce(local_rho , global_rho, op=MPI.SUM)
        ##########
        # pundex #        
        ##########  
        global_pundex = self.__compute_pundex__(global_pindex,
                                            global_kindex)

        return global_pindex, global_kindex, global_rho, global_pundex

    def __compute_pundex__(self, global_pindex, global_kindex):
        """
        Internal helper function which computes the pundex array from a
        pindex and a kindex array. This function is separated from the 
        __compute_indices__ function as it is needed in __bin_power_indices__,
        too.
        """
        ##########
        # pundex #        
        ##########
        ## Prepare the local data
        local_pindex = global_pindex.get_local_data()
        ## Compute the local pundices for the local pindices
        (temp_uniqued_pindex, local_temp_pundex) = np.unique(local_pindex, 
                                                        return_index=True)
        ## Shift the local pundices by the nodes' local_dim_offset
        local_temp_pundex += global_pindex.distributor.local_dim_offset
        
        ## Prepare the pundex arrays used for the Allreduce operation        
        ## pundex has the same length as the kindex array
        local_pundex = np.zeros(shape=global_kindex.shape, dtype=np.int)
        ## Set the default value higher than the maximal possible pundex value
        ## so that MPI.MIN can sort out the default
        local_pundex += np.prod(global_pindex.shape) + 1
        ## Set the default value higher than the length 
        global_pundex = np.empty_like(local_pundex)
        ## Store the individual pundices in the local_pundex array
        local_pundex[temp_uniqued_pindex] = local_temp_pundex
        ## Use Allreduce to find the first occurences/smallest pundices 
        self.comm.Allreduce(local_pundex, global_pundex, op=MPI.MIN)
        return global_pundex
        
    def __compute_index_dict__(self, config_dict):
        """
            Internal helper function which takes a config_dict, asks for the 
            pindex/kindex/rho/pundex set, and bins them according to the config
        """        
        ## if no binning is requested, compute the indices, build the dict, 
        ## and return it straight.        
        if config_dict["log"]==False and config_dict["nbin"]==None and \
          config_dict["binbounds"]==None:
            temp_nkdict = self.compute_nkdict()
            (temp_pindex, temp_kindex, temp_rho, temp_pundex) = self.__compute_indices__(temp_nkdict)
            
        ## if binning is required, make a recursive call to get the unbinned
        ## indices, bin them, compute the pundex and then return everything.
        else:
            temp_unbinned_indices = self.get_index_dict(store=False)
            (temp_pindex, temp_kindex, temp_rho, temp_pundex) = \
                self.__bin_power_indices__(temp_unbinned_indices, **config_dict)
                        
        temp_index_dict = {"config": config_dict, 
                               "pindex": temp_pindex,
                               "kindex": temp_kindex,
                               "rho": temp_rho,
                               "pundex": temp_pundex}
        return temp_index_dict

    def __bin_power_indices__(self, index_dict, **kwargs):
        """
            Returns the binned power indices associated with the Fourier grid.
    
            Parameters
            ----------
            pindex : distributed_data_object
                Index of the Fourier grid points in a distributed_data_object.
            kindex : ndarray
                Array of all k-vector lengths.
            rho : ndarray
                Degeneracy factor of the individual k-vectors.
            log : bool
                Flag specifying if the binning is performed on logarithmic 
                scale.
            nbin : integer
                Number of used bins.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins.
    
            Returns
            -------
            pindex : distributed_data_object
            kindex, rho, pundex : ndarrays
                The (re)binned power indices.
    
        """
        ## Cast the given config
        temp_config_dict = self.__cast_config__(**kwargs)
        log = temp_config_dict['log']
        nbin = temp_config_dict['nbin']
        binbounds = temp_config_dict['binbounds']
        
        ## Extract the necessary indices from the supplied index dict        
        pindex = index_dict["pindex"]
        kindex = index_dict["kindex"]
        rho = index_dict["rho"]
        
        ## boundaries
        if(binbounds is not None):
            binbounds = np.sort(binbounds)
        ## equal binning
        else:
            if(log is None):
                log = False
            if(log):
                k = np.r_[0,np.log(kindex[1:])]
            else:
                k = kindex
            dk = np.max(k[2:]-k[1:-1]) ## minimal dk
            if(nbin is None):
                nbin = int((k[-1]-0.5*(k[2]+k[1]))/dk-0.5) ## maximal nbin
            else:
                nbin = min(int(nbin),int((k[-1]-0.5*(k[2]+k[1]))/dk+2.5))
                dk = (k[-1]-0.5*(k[2]+k[1]))/(nbin-2.5)
            binbounds = np.r_[0.5*(3*k[1]-k[2]),0.5*(k[1]+k[2])+dk*np.arange(nbin-2)]
            if(log):
                binbounds = np.exp(binbounds)
        ## reordering
        reorder = np.searchsorted(binbounds,kindex)
        rho_ = np.zeros(len(binbounds)+1,dtype=rho.dtype)
        kindex_ = np.empty(len(binbounds)+1,dtype=kindex.dtype)    
        for ii in range(len(reorder)):
            if(rho_[reorder[ii]]==0):
                kindex_[reorder[ii]] = kindex[ii]
                rho_[reorder[ii]] += rho[ii]
            else:
                kindex_[reorder[ii]] = (kindex_[reorder[ii]]*rho_[reorder[ii]]+kindex[ii]*rho[ii])/(rho_[reorder[ii]]+rho[ii])
                rho_[reorder[ii]] += rho[ii]
        
        pindex_ = pindex.copy_empty()
        pindex_.set_local_data(reorder[pindex.get_local_data()])
        
        pundex_ = self.__compute_pundex__(pindex_, kindex_)     
        return pindex_, kindex_, rho_, pundex_



        
        

from mpi4py import MPI
#import time
if __name__ == '__main__':    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    p = power_indices((4,4),(1,1), zerocentered=(True,True), nbin = 4)
    """
    obj = p.default_indices['nkdict']
    for i in np.arange(size):
        if rank==i:
            print obj.data
        time.sleep(0.1)
    temp = obj.get_full_data()
    if rank == 0:
        print temp 
    """
    


def draw_vector_nd(axes,dgrid,ps,symtype=0,fourier=False,zerocentered=False,kpack=None):

    """
        Draws a n-dimensional field on a regular grid from a given power
        spectrum. The grid parameters need to be specified, together with a
        couple of global options explained below. The dimensionality of the
        field is determined automatically.

        Parameters
        ----------
        axes : ndarray
            An array with the length of each axis.

        dgrid : ndarray
            An array with the pixel length of each axis.

        ps : ndarray
            The power spectrum as a function of Fourier modes.

        symtype : int {0,1,2} : *optional*
            Whether the output should be real valued (0), complex-hermitian (1)
            or complex without symmetry (2). (default=0)

        fourier : bool : *optional*
            Whether the output should be in Fourier space or not
            (default=False).

        zerocentered : bool : *optional*
            Whether the output array should be zerocentered, i.e. starting with
            negative Fourier modes going over the zero mode to positive modes,
            or not zerocentered, where zero, positive and negative modes are
            simpy ordered consecutively.

        Returns
        -------
        field : ndarray
            The drawn random field.

    """
    if(kpack is None):
        kdict = np.fft.fftshift(nkdict_fast(axes,dgrid,fourier))
        klength = nklength(kdict)
    else:
        kdict = kpack[1][np.fft.ifftshift(kpack[0],axes=shiftaxes(zerocentered,st_to_zero_mode=False))]
        klength = kpack[1]

    #output is in position space
    if(not fourier):

        #output is real-valued
        if(symtype==0):
            vector = drawherm(klength,kdict,ps)
            if(np.any(zerocentered==True)):
                return np.real(np.fft.fftshift(np.fft.ifftn(vector),axes=shiftaxes(zerocentered)))
            else:
                return np.real(np.fft.ifftn(vector))

        #output is complex with hermitian symmetry
        elif(symtype==1):
            vector = drawwild(klength,kdict,ps,real_corr=2)
            if(np.any(zerocentered==True)):
                return np.fft.fftshift(np.fft.ifftn(np.real(vector)),axes=shiftaxes(zerocentered))
            else:
                return np.fft.ifftn(np.real(vector))

        #output is complex without symmetry
        else:
            vector = drawwild(klength,kdict,ps)
            if(np.any(zerocentered==True)):
                return np.fft.fftshift(np.fft.ifftn(vector),axes=shiftaxes(zerocentered))
            else:
                return np.fft.ifftn(vector)

    #output is in fourier space
    else:

        #output is real-valued
        if(symtype==0):
            vector = drawwild(klength,kdict,ps,real_corr=2)
            if np.any(zerocentered == True):
                return np.real(np.fft.fftshift(vector,axes=shiftaxes(zerocentered)))
            else:
                return np.real(vector)

        #output is complex with hermitian symmetry
        elif(symtype==1):
            vector = drawherm(klength,kdict,ps)
            if(np.any(zerocentered==True)):
                return np.fft.fftshift(vector,axes=shiftaxes(zerocentered))
            else:
                return vector

        #output is complex without symmetry
        else:
            vector = drawwild(klength,kdict,ps)
            if(np.any(zerocentered==True)):
                return np.fft.fftshift(vector,axes=shiftaxes(zerocentered))
            else:
                return vector


#def calc_ps(field,axes,dgrid,zerocentered=False,fourier=False):
#
#    """
#        Calculates the power spectrum of a given field assuming that the field
#        is statistically homogenous and isotropic.
#
#        Parameters
#        ----------
#        field : ndarray
#            The input field from which the power spectrum should be determined.
#
#        axes : ndarray
#            An array with the length of each axis.
#
#        dgrid : ndarray
#            An array with the pixel length of each axis.
#
#        zerocentered : bool : *optional*
#            Whether the output array should be zerocentered, i.e. starting with
#            negative Fourier modes going over the zero mode to positive modes,
#            or not zerocentered, where zero, positive and negative modes are
#            simpy ordered consecutively.
#
#        fourier : bool : *optional*
#            Whether the output should be in Fourier space or not
#            (default=False).
#
#    """
#
#    ## field absolutes
#    if(not fourier):
#        foufield = np.fft.fftshift(np.fft.fftn(field))
#    elif(np.any(zerocentered==False)):
#        foufield = np.fft.fftshift(field, axes=shiftaxes(zerocentered,st_to_zero_mode=True))
#    else:
#        foufield = field
#    fieldabs = np.abs(foufield)**2
#
#    kdict = nkdict_fast(axes,dgrid,fourier)
#    klength = nklength(kdict)
#
#    ## power spectrum
#    ps = np.zeros(klength.size)
#    rho = np.zeros(klength.size)
#    for ii in np.ndindex(kdict.shape):
#        position = np.searchsorted(klength,kdict[ii])
#        rho[position] += 1
#        ps[position] += fieldabs[ii]
#    ps = np.divide(ps,rho)
#    return ps

def calc_ps_fast(field,axes,dgrid,zerocentered=False,fourier=False,pindex=None,kindex=None,rho=None):

    """
        Calculates the power spectrum of a given field faster assuming that the
        field is statistically homogenous and isotropic.

        Parameters
        ----------
        field : ndarray
            The input field from which the power spectrum should be determined.

        axes : ndarray
            An array with the length of each axis.

        dgrid : ndarray
            An array with the pixel length of each axis.

        zerocentered : bool : *optional*
            Whether the output array should be zerocentered, i.e. starting with
            negative Fourier modes going over the zero mode to positive modes,
            or not zerocentered, where zero, positive and negative modes are
            simpy ordered consecutively.

        fourier : bool : *optional*
            Whether the output should be in Fourier space or not
            (default=False).

        pindex : ndarray
            Index of the Fourier grid points in a numpy.ndarray ordered
            following the zerocentered flag (default=None).

        kindex : ndarray
            Array of all k-vector lengths (default=None).

        rho : ndarray
            Degeneracy of the Fourier grid, indicating how many k-vectors in
            Fourier space have the same length (default=None).

    """
    ## field absolutes
    if(not fourier):
        foufield = np.fft.fftshift(np.fft.fftn(field))
    elif(np.any(zerocentered==False)):
        foufield = np.fft.fftshift(field, axes=shiftaxes(zerocentered,st_to_zero_mode=True))
    else:
        foufield = field
    fieldabs = np.abs(foufield)**2

    if(rho is None):
        if(pindex is None):
            ## kdict
            kdict = nkdict_fast(axes,dgrid,fourier)
            ## klength
            if(kindex is None):
                klength = nklength(kdict)
            else:
                klength = kindex
            ## power spectrum
            ps = np.zeros(klength.size)
            rho = np.zeros(klength.size)
            for ii in np.ndindex(kdict.shape):
                position = np.searchsorted(klength,kdict[ii])
                ps[position] += fieldabs[ii]
                rho[position] += 1
        else:
            ## zerocenter pindex
            if(np.any(zerocentered==False)):
                pindex = np.fft.fftshift(pindex, axes=shiftaxes(zerocentered,st_to_zero_mode=True))
            ## power spectrum
            ps = np.zeros(np.max(pindex)+1)
            rho = np.zeros(ps.size)
            for ii in np.ndindex(pindex.shape):
                ps[pindex[ii]] += fieldabs[ii]
                rho[pindex[ii]] += 1
    elif(pindex is None):
        ## kdict
        kdict = nkdict_fast(axes,dgrid,fourier)
        ## klength
        if(kindex is None):
            klength = nklength(kdict)
        else:
            klength = kindex
        ## power spectrum
        ps = np.zeros(klength.size)
        for ii in np.ndindex(kdict.shape):
            position = np.searchsorted(klength,kdict[ii])
            ps[position] += fieldabs[ii]
    else:
        ## zerocenter pindex
        if(np.any(zerocentered==False)):
            pindex = np.fft.fftshift(pindex, axes=shiftaxes(zerocentered,st_to_zero_mode=True))
        ## power spectrum
        ps = np.zeros(rho.size)
        for ii in np.ndindex(pindex.shape):
            ps[pindex[ii]] += fieldabs[ii]

    ps = np.divide(ps,rho)
    return ps


def get_power_index(axes,dgrid,zerocentered,irred=False,fourier=True):

    """
        Returns the index of the Fourier grid points in a numpy
        array, ordered following the zerocentered flag.

        Parameters
        ----------
        axes : ndarray
            An array with the length of each axis.

        dgrid : ndarray
            An array with the pixel length of each axis.

        zerocentered : bool
            Whether the output array should be zerocentered, i.e. starting with
            negative Fourier modes going over the zero mode to positive modes,
            or not zerocentered, where zero, positive and negative modes are
            simpy ordered consecutively.

        irred : bool : *optional*
            If True, the function returns an array of all k-vector lengths and
            their degeneracy factors. If False, just the power index array is
            returned.

        fourier : bool : *optional*
            Whether the output should be in Fourier space or not
            (default=False).

        Returns
        -------
            index or {klength, rho} : scalar or list
                Returns either an array of all k-vector lengths and
                their degeneracy factors or just the power index array
                depending on the flag irred.

    """

    ## kdict, klength
    if(np.any(zerocentered==False)):
        kdict = np.fft.fftshift(nkdict_fast(axes,dgrid,fourier),axes=shiftaxes(zerocentered,st_to_zero_mode=True))
    else:
        kdict = nkdict_fast(axes,dgrid,fourier)
    klength = nklength(kdict)
    ## output
    if(irred):
        rho = np.zeros(klength.shape,dtype=np.int)
        for ii in np.ndindex(kdict.shape):
            rho[np.searchsorted(klength,kdict[ii])] += 1
        return klength,rho
    else:
        ind = np.empty(axes,dtype=np.int)
        for ii in np.ndindex(kdict.shape):
            ind[ii] = np.searchsorted(klength,kdict[ii])
        return ind


def get_power_indices(axes,dgrid,zerocentered,fourier=True):
    """
        Returns the index of the Fourier grid points in a numpy
        array, ordered following the zerocentered flag.

        Parameters
        ----------
        axes : ndarray
            An array with the length of each axis.

        dgrid : ndarray
            An array with the pixel length of each axis.

        zerocentered : bool
            Whether the output array should be zerocentered, i.e. starting with
            negative Fourier modes going over the zero mode to positive modes,
            or not zerocentered, where zero, positive and negative modes are
            simpy ordered consecutively.

        irred : bool : *optional*
            If True, the function returns an array of all k-vector lengths and
            their degeneracy factors. If False, just the power index array is
            returned.

        fourier : bool : *optional*
            Whether the output should be in Fourier space or not
            (default=False).

        Returns
        -------
        index, klength, rho : ndarrays
            Returns the power index array, an array of all k-vector lengths and
            their degeneracy factors.

    """

    ## kdict, klength
    if(np.any(zerocentered==False)):
        kdict = np.fft.fftshift(nkdict_fast(axes,dgrid,fourier),axes=shiftaxes(zerocentered,st_to_zero_mode=True))
    else:
        kdict = nkdict_fast(axes,dgrid,fourier)
    klength = nklength(kdict)
    ## output
    ind = np.empty(axes,dtype=np.int)
    rho = np.zeros(klength.shape,dtype=np.int)
    for ii in np.ndindex(kdict.shape):
        ind[ii] = np.searchsorted(klength,kdict[ii])
        rho[ind[ii]] += 1
    return ind,klength,rho


def get_power_indices2(axes,dgrid,zerocentered,fourier=True):
    """
        Returns the index of the Fourier grid points in a numpy
        array, ordered following the zerocentered flag.

        Parameters
        ----------
        axes : ndarray
            An array with the length of each axis.

        dgrid : ndarray
            An array with the pixel length of each axis.

        zerocentered : bool
            Whether the output array should be zerocentered, i.e. starting with
            negative Fourier modes going over the zero mode to positive modes,
            or not zerocentered, where zero, positive and negative modes are
            simpy ordered consecutively.

        irred : bool : *optional*
            If True, the function returns an array of all k-vector lengths and
            their degeneracy factors. If False, just the power index array is
            returned.

        fourier : bool : *optional*
            Whether the output should be in Fourier space or not
            (default=False).

        Returns
        -------
        index, klength, rho : ndarrays
            Returns the power index array, an array of all k-vector lengths and
            their degeneracy factors.

    """

    ## kdict, klength
    if(np.any(zerocentered==False)):
        kdict = np.fft.fftshift(nkdict_fast2(axes,dgrid,fourier),axes=shiftaxes(zerocentered,st_to_zero_mode=True))
    else:
        kdict = nkdict_fast2(axes,dgrid,fourier)

    klength,rho,ind = nkdict_to_indices(kdict)

    return ind,klength,rho

def nkdict_to_indices(kdict):

    kindex,pindex = np.unique(kdict,return_inverse=True)
    pindex = pindex.reshape(kdict.shape)

    rho = pindex.flatten()
    rho.sort()
    rho = np.unique(rho,return_index=True,return_inverse=False)[1]
    rho = np.append(rho[1:]-rho[:-1],[np.prod(pindex.shape)-rho[-1]])

    return kindex,rho,pindex



def bin_power_indices(pindex,kindex,rho,log=False,nbin=None,binbounds=None):
    """
        Returns the (re)binned power indices associated with the Fourier grid.

        Parameters
        ----------
        pindex : ndarray
            Index of the Fourier grid points in a numpy.ndarray ordered
            following the zerocentered flag (default=None).
        kindex : ndarray
            Array of all k-vector lengths (default=None).
        rho : ndarray
            Degeneracy of the Fourier grid, indicating how many k-vectors in
            Fourier space have the same length (default=None).
        log : bool
            Flag specifying if the binning is performed on logarithmic scale
            (default: False).
        nbin : integer
            Number of used bins (default: None).
        binbounds : {list, array}
            Array-like inner boundaries of the used bins (default: None).

        Returns
        -------
        pindex, kindex, rho : ndarrays
            The (re)binned power indices.

    """
    ## boundaries
    if(binbounds is not None):
        binbounds = np.sort(binbounds)
    ## equal binning
    else:
        if(log is None):
            log = False
        if(log):
            k = np.r_[0,np.log(kindex[1:])]
        else:
            k = kindex
        dk = np.max(k[2:]-k[1:-1]) ## minimal dk
        if(nbin is None):
            nbin = int((k[-1]-0.5*(k[2]+k[1]))/dk-0.5) ## maximal nbin
        else:
            nbin = min(int(nbin),int((k[-1]-0.5*(k[2]+k[1]))/dk+2.5))
            dk = (k[-1]-0.5*(k[2]+k[1]))/(nbin-2.5)
        binbounds = np.r_[0.5*(3*k[1]-k[2]),0.5*(k[1]+k[2])+dk*np.arange(nbin-2)]
        if(log):
            binbounds = np.exp(binbounds)
    ## reordering
    reorder = np.searchsorted(binbounds,kindex)
    rho_ = np.zeros(len(binbounds)+1,dtype=rho.dtype)
    kindex_ = np.empty(len(binbounds)+1,dtype=kindex.dtype)
    for ii in range(len(reorder)):
        if(rho_[reorder[ii]]==0):
            kindex_[reorder[ii]] = kindex[ii]
            rho_[reorder[ii]] += rho[ii]
        else:
            kindex_[reorder[ii]] = (kindex_[reorder[ii]]*rho_[reorder[ii]]+kindex[ii]*rho[ii])/(rho_[reorder[ii]]+rho[ii])
            rho_[reorder[ii]] += rho[ii]

    return reorder[pindex],kindex_,rho_



def nhermitianize(field,zerocentered):

    """
        Hermitianizes an arbitrary n-dimensional field. Becomes relatively slow
        for large n.

        Parameters
        ----------
        field : ndarray
            The input field that should be hermitianized.

        zerocentered : bool
            Whether the output array should be zerocentered, i.e. starting with
            negative Fourier modes going over the zero mode to positive modes,
            or not zerocentered, where zero, positive and negative modes are
            simpy ordered consecutively.

        Returns
        -------
        hermfield : ndarray
            The hermitianized field.

    """
    ## shift zerocentered axes
    if(np.any(zerocentered==True)):
        field = np.fft.fftshift(field, axes=shiftaxes(zerocentered))
#    for index in np.ndenumerate(field):
#        negind = tuple(-np.array(index[0]))
#        field[negind] = np.conjugate(index[1])
#        if(field[negind]==field[index[0]]):
#            field[index[0]] = np.abs(index[1])*(np.sign(index[1].real)+(np.sign(index[1].real)==0)*np.sign(index[1].imag)).astype(np.int)
    subshape = np.array(field.shape,dtype=np.int) ## == axes
    maxindex = subshape//2
    subshape[np.argmax(subshape)] = subshape[np.argmax(subshape)]//2+1 ## ~half larges axis
    for ii in np.ndindex(tuple(subshape)):
        negii = tuple(-np.array(ii))
        field[negii] = np.conjugate(field[ii])
    for ii in np.ndindex((2,)*maxindex.size):
        index = tuple(ii*maxindex)
        field[index] = np.abs(field[index])*(np.sign(field[index].real)+(np.sign(field[index].real)==0)*-np.sign(field[index].imag)).astype(np.int) ## minus since overwritten before
    ## reshift zerocentered axes
    if(np.any(zerocentered==True)):
        field = np.fft.fftshift(field,axes=shiftaxes(zerocentered))
    return field

def nhermitianize_fast(field,zerocentered,special=False):

    """
        Hermitianizes an arbitrary n-dimensional field faster.
        Still becomes comparably slow for large n.

        Parameters
        ----------
        field : ndarray
            The input field that should be hermitianized.

        zerocentered : bool
            Whether the output array should be zerocentered, i.e. starting with
            negative Fourier modes going over the zero mode to positive modes,
            or not zerocentered, where zero, positive and negative modes are
            simpy ordered consecutively.

        special : bool, *optional*
            Must be True for random fields drawn from Gaussian or pm1
            distributions.

        Returns
        -------
        hermfield : ndarray
            The hermitianized field.

    """
    ## shift zerocentered axes
    if(np.any(zerocentered==True)):
        field = np.fft.fftshift(field, axes=shiftaxes(zerocentered))
    dummy = np.conjugate(field)
    ## mirror conjugate field
    for ii in range(field.ndim):
        dummy = np.swapaxes(dummy,0,ii)
        dummy = np.flipud(dummy)
        dummy = np.roll(dummy,1,axis=0)
        dummy = np.swapaxes(dummy,0,ii)
    if(special): ## special normalisation for certain random fields
        field = np.sqrt(0.5)*(field+dummy)
        maxindex = np.array(field.shape,dtype=np.int)//2
        for ii in np.ndindex((2,)*maxindex.size):
            index = tuple(ii*maxindex)
            field[index] *= np.sqrt(0.5)
    else: ## regular case
        field = 0.5*(field+dummy)
    ## reshift zerocentered axes
    if(np.any(zerocentered==True)):
        field = np.fft.fftshift(field,axes=shiftaxes(zerocentered))
    return field


def random_hermitian_pm1(datatype,zerocentered,shape):

    """
        Draws a set of hermitianized random, complex pm1 numbers.

    """

    field = np.random.randint(4,high=None,size=np.prod(shape,axis=0,dtype=np.int,out=None)).reshape(shape,order='C')
    dummy = np.copy(field)
    ## mirror field
    for ii in range(field.ndim):
        dummy = np.swapaxes(dummy,0,ii)
        dummy = np.flipud(dummy)
        dummy = np.roll(dummy,1,axis=0)
        dummy = np.swapaxes(dummy,0,ii)
    field = (field+dummy+2*(field>dummy)*((field+dummy)%2))%4 ## wicked magic
    x = np.array([1+0j,0+1j,-1+0j,0-1j],dtype=datatype)[field]
    ## (re)shift zerocentered axes
    if(np.any(zerocentered==True)):
        field = np.fft.fftshift(field,axes=shiftaxes(zerocentered))
    return x


#-----------------------------------------------------------------------------
# Auxiliary functions
#-----------------------------------------------------------------------------

def shiftaxes(zerocentered,st_to_zero_mode=False):

    """
        Shifts the axes in a special way needed for some functions
    """

    axes = []
    for ii in range(len(zerocentered)):
        if(st_to_zero_mode==False)and(zerocentered[ii]):
            axes += [ii]
        if(st_to_zero_mode==True)and(not zerocentered[ii]):
            axes += [ii]
    return axes


def nkdict(axes,dgrid,fourier=True):
    """
        Calculates an n-dimensional array with its entries being the lengths of
        the k-vectors from the zero point of the Fourier grid.

    """
    if(fourier):
        dk = dgrid
    else:
        dk = np.array([1/axes[i]/dgrid[i] for i in range(len(axes))])

    kdict = np.empty(axes)
    for ii in np.ndindex(kdict.shape):
        kdict[ii] = np.sqrt(np.sum(((ii-axes//2)*dk)**2))
    return kdict


def nkdict_fast(axes,dgrid,fourier=True):
    """
        Calculates an n-dimensional array with its entries being the lengths of
        the k-vectors from the zero point of the Fourier grid.

    """
    if(fourier):
        dk = dgrid
    else:
        dk = np.array([1/dgrid[i]/axes[i] for i in range(len(axes))])

    temp_vecs = np.array(np.where(np.ones(axes)),dtype='float').reshape(np.append(len(axes),axes))
    temp_vecs = np.rollaxis(temp_vecs,0,len(temp_vecs.shape))
    temp_vecs -= axes//2
    temp_vecs *= dk
    temp_vecs *= temp_vecs
    return np.sqrt(np.sum((temp_vecs),axis=-1))


def nkdict_fast2(axes,dgrid,fourier=True):
    """
        Calculates an n-dimensional array with its entries being the lengths of
        the k-vectors from the zero point of the grid.

    """
    if(fourier):
        dk = dgrid
    else:
        dk = np.array([1/dgrid[i]/axes[i] for i in range(len(axes))])

    inds = []
    for a in axes:
        inds += [slice(0,a)]
    cords = np.ogrid[inds]

    dists = ((cords[0]-axes[0]//2)*dk[0])**2
    for ii in range(1,len(axes)):
        dists = dists + ((cords[ii]-axes[ii]//2)*dk[ii])**2
    dists = np.sqrt(dists)

    return dists


def nklength(kdict):
    return np.sort(list(set(kdict.flatten())))


#def drawherm(vector,klength,kdict,ps): ## vector = np.zeros(kdict.shape,dtype=np.complex)
#    for ii in np.ndindex(vector.shape):
#        if(vector[ii]==np.complex(0.,0.)):
#            vector[ii] = np.sqrt(0.5*ps[np.searchsorted(klength,kdict[ii])])*np.complex(np.random.normal(0.,1.),np.random.normal(0.,1.))
#            negii = tuple(-np.array(ii))
#            vector[negii] = np.conjugate(vector[ii])
#            if(vector[negii]==vector[ii]):
#                vector[ii] = np.float(np.sqrt(ps[klength==kdict[ii]]))*np.random.normal(0.,1.)
#    return vector

def drawherm(klength,kdict,ps):

    """
        Draws a hermitian random field from a Gaussian distribution.

    """

#    vector = np.zeros(kdict.shape,dtype='complex')
#    for ii in np.ndindex(vector.shape):
#        if(vector[ii]==np.complex(0.,0.)):
#            vector[ii] = np.sqrt(0.5*ps[np.searchsorted(klength,kdict[ii])])*np.complex(np.random.normal(0.,1.),np.random.normal(0.,1.))
#            negii = tuple(-np.array(ii))
#            vector[negii] = np.conjugate(vector[ii])
#            if(vector[negii]==vector[ii]):
#                vector[ii] = np.float(np.sqrt(ps[np.searchsorted(klength,kdict[ii])]))*np.random.normal(0.,1.)
#    return vector
    vec = np.random.normal(loc=0,scale=1,size=kdict.size).reshape(kdict.shape)
    vec = np.fft.fftn(vec)/np.sqrt(np.prod(kdict.shape))
    for ii in np.ndindex(kdict.shape):
        vec[ii] *= np.sqrt(ps[np.searchsorted(klength,kdict[ii])])
    return vec


#def drawwild(vector,klength,kdict,ps,real_corr=1): ## vector = np.zeros(kdict.shape,dtype=np.complex)
#    for ii in np.ndindex(vector.shape):
#        vector[ii] = np.sqrt(real_corr*0.5*ps[klength==kdict[ii]])*np.complex(np.random.normal(0.,1.),np.random.normal(0.,1.))
#    return vector

def drawwild(klength,kdict,ps,real_corr=1):

    """
        Draws a field of arbitrary symmetry from a Gaussian distribution.

    """

    vec = np.empty(kdict.size,dtype=np.complex)
    vec.real = np.random.normal(loc=0,scale=np.sqrt(real_corr*0.5),size=kdict.size)
    vec.imag = np.random.normal(loc=0,scale=np.sqrt(real_corr*0.5),size=kdict.size)
    vec = vec.reshape(kdict.shape)
    for ii in np.ndindex(kdict.shape):
        vec[ii] *= np.sqrt(ps[np.searchsorted(klength,kdict[ii])])
    return vec

