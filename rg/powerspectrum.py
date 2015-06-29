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

#
#def get_power_index(axes,dgrid,zerocentered,irred=False,fourier=True):
#
#    """
#        Returns the index of the Fourier grid points in a numpy
#        array, ordered following the zerocentered flag.
#
#        Parameters
#        ----------
#        axes : ndarray
#            An array with the length of each axis.
#
#        dgrid : ndarray
#            An array with the pixel length of each axis.
#
#        zerocentered : bool
#            Whether the output array should be zerocentered, i.e. starting with
#            negative Fourier modes going over the zero mode to positive modes,
#            or not zerocentered, where zero, positive and negative modes are
#            simpy ordered consecutively.
#
#        irred : bool : *optional*
#            If True, the function returns an array of all k-vector lengths and
#            their degeneracy factors. If False, just the power index array is
#            returned.
#
#        fourier : bool : *optional*
#            Whether the output should be in Fourier space or not
#            (default=False).
#
#        Returns
#        -------
#            index or {klength, rho} : scalar or list
#                Returns either an array of all k-vector lengths and
#                their degeneracy factors or just the power index array
#                depending on the flag irred.
#
#    """
#
#    ## kdict, klength
#    if(np.any(zerocentered==False)):
#        kdict = np.fft.fftshift(nkdict_fast(axes,dgrid,fourier),axes=shiftaxes(zerocentered,st_to_zero_mode=True))
#    else:
#        kdict = nkdict_fast(axes,dgrid,fourier)
#    klength = nklength(kdict)
#    ## output
#    if(irred):
#        rho = np.zeros(klength.shape,dtype=np.int)
#        for ii in np.ndindex(kdict.shape):
#            rho[np.searchsorted(klength,kdict[ii])] += 1
#        return klength,rho
#    else:
#        ind = np.empty(axes,dtype=np.int)
#        for ii in np.ndindex(kdict.shape):
#            ind[ii] = np.searchsorted(klength,kdict[ii])
#        return ind
#
#
#def get_power_indices(axes,dgrid,zerocentered,fourier=True):
#    """
#        Returns the index of the Fourier grid points in a numpy
#        array, ordered following the zerocentered flag.
#
#        Parameters
#        ----------
#        axes : ndarray
#            An array with the length of each axis.
#
#        dgrid : ndarray
#            An array with the pixel length of each axis.
#
#        zerocentered : bool
#            Whether the output array should be zerocentered, i.e. starting with
#            negative Fourier modes going over the zero mode to positive modes,
#            or not zerocentered, where zero, positive and negative modes are
#            simpy ordered consecutively.
#
#        irred : bool : *optional*
#            If True, the function returns an array of all k-vector lengths and
#            their degeneracy factors. If False, just the power index array is
#            returned.
#
#        fourier : bool : *optional*
#            Whether the output should be in Fourier space or not
#            (default=False).
#
#        Returns
#        -------
#        index, klength, rho : ndarrays
#            Returns the power index array, an array of all k-vector lengths and
#            their degeneracy factors.
#
#    """
#
#    ## kdict, klength
#    if(np.any(zerocentered==False)):
#        kdict = np.fft.fftshift(nkdict_fast(axes,dgrid,fourier),axes=shiftaxes(zerocentered,st_to_zero_mode=True))
#    else:
#        kdict = nkdict_fast(axes,dgrid,fourier)
#    klength = nklength(kdict)
#    ## output
#    ind = np.empty(axes,dtype=np.int)
#    rho = np.zeros(klength.shape,dtype=np.int)
#    for ii in np.ndindex(kdict.shape):
#        ind[ii] = np.searchsorted(klength,kdict[ii])
#        rho[ind[ii]] += 1
#    return ind,klength,rho
#
#
#def get_power_indices2(axes,dgrid,zerocentered,fourier=True):
#    """
#        Returns the index of the Fourier grid points in a numpy
#        array, ordered following the zerocentered flag.
#
#        Parameters
#        ----------
#        axes : ndarray
#            An array with the length of each axis.
#
#        dgrid : ndarray
#            An array with the pixel length of each axis.
#
#        zerocentered : bool
#            Whether the output array should be zerocentered, i.e. starting with
#            negative Fourier modes going over the zero mode to positive modes,
#            or not zerocentered, where zero, positive and negative modes are
#            simpy ordered consecutively.
#
#        irred : bool : *optional*
#            If True, the function returns an array of all k-vector lengths and
#            their degeneracy factors. If False, just the power index array is
#            returned.
#
#        fourier : bool : *optional*
#            Whether the output should be in Fourier space or not
#            (default=False).
#
#        Returns
#        -------
#        index, klength, rho : ndarrays
#            Returns the power index array, an array of all k-vector lengths and
#            their degeneracy factors.
#
#    """
#
#    ## kdict, klength
#    if(np.any(zerocentered==False)):
#        kdict = np.fft.fftshift(nkdict_fast2(axes,dgrid,fourier),axes=shiftaxes(zerocentered,st_to_zero_mode=True))
#    else:
#        kdict = nkdict_fast2(axes,dgrid,fourier)
#
#    klength,rho,ind = nkdict_to_indices(kdict)
#
#    return ind,klength,rho
#
#def nkdict_to_indices(kdict):
#
#    kindex,pindex = np.unique(kdict,return_inverse=True)
#    pindex = pindex.reshape(kdict.shape)
#
#    rho = pindex.flatten()
#    rho.sort()
#    rho = np.unique(rho,return_index=True,return_inverse=False)[1]
#    rho = np.append(rho[1:]-rho[:-1],[np.prod(pindex.shape)-rho[-1]])
#
#    return kindex,rho,pindex
#
#
#
#def bin_power_indices(pindex,kindex,rho,log=False,nbin=None,binbounds=None):
#    """
#        Returns the (re)binned power indices associated with the Fourier grid.
#
#        Parameters
#        ----------
#        pindex : ndarray
#            Index of the Fourier grid points in a numpy.ndarray ordered
#            following the zerocentered flag (default=None).
#        kindex : ndarray
#            Array of all k-vector lengths (default=None).
#        rho : ndarray
#            Degeneracy of the Fourier grid, indicating how many k-vectors in
#            Fourier space have the same length (default=None).
#        log : bool
#            Flag specifying if the binning is performed on logarithmic scale
#            (default: False).
#        nbin : integer
#            Number of used bins (default: None).
#        binbounds : {list, array}
#            Array-like inner boundaries of the used bins (default: None).
#
#        Returns
#        -------
#        pindex, kindex, rho : ndarrays
#            The (re)binned power indices.
#
#    """
#    ## boundaries
#    if(binbounds is not None):
#        binbounds = np.sort(binbounds)
#    ## equal binning
#    else:
#        if(log is None):
#            log = False
#        if(log):
#            k = np.r_[0,np.log(kindex[1:])]
#        else:
#            k = kindex
#        dk = np.max(k[2:]-k[1:-1]) ## minimal dk
#        if(nbin is None):
#            nbin = int((k[-1]-0.5*(k[2]+k[1]))/dk-0.5) ## maximal nbin
#        else:
#            nbin = min(int(nbin),int((k[-1]-0.5*(k[2]+k[1]))/dk+2.5))
#            dk = (k[-1]-0.5*(k[2]+k[1]))/(nbin-2.5)
#        binbounds = np.r_[0.5*(3*k[1]-k[2]),0.5*(k[1]+k[2])+dk*np.arange(nbin-2)]
#        if(log):
#            binbounds = np.exp(binbounds)
#    ## reordering
#    reorder = np.searchsorted(binbounds,kindex)
#    rho_ = np.zeros(len(binbounds)+1,dtype=rho.dtype)
#    kindex_ = np.empty(len(binbounds)+1,dtype=kindex.dtype)
#    for ii in range(len(reorder)):
#        if(rho_[reorder[ii]]==0):
#            kindex_[reorder[ii]] = kindex[ii]
#            rho_[reorder[ii]] += rho[ii]
#        else:
#            kindex_[reorder[ii]] = (kindex_[reorder[ii]]*rho_[reorder[ii]]+kindex[ii]*rho[ii])/(rho_[reorder[ii]]+rho[ii])
#            rho_[reorder[ii]] += rho[ii]
#
#    return reorder[pindex],kindex_,rho_
#


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
        print ('maxindex: ', maxindex)
        for ii in np.ndindex((2,)*maxindex.size):
            print ('ii: ', ii)
            index = tuple(ii*maxindex)
            print ('index: ', index)
            field[index] *= np.sqrt(0.5)
    else: ## regular case
        field = 0.5*(field+dummy)
        #field = dummy
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


#def nkdict(axes,dgrid,fourier=True):
#    """
#        Calculates an n-dimensional array with its entries being the lengths of
#        the k-vectors from the zero point of the Fourier grid.
#
#    """
#    if(fourier):
#        dk = dgrid
#    else:
#        dk = np.array([1/axes[i]/dgrid[i] for i in range(len(axes))])
#
#    kdict = np.empty(axes)
#    for ii in np.ndindex(kdict.shape):
#        kdict[ii] = np.sqrt(np.sum(((ii-axes//2)*dk)**2))
#    return kdict
#
#
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


#def nkdict_fast2(axes,dgrid,fourier=True):
#    """
#        Calculates an n-dimensional array with its entries being the lengths of
#        the k-vectors from the zero point of the grid.
#
#    """
#    if(fourier):
#        dk = dgrid
#    else:
#        dk = np.array([1/dgrid[i]/axes[i] for i in range(len(axes))])
#
#    inds = []
#    for a in axes:
#        inds += [slice(0,a)]
#    cords = np.ogrid[inds]
#
#    dists = ((cords[0]-axes[0]//2)*dk[0])**2
#    for ii in range(1,len(axes)):
#        dists = dists + ((cords[ii]-axes[ii]//2)*dk[ii])**2
#    dists = np.sqrt(dists)
#
#    return dists
#
#
def nklength(kdict):
    return np.sort(list(set(kdict.flatten())))


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

