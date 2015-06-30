## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2015 Max-Planck-Society
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

"""
    ..                  __   ____   __
    ..                /__/ /   _/ /  /_
    ..      __ ___    __  /  /_  /   _/  __   __
    ..    /   _   | /  / /   _/ /  /   /  / /  /
    ..   /  / /  / /  / /  /   /  /_  /  /_/  /
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  rg
    ..                               /______/

    NIFTY submodule for regular Cartesian grids.

"""
from __future__ import division

import os
import numpy as np
from scipy.special import erf 
import pylab as pl
from matplotlib.colors import LogNorm as ln
from matplotlib.ticker import LogFormatter as lf

from mpi4py import MPI

from nifty.nifty_about import about
from nifty.nifty_core import point_space,                                    \
                             field
from nifty.nifty_random import random
from nifty.nifty_mpi_data import distributed_data_object
from nifty.nifty_paradict import rg_space_paradict

import fft_rg

'''
try:
    import gfft as gf
except(ImportError):
    about.infos.cprint('INFO: "plain" gfft version 0.1.0')
    import gfft_rg as gf
'''




##-----------------------------------------------------------------------------

class rg_space(point_space):
    """
        ..      _____   _______
        ..    /   __/ /   _   /
        ..   /  /    /  /_/  /
        ..  /__/     \____  /  space class
        ..          /______/

        NIFTY subclass for spaces of regular Cartesian grids.

        Parameters
        ----------
        num : {int, numpy.ndarray}
            Number of gridpoints or numbers of gridpoints along each axis.
        naxes : int, *optional*
            Number of axes (default: None).
        zerocenter : {bool, numpy.ndarray}, *optional*
            Whether the Fourier zero-mode is located in the center of the grid
            (or the center of each axis speparately) or not (default: True).
        hermitian : bool, *optional*
            Whether the fields living in the space follow hermitian symmetry or
            not (default: True).
        purelyreal : bool, *optional*
            Whether the field values are purely real (default: True).
        dist : {float, numpy.ndarray}, *optional*
            Distance between two grid points along each axis (default: None).
        fourier : bool, *optional*
            Whether the space represents a Fourier or a position grid
            (default: False).

        Notes
        -----
        Only even numbers of grid points per axis are supported.
        The basis transformations between position `x` and Fourier mode `k`
        rely on (inverse) fast Fourier transformations using the
        :math:`exp(2 \pi i k^\dagger x)`-formulation.

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing information on the axes of the
            space in the following form: The first entries give the grid-points
            along each axis in reverse order; the next entry is 0 if the
            fields defined on the space are purely real-valued, 1 if they are
            hermitian and complex, and 2 if they are not hermitian, but
            complex-valued; the last entries hold the information on whether
            the axes are centered on zero or not, containing a one for each
            zero-centered axis and a zero for each other one, in reverse order.
        datatype : numpy.dtype
            Data type of the field values for a field defined on this space,
            either ``numpy.float64`` or ``numpy.complex128``.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for regular grids.
        vol : numpy.ndarray
            One-dimensional array containing the distances between two grid
            points along each axis, in reverse order. By default, the total
            length of each axis is assumed to be one.
        fourier : bool
            Whether or not the grid represents a Fourier basis.
    """
    epsilon = 0.0001 ## relative precision for comparisons

    def __init__(self, num, naxes=None, zerocenter=False, hermitian=True,\
                purelyreal=True, dist=None, fourier=False):
        """
            Sets the attributes for an rg_space class instance.

            Parameters
            ----------
            num : {int, numpy.ndarray}
                Number of gridpoints or numbers of gridpoints along each axis.
            naxes : int, *optional*
                Number of axes (default: None).
            zerocenter : {bool, numpy.ndarray}, *optional*
                Whether the Fourier zero-mode is located in the center of the
                grid (or the center of each axis speparately) or not
                (default: False).
            hermitian : bool, *optional*
                Whether the fields living in the space follow hermitian
                symmetry or not (default: True).
            purelyreal : bool, *optional*
                Whether the field values are purely real (default: True).
            dist : {float, numpy.ndarray}, *optional*
                Distance between two grid points along each axis
                (default: None).
            fourier : bool, *optional*
                Whether the space represents a Fourier or a position grid
                (default: False).

            Returns
            -------
            None
        """

        complexity = 2-(bool(hermitian) or bool(purelyreal))-bool(purelyreal)
        if np.isscalar(num):
            num = (num,)*np.asscalar(np.array(naxes))
            
        self.paradict = rg_space_paradict(num=num, complexity=complexity, 
                                          zerocenter=zerocenter)        
        
        
        naxes = len(self.paradict['num'])

        ## set data type
        if  self.paradict['complexity'] == 0:
            self.datatype = np.float64
        else:
            self.datatype = np.complex128

        self.discrete = False

        ## set volume
        if(dist is None):
            dist = 1/np.array(self.paradict['num'], dtype=self.datatype)
        elif(np.isscalar(dist)):
            dist = self.datatype(dist)*np.ones(naxes,dtype=self.datatype,\
                                                order='C')
        else:
            dist = np.array(dist,dtype=self.datatype)
            if(np.size(dist) == 1):
                dist = dist*np.ones(naxes,dtype=self.datatype,order='C')
            if(np.size(dist)!=naxes):
                raise ValueError(about._errors.cstring(\
                    "ERROR: size mismatch ( "+str(np.size(dist))+" <> "+\
                    str(naxes)+" )."))
        if(np.any(dist<=0)):
            raise ValueError(about._errors.cstring(\
                "ERROR: nonpositive distance(s)."))
        self.vol = np.real(dist)

        self.fourier = bool(fourier)
        
        ## Initializes the fast-fourier-transform machine, which will be used 
        ## to transform the space
        self.fft_machine = fft_rg.fft_factory()
        
        ## Initialize the power_indices object which takes care of kindex,
        ## pindex, rho and the pundex for a given set of parameters
        if self.fourier:        
            self.power_indices = power_indices(shape=self.shape(),
                                dgrid = dist,
                                zerocentered = self.paradict['zerocenter']
                                )

    @property
    def para(self):
        temp = np.array(self.paradict['num'] + \
                         [self.paradict['complexity']] + \
                         self.paradict['zerocenter'], dtype=int)
        return temp
        
    
    @para.setter
    def para(self, x):
        self.paradict['num'] = x[:(np.size(x)-1)//2]
        self.paradict['zerocenter'] = x[(np.size(x)+1)//2:]
        self.paradict['complexity'] = x[(np.size(x)-1)//2]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def apply_scalar_function(self, x, function, inplace=False):
        return x.apply_scalar_function(function, inplace=inplace)

    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
    def unary_operation(self, x, op='None', **kwargs):
        """
        x must be a distributed_data_object which is compatible with the space!
        Valid operations are
        
        """
        
        translation = {"pos" : lambda y: getattr(y, '__pos__')(),
                        "neg" : lambda y: getattr(y, '__neg__')(),
                        "abs" : lambda y: getattr(y, '__abs__')(),
                        "nanmin" : lambda y: getattr(y, 'nanmin')(),
                        "min" : lambda y: getattr(y, 'amin')(),
                        "nanmax" : lambda y: getattr(y, 'nanmax')(),
                        "max" : lambda y: getattr(y, 'amax')(),
                        "median" : lambda y: getattr(y, 'median')(),
                        "mean" : lambda y: getattr(y, 'mean')(),
                        "std" : lambda y: getattr(y, 'std')(),
                        "var" : lambda y: getattr(y, 'var')(),
                        "argmin" : lambda y: getattr(y, 'argmin')(),
                        "argmin_flat" : lambda y: getattr(y, 'argmin_flat')(),
                        "argmax" : lambda y: getattr(y, 'argmax')(),
                        "argmax_flat" : lambda y: getattr(y, 'argmax_flat')(),
                        "conjugate" : lambda y: getattr(y, 'conjugate')(),
                        "None" : lambda y: y}
                        
        return translation[op](x, **kwargs)      


    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def naxes(self):
        """
            Returns the number of axes of the grid.

            Returns
            -------
            naxes : int
                Number of axes of the regular grid.
        """
#        return (np.size(self.para)-1)//2
        return len(self.shape())

    def zerocenter(self):
        """
            Returns information on the centering of the axes.

            Returns
            -------
            zerocenter : numpy.ndarray
                Whether the grid is centered on zero for each axis or not.
        """
        #return self.para[-(np.size(self.para)-1)//2:][::-1].astype(np.bool)
        return self.paradict['zerocenter']

    def dist(self):
        """
            Returns the distances between grid points along each axis.

            Returns
            -------
            dist : np.ndarray
                Distances between two grid points on each axis.
        """
        return self.vol
 
    def shape(self):
        return np.array(self.paradict['num'])

    def dim(self, split=False):
        """
            Computes the dimension of the space, i.e.\  the number of pixels.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension split up, i.e. the numbers of
                pixels along each axis, or their product (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension(s) of the space. If ``split==True``, a
                one-dimensional array with an entry for each axis is returned.
        """
        ## dim = product(n)
        if split == True:
            return self.shape()
        else:
            return np.prod(self.shape())

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space, i.e.\  the
            number of grid points multiplied with one or two, depending on
            complex-valuedness and hermitian symmetry of the fields.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        ## dof ~ dim
        if self.paradict['complexity'] < 2:
            return np.prod(self.paradict['num'])
        else:
            return 2*np.prod(self.paradict['num'])

#        if(self.para[(np.size(self.para)-1)//2]<2):
#            return np.prod(self.para[:(np.size(self.para)-1)//2],axis=0,dtype=None,out=None)
#        else:
#            return 2*np.prod(self.para[:(np.size(self.para)-1)//2],axis=0,dtype=None,out=None)


    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def enforce_power(self, spec, size=None, kindex=None, codomain=None,
                      log=False, nbin=None, binbounds=None):
        """
            Provides a valid power spectrum array from a given object.

            Parameters
            ----------
            spec : {float, list, numpy.ndarray, nifty.field, function}
                Fiducial power spectrum from which a valid power spectrum is to
                be calculated. Scalars are interpreted as constant power
                spectra.

            Returns
            -------
            spec : numpy.ndarray
                Valid power spectrum.

            Other parameters
            ----------------
            size : int, *optional*
                Number of bands the power spectrum shall have (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band.
            codomain : nifty.space, *optional*
                A compatible codomain for power indexing (default: None).
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on 
                logarithmic scale or not; if set, the number of used bins is 
                set automatically (if not given otherwise); by default no 
                binning is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to 
                ``False``; iintegers below the minimum of 3 induce an automatic
                setting; by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
        """
        
        
        
        ## Setting up the local variables: kindex 
        ## The kindex is only necessary if spec is a function or if 
        ## the size is not set explicitly 
        if kindex == None and (size == None or callable(spec) == True):
            ## Determine which space should be used to get the kindex
            if self.fourier == True:
                kindex_supply_space = self
            else:
                ## Check if the given codomain is compatible with the space  
                try:                
                    assert(self.check_codomain(codomain))
                    kindex_supply_space = codomain
                except(AssertionError):
                    about.warnings.cprint("WARNING: Supplied codomain is "+\
                    "incompatible. Generating a generic codomain. This can "+\
                    "be expensive!")
                    kindex_supply_space = self.get_codomain()
            kindex = kindex_supply_space.\
                        power_indices.get_index_dict(log=log, nbin=nbin,
                                                     binbounds=binbounds)\
                                                     ['kindex']
        

        
        ## Now it's about to extract a powerspectrum from spec
        ## First of all just extract a numpy array. The shape is cared about
        ## later.
                    
        ## Case 1: spec is a function
        if callable(spec) == True:
            ## Try to plug in the kindex array in the function directly            
            try:
                spec = np.array(spec(kindex), dtype=self.datatype)
            except:
                ## Second try: Use a vectorized version of the function.
                ## This is slower, but better than nothing
                try:
                    spec = np.vectorize(spec)(kindex)
                except:
                    raise TypeError(about._errors.cstring(
                        "ERROR: invalid power spectra function.")) 
    
        ## Case 2: spec is a field:
        elif isinstance(spec, field):
            spec = spec[:]
            spec = np.array(spec, dtype = self.datatype).flatten()
            
        ## Case 3: spec is a scalar or something else:
        else:
            spec = np.array(spec, dtype = self.datatype).flatten()
        
            
        ## Make some sanity checks
        ## Drop imaginary part
        temp_spec = np.real(spec)
        try:
            np.testing.assert_allclose(spec, temp_spec)
        except(AssertionError):
            about.warnings.cprint("WARNING: Dropping imaginary part.")
        spec = temp_spec
        
        ## check finiteness
        if not np.all(np.isfinite(spec)):
            about.warnings.cprint("WARNING: infinite value(s).")
        
        ## check positivity (excluding null)
        if np.any(spec<0):
            raise ValueError(about._errors.cstring(
                                "ERROR: nonpositive value(s)."))
        if np.any(spec==0):
            about.warnings.cprint("WARNING: nonpositive value(s).")            
        
        ## Set the size parameter        
        if size == None:
            size = len(kindex)
        
        ## Fix the size of the spectrum
        ## If spec is singlevalued, expand it
        if np.size(spec) == 1:
            spec = spec*np.ones(size, dtype=spec.dtype, order='C')
        ## If the size does not fit at all, throw an exception
        elif np.size(spec) < size:
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+\
                             str(np.size(spec))+" < "+str(size)+" )."))
        elif np.size(spec) > size:
            about.warnings.cprint("WARNING: power spectrum cut to size ( == "+\
                                str(size)+" ).")
            spec = spec[:size]
        
        return spec


    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_power_indices(self, log=False, nbin=None, binbounds=None, **kwargs):
        """
            Sets the (un)indexing objects for spectral indexing internally.

            Parameters
            ----------
            log : bool
                Flag specifying if the binning is performed on logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer
                Number of used bins; if given `log` is set to ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).

            Returns
            -------
            None

            See Also
            --------
            get_power_indices

            Raises
            ------
            AttributeError
                If ``self.fourier == False``.
            ValueError
                If the binning leaves one or more bins empty.

        """

        about.warnings.cflush("WARNING: set_power_indices is a deprecated"+\
                                "function. Please use the interface of"+\
                                "self.power_indices in future!")
        self.power_indices.set_default(log=log, nbin=nbin, binbounds=binbounds)
        return None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def cast(self, x, verbose=False):
        """
            Computes valid field values from a given object, trying
            to translate the given data into a valid form. Thereby it is as 
            benevolent as possible. 

            Parameters
            ----------
            x : {float, numpy.ndarray, nifty.field}
                Object to be transformed into an array of valid field values.

            Returns
            -------
            x : numpy.ndarray, distributed_data_object
                Array containing the field values, which are compatible to the
                space.

            Other parameters
            ----------------
            verbose : bool, *optional*
                Whether the method should raise a warning if information is 
                lost during casting (default: False).
        """
        ## Case 1: x is a field
        if isinstance(x, field):
            if verbose:
                ## Check if the domain matches
                if(self != x.domain):
                    about.warnings.cflush(\
                    "WARNING: Getting data from foreign domain!")
            ## Extract the data, whatever it is, and cast it again
            return self.cast(x.val)
        
        ## Case 2: x is a distributed_data_object
        if isinstance(x, distributed_data_object):
            ## Check the shape
            if np.any(x.shape != self.shape()):           
                ## Check if at least the number of degrees of freedom is equal
                if x.dim() == self.dim():
                    ## If the number of dof is equal or 1, use np.reshape...
                    about.warnings.cflush(\
                    "WARNING: Trying to reshape the data. This operation is "+\
                    "expensive as it consolidates the full data!\n")
                    temp = x.get_full_data()
                    temp = np.reshape(temp, self.shape())             
                    ## ... and cast again
                    return self.cast(temp)
              
                else:
                    raise ValueError(about._errors.cstring(\
                    "ERROR: Data has incompatible shape!"))
                    
            ## Check the datatype
            if x.dtype != self.datatype:
                about.warnings.cflush(\
                "WARNING: Datatypes are uneqal (own: "\
                + str(self.datatype) + " <> foreign: " + str(x.dtype) \
                + ") and will be casted! "\
                + "Potential loss of precision!\n")
                temp = x.copy_empty(dtype=self.datatype)
                temp.set_local_data(x.get_local_data())
                temp.hermitian = x.hermitian
                x = temp
            
            ## Check hermitianity/reality
            if self.paradict['complexity'] == 0:
                if x.is_completely_real == False:
                    about.warnings.cflush(\
                    "WARNING: Data is not completely real. Imaginary part "+\
                    "will be discarded!\n")
                    temp = x.copy_empty()            
                    temp.set_local_data(np.real(x.get_local_data()))
                    x = temp
            
            elif self.paradict['complexity'] == 1:
                if x.hermitian == False and about.hermitianize.status:
                    about.warnings.cflush(\
                    "WARNING: Data gets hermitianized. This operation is "+\
                    "extremely expensive\n")
                    #temp = x.copy_empty()            
                    #temp.set_full_data(gp.nhermitianize_fast(x.get_full_data(), 
                    #    (False, )*len(x.shape)))
                    x = utilities.hermitianize(x)
                
            return x
                
        ## Case 3: x is something else
        ## Use general d2o casting 
        x = distributed_data_object(x, global_shape=self.shape(),\
            dtype=self.datatype)       
        ## Cast the d2o
        return self.cast(x)

    def enforce_values(self,x,extend=True):
        """
            Computes valid field values from a given object, taking care of
            data types, shape, and symmetry.

            Parameters
            ----------
            x : {float, numpy.ndarray, nifty.field}
                Object to be transformed into an array of valid field values.

            Returns
            -------
            x : numpy.ndarray
                Array containing the valid field values.

            Other parameters
            ----------------
            extend : bool, *optional*
                Whether a scalar is extented to a constant array or not
                (default: True).
        """
        about.warnings.cflush(\
            "WARNING: enforce_values is deprecated function. Please use self.cast")
        if(isinstance(x,field)):
            if(self==x.domain):
                if(self.datatype is not x.domain.datatype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '"+str(np.result_type(self.datatype))+"' <> '"+str(np.result_type(x.domain.datatype))+"' )."))
                else:
                    x = np.copy(x.val)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            if(np.size(x)==1):
                if(extend):
                    x = self.datatype(x)*np.ones(self.dim(split=True),dtype=self.datatype,order='C')
                else:
                    if(np.isscalar(x)):
                        x = np.array([x],dtype=self.datatype)
                    else:
                        x = np.array(x,dtype=self.datatype)
            else:
                x = self.enforce_shape(np.array(x,dtype=self.datatype))

        ## hermitianize if ...
        if(about.hermitianize.status)and(np.size(x)!=1)and(self.para[(np.size(self.para)-1)//2]==1):
            #x = gp.nhermitianize_fast(x,self.para[-((np.size(self.para)-1)//2):].astype(np.bool),special=False)
            x = utilities.hermitianize(x)
        ## check finiteness
        if(not np.all(np.isfinite(x))):
            about.warnings.cprint("WARNING: infinite value(s).")

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters, taking into account possible complex-valuedness
            and hermitian symmetry.

            Returns
            -------
            x : numpy.ndarray
                Valid field values.

            Other parameters
            ----------------
            random : string, *optional*
                Specifies the probability distribution from which the random
                numbers are to be drawn.
                Supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard
                    deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            spec : {scalar, list, numpy.ndarray, nifty.field, function}, *optional*
                Power spectrum (default: 1).
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index of each band
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band (default: None).
            codomain : nifty.rg_space, *optional*
                A compatible codomain (default: None).
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).            
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        
        ## Parse the keyword arguments
        arg = random.parse_arguments(self,**kwargs)
        
        ## Prepare the empty distributed_data_object
        sample = distributed_data_object(global_shape=self.shape(), 
                                         dtype=self.datatype)

        ## Should the output be hermitianized?
        hermitianizeQ = about.hermitianize.status and \
                        self.paradict['complexity']          

        ## Case 1: uniform distribution over {-1,+1}/{1,i,-1,-i}
        if arg[0] == 'pm1' and hermitianizeQ == False:
            gen = lambda s: random.pm1(datatype=self.datatype,
                                       shape = s)
            sample.apply_generator(gen)
                        
        elif arg[0] == 'pm1' and hermitianizeQ == True:
            sample = self.get_random_values(random = 'uni', vmin=-1, vmax=1)
            local_data = sample.get_local_data()
            if issubclass(sample.dtype, np.complexfloating):
                temp_data = local_data.copy()
                local_data[temp_data.real >= 0.5] = 1
                local_data[(temp_data.real >= 0)*(temp_data.real < 0.5)] = -1
                local_data[(temp_data.real < 0)*(temp_data.imag >= 0)] = 1j
                local_data[(temp_data.real < 0)*(temp_data.imag < 0)] = -1j
            else:
                local_data[local_data >= 0] = 1
                local_data[local_data < 0] = -1
            sample.set_local_data(local_data)
            
        ## Case 2: normal distribution with zero-mean and a given standard
        ##         deviation or variance
        elif arg[0] == 'gau':
            gen = lambda s: random.gau(datatype=self.datatype,
                                       shape = s,
                                       mean = arg[1],
                                       dev = arg[2],
                                       var = arg[3])
            sample.apply_generator(gen)
            
            if hermitianizeQ == True:
                sample = utilities.hermitianize(sample)

        ## Case 3: uniform distribution
        elif arg[0] == "uni" and hermitianizeQ == False:
            gen = lambda s: random.uni(datatype=self.datatype,
                                       shape = s,
                                       vmin = arg[1],
                                       vmax = arg[2])
            sample.apply_generator(gen)
            
        elif arg[0] == "uni" and hermitianizeQ == True:
            ## For a hermitian uniform sample, generate a gaussian one
            ## and then convert it to a uniform one
            sample = self.get_random_values(random = 'gau')
            ## Use the cummulative of the gaussian, the error function in order 
            ## to transform it to a uniform distribution.
            if issubclass(sample.dtype, np.complexfloating):
                temp_func = lambda x: erf(x.real) + 1j*erf(x.imag)                  
            else:
                temp_func = lambda x: erf(x/np.sqrt(2))
            sample.apply_scalar_function(function = temp_func,
                                             inplace = True)
            
            ## Shift and stretch the uniform distribution into the given limits
            ## sample = (sample + 1)/2 * (vmax-vmin) + vmin
            vmin = arg[1]
            vmax = arg[2]            
            sample *= (vmax-vmin)/2.
            sample += 1/2.*(vmax+vmin)
            

        elif(arg[0]=="syn"):
            spec = arg[1]
            kpack = arg[2]
            harmonic_domain = arg[3]
            log = arg[4]
            nbin = arg[5]
            binbounds = arg[6]
            ## Check whether there is a kpack available or not.
            ## kpack is only used for computing kdict and extracting kindex
            ## If not, take kdict and kindex from the fourier_domain
            if kpack == None:
                power_indices =\
                    harmonic_domain.power_indices.get_index_dict(log = log,
                                                        nbin = nbin,
                                                        binbounds = binbounds)
                
                kindex = power_indices['kindex']
                kdict = power_indices['kdict']
                kpack = [power_indices['pindex'], power_indices['kindex']]
            else:
                kindex = kpack[1]
                kdict = harmonic_domain.power_indices.\
                    __compute_kdict_from_pindex_kindex__(kpack[0], kpack[1])           
                

            ## draw the random samples
            ## Case 1: self is a harmonic space
            if self.fourier:
                ## subcase 1: self is real
                ## -> simply generate a random field in fourier space and 
                ## weight the entries accordingly to the powerspectrum
                if self.paradict['complexity'] == 0:
                    ## set up the sample object. Overwrite the default from 
                    ## above to be sure, that the distribution strategy matches
                    ## with the one from kdict
                    sample = kdict.copy_empty(dtype = self.datatype)
                    ## set up the random number generator
                    gen = lambda s: np.random.normal(loc=0, scale=1, size=s)
                    ## apply the random number generator                    
                    sample.apply_generator(gen)

                
                ## subcase 2: self is hermitian but probably complex
                ## -> generate a real field (in position space) and transform
                ## it to harmonic space -> field in harmonic space is 
                ## hermitian. Now weight the modes accordingly to the 
                ## powerspectrum.
                elif self.paradict['complexity'] == 1:
                    temp_codomain = self.get_codomain()
                    ## set up the sample object. Overwrite the default from 
                    ## above to be sure, that the distribution strategy matches
                    ## with the one from kdict
                    sample = kdict.copy_empty(
                                            dtype = temp_codomain.datatype)
                    ## set up the random number generator

                    gen = lambda s: np.random.normal(loc=0, scale=1, size=s)
                    ## apply the random number generator                    
                    sample.apply_generator(gen)
                    
                    ## In order to get the normalisation right, the sqrt
                    ## of self.dim must be divided out. 
                    ## Furthermore, the normalisation in the fft routine 
                    ## must be undone
                    ## TODO: Insert explanation
                    sqrt_of_dim = np.sqrt(self.dim())
                    sample /= sqrt_of_dim
                    sample = temp_codomain.calc_weight(sample, power=-1)

                    ## tronsform the random field to harmonic space
                    sample = temp_codomain.\
                                        calc_transform(sample, codomain=self)
                    
                    ## ensure that the kdict and the harmonic_sample have the
                    ## same distribution strategy
                    assert(kdict.distribution_strategy ==\
                            sample.distribution_strategy)

                    
                ## subcase 3: self is fully complex
                ## -> generate a complex random field in harmonic space and
                ## weight the modes accordingly to the powerspectrum
                elif self.paradict['complexity'] == 2:
                    ## set up the sample object. Overwrite the default from 
                    ## above to be sure, that the distribution strategy matches
                    ## with the one from kdict
                    sample = kdict.copy_empty(dtype = self.datatype)
                    ## set up the random number generator
                    gen = lambda s: (
                        np.random.normal(loc=0, scale=1/np.sqrt(2), size=s)+
                        np.random.normal(loc=0, scale=1/np.sqrt(2), size=s)*1.j
                        )
                    ## apply the random number generator                    
                    sample.apply_generator(gen)
                
                ## apply the powerspectrum renormalization
                ## therefore extract the local data from kdict
                local_kdict = kdict.get_local_data()
                rescaler = np.sqrt(
                            spec[np.searchsorted(kindex,local_kdict)])
                sample.apply_scalar_function(lambda x: x*rescaler, 
                                             inplace=True)
            ## Case 2: self is a position space
            else:
                ## get a suitable codomain
                temp_codomain = self.get_codomain()                   

                ## subcase 1: self is a real space. 
                ## -> generate a hermitian sample with the codomain in harmonic
                ## space and make a fourier transformation.
                if self.paradict['complexity'] == 0:
                    ## check that the codomain is hermitian
                    assert(temp_codomain.paradict['complexity'] == 1)
                                                          
                ## subcase 2: self is hermitian but probably complex
                ## -> generate a real-valued random sample in fourier space
                ## and transform it to real space
                elif self.paradict['complexity'] == 1:
                    ## check that the codomain is real
                    assert(temp_codomain.paradict['complexity'] == 0)            
                
                ## subcase 3: self is fully complex
                ## -> generate a complex-valued random sample in fourier space
                ## and transform it to real space
                elif self.paradict['complexity'] == 2:
                    ## check that the codomain is real
                    assert(temp_codomain.paradict['complexity'] == 2)            


                ## Get a hermitian/real/complex sample in harmonic space from 
                ## the codomain
                sample = temp_codomain.get_random_values(
                                                    random='syn',
                                                    pindex = kpack[0],
                                                    kindex = kpack[1],
                                                    spec = spec,
                                                    codomain = self,
                                                    log = log,
                                                    nbin = nbin,
                                                    binbounds = binbounds
                                                    )
                ## Correct the weighting
                #sample = self.calc_weight(sample, power=-1)
                
                ## Take the fourier transform
                sample = temp_codomain.calc_transform(sample, 
                                                      codomain = self)

            if self.paradict['complexity'] == 1:
               sample.hermitian = True
           
        else:
            raise KeyError(about._errors.cstring(
                        "ERROR: unsupported random key '"+str(arg[0])+"'."))
     
             
        return sample





    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def check_codomain(self,codomain):
        """
            Checks whether a given codomain is compatible to the space or not.

            Parameters
            ----------
            codomain : nifty.space
                Space to be checked for compatibility.

            Returns
            -------
            check : bool
                Whether or not the given codomain is compatible to the space.
        """
        if codomain == None:
            return False
            
        if(not isinstance(codomain,rg_space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        ## check number of number and size of axes 
        if not np.all(self.paradict['num'] == codomain.paradict['num']):
            return False
            
        ## check fourier flag
        if self.fourier == codomain.fourier:
            return False
            
        ## check complexity-type
        ## prepare the shorthands
        dcomp = self.paradict['complexity']
        cocomp = codomain.paradict['complexity']
        
        ## Case 1: if the domain is copmleteley complex 
        ## -> the codomain must be complex, too
        if dcomp == 2:
            if cocomp != 2:
                return False
        ## Case 2: domain is hermitian
        ## -> codmomain can be real. If it is marked as hermitian or even 
        ## fully complex, a warning is raised
        elif dcomp == 1:
            if cocomp > 0:
                about.warnings.cprint("WARNING: Unrecommended codomain! "+\
                    "The domain is hermitian, hence the codomain should "+\
                    "be restricted to real values!")
        
        ## Case 3: domain is real
        ## -> codmain should be hermitian
        elif dcomp == 0:
            if cocomp == 2:
                about.warnings.cprint("WARNING: Unrecommended codomain! "+\
                    "The domain is real, hence the codomain should "+\
                    "be restricted to hermitian configurations!")
            elif cocomp == 0:
                return False

        ## Check if the distances match, i.e. dist'=1/(num*dist)
        if not np.all(
                np.absolute(self.paradict['num']*
                            self.vol*
                            codomain.vol-1) < self.epsilon):
            return False
            
        return True

        
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self, coname=None, cozerocenter=None, **kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  either a shifted grid or a Fourier conjugate
            grid.

            Parameters
            ----------
            coname : string, *optional*
                String specifying a desired codomain (default: None).
            cozerocenter : {bool, numpy.ndarray}, *optional*
                Whether or not the grid is zerocentered for each axis or not
                (default: None).

            Returns
            -------
            codomain : nifty.rg_space
                A compatible codomain.

            Notes
            -----
            Possible arguments for `coname` are ``'f'`` in which case the
            codomain arises from a Fourier transformation, ``'i'`` in which 
            case it arises from an inverse Fourier transformation.If no 
            `coname` is given, the Fourier conjugate grid is produced.
        """
        naxes = self.naxes() 
        ## Parse the cozerocenter input
        if(cozerocenter is None):
            cozerocenter = self.paradict['zerocenter']
        ## if the input is something scalar, cast it to a boolean
        elif(np.isscalar(cozerocenter)):
            cozerocenter = bool(cozerocenter)
        ## if it is not a scalar...
        else:
            ## ...cast it to a numpy array of booleans
            cozerocenter = np.array(cozerocenter,dtype=np.bool)
            ## if it was a list of length 1, extract the boolean
            if(np.size(cozerocenter)==1):
                cozerocenter = np.asscalar(cozerocenter)
            ## if the length of the input does not match the number of 
            ## dimensions, raise an exception
            elif(np.size(cozerocenter)!=naxes):
                raise ValueError(about._errors.cstring(
                    "ERROR: size mismatch ( "+\
                    str(np.size(cozerocenter))+" <> "+str(naxes)+" )."))
        
        ## Set up the initialization variables
        num = self.paradict['num']
        purelyreal = (self.paradict['complexity'] == 1)        
        hermitian = (self.paradict['complexity'] < 2)
        dist = 1/(self.paradict['num']*self.vol)        
        
        if coname == None:
            fourier = bool(not self.fourier)            
        elif coname[0] == 'f':
            fourier = True
        elif coname[0] == 'i':
            fourier = False
        else:
            raise ValueError(about._errors.cstring(
                                            "ERROR: Unknown coname keyword"))
        new_space = rg_space(num,
                             zerocenter = cozerocenter,
                             hermitian = hermitian,
                             purelyreal = purelyreal,
                             dist = dist,
                             fourier = fourier)                                            
        return new_space

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self, total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions. In the case of an :py:class:`rg_space`, the
            meta volumes are simply the pixel volumes.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each pixel (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the pixels or the complete space.
        """
        if(total):
            return self.dim(split=False)*np.prod(self.vol)
        else:
            mol = np.ones(self.dim(split=True),dtype=self.vol.dtype)
            return self.calc_weight(mol,power=1)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_weight(self, x, power=1):
        """
            Weights a given array with the pixel volumes to a given power.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be weighted.
            power : float, *optional*
                Power of the pixel volumes to be used (default: 1).

            Returns
            -------
            y : numpy.ndarray
                Weighted array.
        """
        x = self.cast(x)
        is_hermitianQ = x.hermitian
        ## weight
        x =  x * (np.prod(self.vol)**power)
        x.hermitian = is_hermitianQ
        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def calc_dot(self, x, y):
        """
            Computes the discrete inner product of two given arrays of field
            values.

            Parameters
            ----------
            x : numpy.ndarray
                First array
            y : numpy.ndarray
                Second array

            Returns
            -------
            dot : scalar
                Inner product of the two arrays.
        """
        x = self.cast(x)
        y = self.cast(y)
        result = x.vdot(y)
        if np.isreal(result):
            result = np.asscalar(np.real(result))      
        if self.paradict['complexity'] != 2:
            if(np.absolute(result.imag) > self.epsilon**2\
                                          *np.absolute(result.real)):
                about.warnings.cprint(
                    "WARNING: Discarding considerable imaginary part.")
            result = np.asscalar(np.real(result))      
        return result
        

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self, x, codomain=None, **kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.rg_space, *optional*
                Target space to which the transformation shall map
                (default: None).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array
        """
        x = self.cast(x)
        
        if(codomain is None):
            codomain = self.get_codomain()
        
        ## Check if the given codomain is suitable for the transformation
        if (not isinstance(codomain, rg_space)) or \
                (not self.check_codomain(codomain)):
            raise ValueError(about._errors.cstring(
                                "ERROR: unsupported codomain."))
        if codomain.fourier == True:
            ## correct for forward fft
            x = self.calc_weight(x, power=1)
        
#            ## correct for inverse fft
#            x = self.calc_weight(x, power=1)
#            x *= self.dim(split=False)
        
        ## Perform the transformation
        Tx = self.fft_machine.transform(val=x, domain=self, codomain=codomain, 
                                        **kwargs)

        if codomain.fourier == False:
            ## correct for inverse fft
            Tx = codomain.calc_weight(Tx, power=-1)


        ## when the target space is purely real, the result of the 
        ## transformation must be corrected accordingly. Using the casting 
        ## method of codomain is sufficient
        ## TODO: Let .transform  yield the correct datatype
        Tx = codomain.cast(Tx)
        
        return Tx

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self, x, sigma=0, codomain=None):
        """
            Smoothes an array of field values by convolution with a Gaussian
            kernel.

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values to be smoothed.
            sigma : float, *optional*
                Standard deviation of the Gaussian kernel, specified in units
                of length in position space; for testing: a sigma of -1 will be
                reset to a reasonable value (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.
        """

        
        ## Check sigma
        if sigma == 0:
            return x
        elif sigma == -1:
            about.infos.cprint(
                "INFO: Resetting sigma to sqrt(2)*max(dist).")
            sigma = np.sqrt(2)*np.max(self.dist())
        elif(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        

        ## if a codomain was given...
        if codomain != None:
            ## ...check if it was suitable
            if not isinstance(codomain, rg_space):
                raise ValueError(about._errors.cstring(
                    "ERROR: codomain is not a rg_space instance!"))
            if self.fourier == False and codomain.fourier == False:
                raise ValueError(about._errors.cstring(
                    "ERROR: fourier_domain is not a fourier space!"))
            if not self.check_codomain(codomain):
                raise ValueError(about._errors.cstring(
                    "ERROR: fourier_codomain is not a valid codomain!"))
        elif self.fourier == False:
            codomain = self.get_codomain()

        ## Case1: 
        ## If self is a position-space, fourier transform the input and
        ## call calc_smooth of the fourier codomain
        if self.fourier == False:
            x = self.calc_transform(x, codomain = codomain)
            x = codomain.calc_smooth(x, sigma)
            x = codomain.calc_transform(x, codomain = self)
            return x
        
        ## Case 2: 
        ## if self is fourier multiply the gaussian kernel, etc...
        
        ## Cast the input
        x = self.cast(x)       
         
        ## if x is hermitian it remains hermitian during smoothing
        remeber_hermitianQ = x.hermitian

        ## Define the Gaussian kernel function 
        gaussian = lambda x: np.exp(-2.*np.pi**2*x**2*sigma**2)
        
        ## Define the variables in the dialect of the legacy smoothing.py 
        nx = self.shape()
        dx = 1/nx/self.vol
        ## Multiply the data along each axis with suitable the gaussian kernel
        for i in range(len(nx)):
            ## Prepare the exponent
            dk = 1./nx[i]/dx[i]
            nk = nx[i]
            k = -0.5*nk*dk + np.arange(nk)*dk
            if self.paradict['zerocenter'][i] == False:
                k = np.fft.fftshift(k)
            ## compute the actual kernel vector
            gaussian_kernel_vector = gaussian(k)
            ## blow up the vector to an array of shape (1,.,1,len(nk),1,.,1)
            blown_up_shape = [1,]*len(nx)
            blown_up_shape[i] = len(gaussian_kernel_vector)
            gaussian_kernel_vector =\
                gaussian_kernel_vector.reshape(blown_up_shape)
            ## apply the blown-up gaussian_kernel_vector
            x *= gaussian_kernel_vector
        x.hermitian = remeber_hermitianQ
        
        return x
        

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Computes the power of an array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values of which the power is to be
                calculated.

            Returns
            -------
            spec : numpy.ndarray
                Power contained in the input array.

            Other parameters
            ----------------
            pindex : numpy.ndarray, *optional*
                Indexing array assigning the input array components to
                components of the power spectrum (default: None).
            rho : numpy.ndarray, *optional*
                Number of degrees of freedom per band (default: None).
            codomain : nifty.space, *optional*
                A compatible codomain for power indexing (default: None).
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        x = self.cast(x)

        ## If self is a position space, delegate calc_power to its codomain.
        if self.fourier == False:
            try:
                codomain = kwargs.get('codomain')
            except(KeyError):
                codomain = self.get_codomain()
                
            y = self.calc_transform(x, codomain)
            kwargs.update({'codomain': self})
            return codomain.calc_power(y, **kwargs)
            
        ## If some of the pindex, kindex or rho arrays are given explicitly,
        ## favor them over those from the self.power_indices dictionary.
        ## As the default value in kwargs.get(key, default) does NOT evaluate
        ## lazy, a distinction of cases is necessary. Otherwise the 
        ## powerindices might be computed, although not necessary
        if kwargs.has_key('pindex') and kwargs.has_key('kindex') and\
                kwargs.has_key('rho'):
            pindex = kwargs.get('pindex')
            rho = kwargs.get('rho')
        else:
            log = kwargs.get('log', None)
            nbin = kwargs.get('nbin', None)
            binbounds = kwargs.get('binbounds', None)            
            power_indices = self.power_indices.get_index_dict(log = log,
                                                              nbin = nbin,
                                                        binbounds = binbounds)            
            pindex = kwargs.get('pindex', power_indices['pindex'])
            rho = kwargs.get('rho', power_indices['rho'])
        
        fieldabs = abs(x)**2
        power_spectrum = np.zeros(rho.shape) 
        """
        ##TODO: Replace this super slow ndindex solution
        for ii in np.ndindex(pindex.shape):
            power_spectrum[pindex[ii]] += fieldabs[ii]
        """
                
        ## In order to make the summation over identical pindices fast, 
        ## the pindex and the kindex must have the same distribution strategy
        if pindex.distribution_strategy == fieldabs.distribution_strategy and\
            pindex.distributor.comm == fieldabs.distributor.comm:
            working_field = fieldabs
        else:
            working_field = pindex.copy_empty(dtype = fieldabs.dtype)
            working_field.inject((slice(None),), fieldabs, (slice(None,)))
        
        local_power_spectrum = np.bincount(pindex.get_local_data().flatten(),
                        weights = working_field.get_local_data().flatten())        
        power_spectrum =\
            pindex.distributor._allgather(local_power_spectrum)
        power_spectrum = np.sum(power_spectrum, axis = 0)
                    
        """
        ## Iterate over the k-vectors, extract those fieldabs, where the pindex
        ## has the according value and build the sum of the resulting array    
        
        power_spectrum = np.zeros(rho.size, dtype = np.float)
        
        
        for ii in xrange(rho.size):
            ## extract those fieldabs where the pindex equals the current ii
            extracted_fieldabs = working_field[pindex == ii]            
            ## sum the extracted field values up and store them
            power_spectrum[ii] = np.sum(extracted_fieldabs)
        """
        ## Divide out the degeneracy factor        
        power_spectrum /= rho
        return power_spectrum
        

        
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,power=None,unit="",norm=None,cmap=None,cbar=True,other=None,legend=False,mono=True,**kwargs):
        """
            Creates a plot of field values according to the specifications
            given by the parameters.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values.

            Returns
            -------
            None

            Other parameters
            ----------------
            title : string, *optional*
                Title of the plot (default: "").
            vmin : float, *optional*
                Minimum value to be displayed (default: ``min(x)``).
            vmax : float, *optional*
                Maximum value to be displayed (default: ``max(x)``).
            power : bool, *optional*
                Whether to plot the power contained in the field or the field
                values themselves (default: False).
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            other : {single object, tuple of objects}, *optional*
                Object or tuple of objects to be added, where objects can be
                scalars, arrays, or fields (default: None).
            legend : bool, *optional*
                Whether to show the legend or not (default: False).
            mono : bool, *optional*
                Whether to plot the monopole or not (default: True).
            save : string, *optional*
                Valid file name where the figure is to be stored, by default
                the figure is not saved (default: False).
            error : {float, numpy.ndarray, nifty.field}, *optional*
                Object indicating some confidence interval to be plotted
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale corresponding to each band in the power spectrum
                (default: None).
            codomain : nifty.space, *optional*
                A compatible codomain for power indexing (default: None).
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        if(not pl.isinteractive())and(not bool(kwargs.get("save",False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        naxes = (np.size(self.para)-1)//2
        if(power is None):
            power = bool(self.para[naxes])

        if(power):
            x = self.calc_power(x,**kwargs)

            fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

            ## explicit kindex
            xaxes = kwargs.get("kindex",None)
            ## implicit kindex
            if(xaxes is None):
                try:
                    self.set_power_indices(**kwargs)
                except:
                    codomain = kwargs.get("codomain",self.get_codomain())
                    codomain.set_power_indices(**kwargs)
                    xaxes = codomain.power_indices.get("kindex")
                else:
                    xaxes = self.power_indices.get("kindex")

            if(norm is None)or(not isinstance(norm,int)):
                norm = naxes
            if(vmin is None):
                vmin = np.min(x[:mono].tolist()+(xaxes**norm*x)[1:].tolist(),axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist()+(xaxes**norm*x)[1:].tolist(),axis=None,out=None)
            ax0.loglog(xaxes[1:],(xaxes**norm*x)[1:],color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
            if(mono):
                ax0.scatter(0.5*(xaxes[1]+xaxes[2]),x[0],s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=1)

            if(other is not None):
                if(isinstance(other,tuple)):
                    other = list(other)
                    for ii in xrange(len(other)):
                        if(isinstance(other[ii],field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii],size=np.size(xaxes),kindex=xaxes)
                elif(isinstance(other,field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other,size=np.size(xaxes),kindex=xaxes)]
                imax = max(1,len(other)-1)
                for ii in xrange(len(other)):
                    ax0.loglog(xaxes[1:],(xaxes**norm*other[ii])[1:],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5*(xaxes[1]+xaxes[2]),other[ii][0],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1],xaxes[-1])
            ax0.set_xlabel(r"$|k|$")
            ax0.set_ylim(vmin,vmax)
            ax0.set_ylabel(r"$|k|^{%i} P_k$"%norm)
            ax0.set_title(title)

        else:
            x = self.enforce_shape(np.array(x))

            if(naxes==1):
                fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
                ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

                xaxes = (np.arange(self.para[0],dtype=np.int)+self.para[2]*(self.para[0]//2))*self.vol
                if(vmin is None):
                    if(np.iscomplexobj(x)):
                        vmin = min(np.min(np.absolute(x),axis=None,out=None),np.min(np.real(x),axis=None,out=None),np.min(np.imag(x),axis=None,out=None))
                    else:
                        vmin = np.min(x,axis=None,out=None)
                if(vmax is None):
                    if(np.iscomplexobj(x)):
                        vmax = max(np.max(np.absolute(x),axis=None,out=None),np.max(np.real(x),axis=None,out=None),np.max(np.imag(x),axis=None,out=None))
                    else:
                        vmax = np.max(x,axis=None,out=None)
                if(norm=="log"):
                    ax0graph = ax0.semilogy
                    if(vmin<=0):
                        raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))
                else:
                    ax0graph = ax0.plot

                if(np.iscomplexobj(x)):
                    ax0graph(xaxes,np.absolute(x),color=[0.0,0.5,0.0],label="graph (absolute)",linestyle='-',linewidth=2.0,zorder=1)
                    ax0graph(xaxes,np.real(x),color=[0.0,0.5,0.0],label="graph (real part)",linestyle="--",linewidth=1.0,zorder=0)
                    ax0graph(xaxes,np.imag(x),color=[0.0,0.5,0.0],label="graph (imaginary part)",linestyle=':',linewidth=1.0,zorder=0)
                    if(legend):
                        ax0.legend()
                elif(other is not None):
                    ax0graph(xaxes,x,color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
                    if(isinstance(other,tuple)):
                        other = [self.enforce_values(xx,extend=True) for xx in other]
                    else:
                        other = [self.enforce_values(other,extend=True)]
                    imax = max(1,len(other)-1)
                    for ii in xrange(len(other)):
                        ax0graph(xaxes,other[ii],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if("error" in kwargs):
                        error = self.enforce_values(np.absolute(kwargs.get("error")),extend=True)
                        ax0.fill_between(xaxes,x-error,x+error,color=[0.8,0.8,0.8],label="error 0",zorder=-len(other))
                    if(legend):
                        ax0.legend()
                else:
                    ax0graph(xaxes,x,color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
                    if("error" in kwargs):
                        error = self.enforce_values(np.absolute(kwargs.get("error")),extend=True)
                        ax0.fill_between(xaxes,x-error,x+error,color=[0.8,0.8,0.8],label="error 0",zorder=0)

                ax0.set_xlim(xaxes[0],xaxes[-1])
                ax0.set_xlabel("coordinate")
                ax0.set_ylim(vmin,vmax)
                if(unit):
                    unit = " ["+unit+"]"
                ax0.set_ylabel("values"+unit)
                ax0.set_title(title)

            elif(naxes==2):
                if(np.iscomplexobj(x)):
                    about.infos.cprint("INFO: absolute values and phases are plotted.")
                    if(title):
                        title += " "
                    if(bool(kwargs.get("save",False))):
                        save_ = os.path.splitext(os.path.basename(str(kwargs.get("save"))))
                        kwargs.update(save=save_[0]+"_absolute"+save_[1])
                    self.get_plot(np.absolute(x),title=title+"(absolute)",vmin=vmin,vmax=vmax,power=False,unit=unit,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
#                    self.get_plot(np.real(x),title=title+"(real part)",vmin=vmin,vmax=vmax,power=False,unit=unit,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
#                    self.get_plot(np.imag(x),title=title+"(imaginary part)",vmin=vmin,vmax=vmax,power=False,unit=unit,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
                    if(unit):
                        unit = "rad"
                    if(cmap is None):
                        cmap = pl.cm.hsv_r
                    if(bool(kwargs.get("save",False))):
                        kwargs.update(save=save_[0]+"_phase"+save_[1])
                    self.get_plot(np.angle(x,deg=False),title=title+"(phase)",vmin=-3.1416,vmax=3.1416,power=False,unit=unit,norm=None,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs) ## values in [-pi,pi]
                    return None ## leave method
                else:
                    if(vmin is None):
                        vmin = np.min(x,axis=None,out=None)
                    if(vmax is None):
                        vmax = np.max(x,axis=None,out=None)
                    if(norm=="log")and(vmin<=0):
                        raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

                    s_ = np.array([self.para[1]*self.vol[1]/np.max(self.para[:naxes]*self.vol,axis=None,out=None),self.para[0]*self.vol[0]/np.max(self.para[:naxes]*self.vol,axis=None,out=None)*(1.0+0.159*bool(cbar))])
                    fig = pl.figure(num=None,figsize=(6.4*s_[0],6.4*s_[1]),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
                    ax0 = fig.add_axes([0.06/s_[0],0.06/s_[1],1.0-0.12/s_[0],1.0-0.12/s_[1]])

                    xaxes = (np.arange(self.para[1]+1,dtype=np.int)-0.5+self.para[4]*(self.para[1]//2))*self.vol[1]
                    yaxes = (np.arange(self.para[0]+1,dtype=np.int)-0.5+self.para[3]*(self.para[0]//2))*self.vol[0]
                    if(norm=="log"):
                        n_ = ln(vmin=vmin,vmax=vmax)
                    else:
                        n_ = None
                    sub = ax0.pcolormesh(xaxes,yaxes,x,cmap=cmap,norm=n_,vmin=vmin,vmax=vmax)
                    ax0.set_xlim(xaxes[0],xaxes[-1])
                    ax0.set_xticks([0],minor=False)
                    ax0.set_ylim(yaxes[0],yaxes[-1])
                    ax0.set_yticks([0],minor=False)
                    ax0.set_aspect("equal")
                    if(cbar):
                        if(norm=="log"):
                            f_ = lf(10,labelOnlyBase=False)
                            b_ = sub.norm.inverse(np.linspace(0,1,sub.cmap.N+1))
                            v_ = np.linspace(sub.norm.vmin,sub.norm.vmax,sub.cmap.N)
                        else:
                            f_ = None
                            b_ = None
                            v_ = None
                        cb0 = fig.colorbar(sub,ax=ax0,orientation="horizontal",fraction=0.1,pad=0.05,shrink=0.75,aspect=20,ticks=[vmin,vmax],format=f_,drawedges=False,boundaries=b_,values=v_)
                        cb0.ax.text(0.5,-1.0,unit,fontdict=None,withdash=False,transform=cb0.ax.transAxes,horizontalalignment="center",verticalalignment="center")
                    ax0.set_title(title)

            else:
                raise ValueError(about._errors.cstring("ERROR: unsupported number of axes ( "+str(naxes)+" > 2 )."))

        if(bool(kwargs.get("save",False))):
            fig.savefig(str(kwargs.get("save")),dpi=None,facecolor="none",edgecolor="none",orientation="portrait",papertype=None,format=None,transparent=False,bbox_inches=None,pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_rg.rg_space>"

    def __str__(self):
        naxes = (np.size(self.para)-1)//2
        num = self.para[:naxes].tolist()
        zerocenter = self.para[-naxes:].astype(np.bool).tolist()
        dist = self.vol.tolist()
        return "nifty_rg.rg_space instance\n- num        = " + str(num) + \
                "\n- naxes      = " + str(naxes) + \
                "\n- hermitian  = " + str(bool(self.para[naxes]==1)) + \
                "\n- purelyreal = " + str(bool(not self.para[naxes])) + \
                "\n- zerocenter = " + str(zerocenter) + \
                "\n- dist       = " + str(dist) + \
                "\n- fourier    = " + str(self.fourier)
    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ## __identiftier__ returns an object which contains all information needed 
    ## to uniquely identify a space. It returns a (immutable) tuple which 
    ## therefore can be compared. 
    ## The rg_space version of __identifier__ filters out the vars-information
    ## which is describing the rg_space's structure
    def _identifier(self):
        ## Extract the identifying parts from the vars(self) dict.
        temp = [(ii[0], 
            ((lambda x: tuple(x) if isinstance(x,np.ndarray) else x)(ii[1])))\
            for ii in vars(self).iteritems()\
            if ii[0] not in ["fft_machine","power_indices"]]
        ## Return the sorted identifiers as a tuple.
        return tuple(sorted(temp))

##-----------------------------------------------------------------------------


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
        self.shape = np.array(shape, dtype = int)
        self.dgrid = np.abs(np.array(dgrid))
        if self.shape.shape != self.dgrid.shape:
            raise ValueError(about._errors.cstring("ERROR: The supplied shape\
                and dgrid have not the same dimensionality"))         
        self.zerocentered = self.__cast_zerocentered__(zerocentered)

        ## Compute the global kdict
        self.kdict = self.compute_kdict()
        
        
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
        
    
    def compute_kdict(self):
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
    
#    def compute_klength(self, kdict):
#        local_klength = np.sort(list(set(kdict.get_local_data().flatten())))
#        
#        global_klength = kdict.distributor._allgather(local_klength)
#        global_klength = np.array(global_klength).flatten()
#        global_klength = np.sort(list(set(global_klength)))
#
#        return global_klength


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
        ## Prepare the local pindex data in order to count the degeneracy 
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
    
    def __compute_kdict_from_pindex_kindex__(self, pindex, kindex):
        tempindex = pindex.copy(dtype=kindex.dtype.type)        
        return tempindex.apply_scalar_function(lambda x: kindex[x])

    def __compute_index_dict__(self, config_dict):
        """
            Internal helper function which takes a config_dict, asks for the 
            pindex/kindex/rho/pundex set, and bins them according to the config
        """        
        ## if no binning is requested, compute the indices, build the dict, 
        ## and return it straight.        
        if config_dict["log"]==False and config_dict["nbin"]==None and \
          config_dict["binbounds"]==None:
            (temp_pindex, temp_kindex, temp_rho, temp_pundex) =\
                                        self.__compute_indices__(self.kdict)
            temp_kdict = self.kdict
            
        ## if binning is required, make a recursive call to get the unbinned
        ## indices, bin them, compute the pundex and then return everything.
        else:
            ## Get the unbinned indices 
            temp_unbinned_indices = self.get_index_dict(store=False)            
            ## Bin them            
            (temp_pindex, temp_kindex, temp_rho, temp_pundex) = \
                self.__bin_power_indices__(temp_unbinned_indices, **config_dict)
            ## Make a binned version of kdict
            temp_kdict = self.__compute_kdict_from_pindex_kindex__(temp_pindex, 
                                                                   temp_kindex)
            
        temp_index_dict = {"config": config_dict, 
                               "pindex": temp_pindex,
                               "kindex": temp_kindex,
                               "rho": temp_rho,
                               "pundex": temp_pundex,
                               "kdict": temp_kdict}
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

class utilities(object):
    def __init__(self):
        pass

    @staticmethod
    def hermitianize(x):
        ## make the point inversions
        flipped_x = utilities._hermitianize_inverter(x)
        flipped_x = flipped_x.conjugate()
        ## average x and flipped_x. 
        ## Correct the variance by multiplying sqrt(0.5)
        x = (x + flipped_x) * np.sqrt(0.5)
        ## The fixed points of the point inversion must not be avaraged.
        ## Hence one must multiply them again with sqrt(0.5)
        ## -> Get the middle index of the array
        mid_index = np.array(x.shape, dtype=np.int)//2
        dimensions = mid_index.size
        ## Use ndindex to iterate over all combinations of zeros and the
        ## mid_index in order to correct all fixed points.
        for i in np.ndindex((2,)*dimensions):
            temp_index = tuple(i*mid_index)
            x[temp_index] *= np.sqrt(0.5)
        try:
            x.hermitian = True
        except(AttributeError):
            pass
            
        return x
    
    @staticmethod
    def _hermitianize_inverter(x):
        ## calculate the number of dimensions the input array has
        dimensions = len(x.shape)
        ## prepare the slicing object which will be used for mirroring
        slice_primitive = [slice(None),]*dimensions
        ## copy the input data
        y = x.copy()
        ## flip in every direction
        for i in xrange(dimensions):
            slice_picker = slice_primitive[:]
            slice_picker[i] = slice(1, None)
            slice_inverter = slice_primitive[:]
            slice_inverter[i] = slice(None, 0, -1)
            y[slice_picker] = y[slice_inverter]
        return y

