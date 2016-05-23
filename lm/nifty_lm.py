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
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  lm
    ..                               /______/

    NIFTY submodule for grids on the two-sphere.

"""
from __future__ import division
#from nifty import *
import os
import numpy as np
import pylab as pl
from matplotlib.colors import LogNorm as ln
from matplotlib.ticker import LogFormatter as lf
from nifty import pi,                                                        \
                  about,                                                     \
                  random,                                                    \
                  space,                                                     \
                  field
#import libsharp_wrapper_gl as gl
try:
    import libsharp_wrapper_gl as gl
except(ImportError):
    about.infos.cprint("INFO: libsharp_wrapper_gl not available.")
    _gl_available = False
    about.warnings.cprint("WARNING: global setting 'about.lm2gl' corrected.")
    about.lm2gl.off()
else:
    _gl_available = True
#import healpy as hp
try:
    import healpy as hp
except(ImportError):
    about.infos.cprint("INFO: healpy not available.")
    _hp_available = False
else:
    _hp_available = True


##-----------------------------------------------------------------------------

class lm_space(space):
    """
        ..       __
        ..     /  /
        ..    /  /    __ ____ ___
        ..   /  /   /   _    _   |
        ..  /  /_  /  / /  / /  /
        ..  \___/ /__/ /__/ /__/  space class

        NIFTY subclass for spherical harmonics components, for representations
        of fields on the two-sphere.

        Parameters
        ----------
        lmax : int
            Maximum :math:`\ell`-value up to which the spherical harmonics
            coefficients are to be used.
        mmax : int, *optional*
            Maximum :math:`m`-value up to which the spherical harmonics
            coefficients are to be used (default: `lmax`).
        datatype : numpy.dtype, *optional*
            Data type of the field values (default: numpy.complex128).

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        gl_space : A class for the Gauss-Legendre discretization of the
            sphere [#]_.

        Notes
        -----
        Hermitian symmetry, i.e. :math:`a_{\ell -m} = \overline{a}_{\ell m}` is
        always assumed for the spherical harmonics components, i.e. only fields
        on the two-sphere with real-valued representations in position space
        can be handled.

        References
        ----------
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing the two numbers `lmax` and
            `mmax`.
        datatype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Parameter captioning the fact that an :py:class:`lm_space` is
            always discrete.
        vol : numpy.ndarray
            Pixel volume of the :py:class:`lm_space`, which is always 1.
    """
    def __init__(self,lmax,mmax=None,datatype=None):
        """
            Sets the attributes for an lm_space class instance.

            Parameters
            ----------
            lmax : int
                Maximum :math:`\ell`-value up to which the spherical harmonics
                coefficients are to be used.
            mmax : int, *optional*
                Maximum :math:`m`-value up to which the spherical harmonics
                coefficients are to be used (default: `lmax`).
            datatype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.complex128).

            Returns
            -------
            None.

            Raises
            ------
            ImportError
                If neither the libsharp_wrapper_gl nor the healpy module are
                available.
            ValueError
                If input `nside` is invaild.

        """
        ## check imports
        if(not _gl_available)and(not _hp_available):
            raise ImportError(about._errors.cstring("ERROR: neither libsharp_wrapper_gl nor healpy available."))
        ## check parameters
        if(lmax<1):
            raise ValueError(about._errors.cstring("ERROR: nonpositive number."))
        if(lmax%2==0)and(lmax>2): ## exception lmax == 2 (nside == 1)
            about.warnings.cprint("WARNING: unrecommended parameter ( lmax <> 2*n+1 ).")
        if(mmax is None):
            mmax = lmax
        elif(mmax<1)or(mmax>lmax):
            about.warnings.cprint("WARNING: parameter set to default.")
            mmax = lmax
        if(mmax!=lmax):
            about.warnings.cprint("WARNING: unrecommended parameter ( mmax <> lmax ).")
        self.para = np.array([lmax,mmax],dtype=np.int)

        ## check data type
        if(datatype is None):
            datatype = np.complex128
        elif(datatype not in [np.complex64,np.complex128]):
            about.warnings.cprint("WARNING: data type set to default.")
            datatype = np.complex128
        self.datatype = datatype

        self.discrete = True
        self.vol = np.real(np.array([1],dtype=self.datatype))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def lmax(self):
        """
            Returns the maximum quantum number :math:`\ell`.

            Returns
            -------
            lmax : int
                Maximum quantum number :math:`\ell`.
        """
        return self.para[0]

    def mmax(self):
        """
            Returns the maximum quantum number :math:`m`.

            Returns
            -------
            mmax : int
                Maximum quantum number :math:`m`.

        """
        return self.para[1]

    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of spherical
            harmonics components that are stored.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension as an array with one component
                or as a scalar (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Number of spherical harmonics components.

            Notes
            -----
            Due to the symmetry assumption, only the components with
            non-negative :math:`m` are stored and only these components are
            counted here.
        """
        ## dim = (mmax+1)*(lmax-mmax/2+1)
        if(split):
            return np.array([(self.para[0]+1)*(self.para[1]+1)-(self.para[1]+1)*self.para[1]//2],dtype=np.int)
        else:
            return (self.para[0]+1)*(self.para[1]+1)-(self.para[1]+1)*self.para[1]//2

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space, taking into
            account symmetry constraints and complex-valuedness.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.

            Notes
            -----
            The number of degrees of freedom is reduced due to the hermitian
            symmetry, which is assumed for the spherical harmonics components.
        """
        ## dof = 2*dim-(lmax+1) = (lmax+1)*(2*mmax+1)*(mmax+1)*mmax
        return (self.para[0]+1)*(2*self.para[1]+1)-(self.para[1]+1)*self.para[1]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
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
        """
        if(isinstance(spec,field)):
            spec = spec.val.astype(dtype=self.datatype)
        elif(callable(spec)):
            try:
                spec = np.array(spec(np.arange(self.para[0]+1,dtype=self.vol.dtype)),dtype=self.datatype) ## prevent integer division
            except:
                raise TypeError(about._errors.cstring("ERROR: invalid power spectra function.")) ## exception in ``spec(kindex)``
        elif(np.isscalar(spec)):
            spec = np.array([spec],dtype=self.datatype)
        else:
            spec = np.array(spec,dtype=self.datatype)

        ## drop imaginary part
        spec = np.real(spec)
        ## check finiteness
        if(not np.all(np.isfinite(spec))):
            about.warnings.cprint("WARNING: infinite value(s).")
        ## check positivity (excluding null)
        if(np.any(spec<0)):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))
        elif(np.any(spec==0)):
            about.warnings.cprint("WARNING: nonpositive value(s).")

        size = self.para[0]+1 ## lmax+1
        ## extend
        if(np.size(spec)==1):
            spec = spec*np.ones(size,dtype=spec.dtype,order='C')
        ## size check
        elif(np.size(spec)<size):
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(spec))+" < "+str(size)+" )."))
        elif(np.size(spec)>size):
            about.warnings.cprint("WARNING: power spectrum cut to size ( == "+str(size)+" ).")
            spec = spec[:size]

        return spec

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _getlm(self): ## > compute all (l,m)
        index = np.arange(self.dim(split=False))
        n = 2*self.para[0]+1
        m = np.ceil((n-np.sqrt(n**2-8*(index-self.para[0])))/2).astype(np.int)
        l = index-self.para[0]*m+m*(m-1)//2
        return l,m

    def set_power_indices(self,**kwargs):
        """
            Sets the (un)indexing objects for spectral indexing internally.

            Parameters
            ----------
            None

            Returns
            -------
            None

            See Also
            --------
            get_power_indices

        """
        ## check storage
        if(not hasattr(self,"power_indices")):
            ## power indices
#            about.infos.cflush("INFO: setting power indices ...")
            kindex = np.arange(self.para[0]+1,dtype=np.int)
            rho = 2*kindex+1
            if(_hp_available): ## default
                pindex = hp.Alm.getlm(self.para[0],i=None)[0] ## l of (l,m)
            else:
                pindex = self._getlm()[0] ## l of (l,m)
            pundex = np.unique(pindex,return_index=True,return_inverse=False)[1]
            ## storage
            self.power_indices = {"kindex":kindex,"pindex":pindex,"pundex":pundex,"rho":rho} ## alphabetical
#            about.infos.cprint(" done.")

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_values(self,x,extend=True):
        """
            Computes valid field values from a given object, taking into
            account data types, size, and hermitian symmetry.

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

        if(np.size(x)!=1)and(np.any(x.imag[:self.para[0]+1]!=0)):
            about.warnings.cprint("WARNING: forbidden values reset.")
            x.real[:self.para[0]+1] = np.absolute(x[:self.para[0]+1])*(np.sign(x.real[:self.para[0]+1])+(np.sign(x.real[:self.para[0]+1])==0)*np.sign(x.imag[:self.para[0]+1])).astype(np.int)
            x.imag[:self.para[0]+1] = 0 ## x.imag[l,m==0] = 0

        ## check finiteness
        if(not np.all(np.isfinite(x))):
            about.warnings.cprint("WARNING: infinite value(s).")

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters, taking into account complex-valuedness and
            hermitian symmetry.

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
            spec : {scalar, list, numpy.array, nifty.field, function}, *optional*
                Power spectrum (default: 1).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.arguments(self,**kwargs)

        if(arg is None):
            return np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="syn"):
            lmax = self.para[0] ## lmax
            if(self.datatype==np.complex64):
                if(_gl_available): ## default
                    x = gl.synalm_f(arg[1],lmax=lmax,mmax=lmax)
                else:
                    x = hp.synalm(arg[1].astype(np.complex128),lmax=lmax,mmax=lmax).astype(np.complex64) ## FIXME: `verbose` kwarg
            else:
                if(_hp_available): ## default
                    x = hp.synalm(arg[1],lmax=lmax,mmax=lmax) ## FIXME: `verbose` kwarg
                else:
                    x = gl.synalm(arg[1],lmax=lmax,mmax=lmax)

            return x

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        if(np.any(x.imag[:self.para[0]+1]!=0)):
            x.real[:self.para[0]+1] = np.absolute(x[:self.para[0]+1])*(np.sign(x.real[:self.para[0]+1])+(np.sign(x.real[:self.para[0]+1])==0)*np.sign(x.imag[:self.para[0]+1])).astype(np.int)
            x.imag[:self.para[0]+1] = 0 ## x.imag[l,m==0] = 0

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def check_codomain(self,codomain):
        """
            Checks whether a given codomain is compatible to the
            :py:class:`lm_space` or not.

            Parameters
            ----------
            codomain : nifty.space
                Space to be checked for compatibility.

            Returns
            -------
            check : bool
                Whether or not the given codomain is compatible to the space.

            Notes
            -----
            Compatible codomains are instances of :py:class:`lm_space`,
            :py:class:`gl_space`, and :py:class:`hp_space`.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        elif(isinstance(codomain,gl_space)):
            ##         lmax==mmax                         nlat==lmax+1                         nlon==2*lmax+1
            if(self.para[0]==self.para[1])and(codomain.para[0]==self.para[0]+1)and(codomain.para[1]==2*self.para[0]+1):
                return True
            else:
                about.warnings.cprint("WARNING: unrecommended codomain.")

        elif(isinstance(codomain,hp_space)):
            ##         lmax==mmax                        3*nside-1==lmax
            if(self.para[0]==self.para[1])and(3*codomain.para[0]-1==self.para[0]):
                return True
            else:
                about.warnings.cprint("WARNING: unrecommended codomain.")

        return False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,coname=None,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  a pixelization of the two-sphere.

            Parameters
            ----------
            coname : string, *optional*
                String specifying a desired codomain (default: None).

            Returns
            -------
            codomain : nifty.space
                A compatible codomain.

            Notes
            -----
            Possible arguments for `coname` are ``'gl'`` in which case a Gauss-
            Legendre pixelization [#]_ of the sphere is generated, and ``'hp'``
            in which case a HEALPix pixelization [#]_ is generated.

            References
            ----------
            .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
                   High-Resolution Discretization and Fast Analysis of Data
                   Distributed on the Sphere", *ApJ* 622..759G.
            .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
                   harmonic transforms revisited";
                   `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

        """
        if(coname=="gl")or(coname is None)and(about.lm2gl.status): ## order matters
            if(self.datatype==np.complex64):
                return gl_space(self.para[0]+1,nlon=2*self.para[0]+1,datatype=np.float32) ## nlat,nlon = lmax+1,2*lmax+1
            else:
                return gl_space(self.para[0]+1,nlon=2*self.para[0]+1,datatype=np.float64) ## nlat,nlon = lmax+1,2*lmax+1

        elif(coname=="hp")or(coname is None)and(not about.lm2gl.status):
            return hp_space((self.para[0]+1)//3) ## nside = (lmax+1)/3

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported or incompatible space '"+coname+"'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each field component (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the field components or the complete space.

            Notes
            -----
            The spherical harmonics components with :math:`m=0` have meta
            volume 1, the ones with :math:`m>0` have meta volume 2, sinnce they
            each determine another component with negative :math:`m`.
        """
        if(total):
            return self.dof()
        else:
            mol = np.ones(self.dim(split=True),dtype=self.vol.dtype,order='C')
            mol[self.para[0]+1:] = 2 ## redundant in (l,m) and (l,-m)
            return mol

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _dotlm(self,x,y): ## > compute inner product
        dot = np.sum(x.real[:self.para[0]+1]*y.real[:self.para[0]+1],axis=None,dtype=None,out=None)
        dot += 2*np.sum(x.real[self.para[0]+1:]*y.real[:self.para[0]+1:],axis=None,dtype=None,out=None)
        dot += 2*np.sum(x.imag[self.para[0]+1:]*y.imag[:self.para[0]+1:],axis=None,dtype=None,out=None)
        return dot

    def calc_dot(self,x,y):
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
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        y = self.enforce_shape(np.array(y,dtype=self.datatype))
        ## inner product
        if(_gl_available): ## default
            if(self.datatype==np.complex64):
                return gl.dotlm_f(x,y,lmax=self.para[0],mmax=self.para[1])
            else:
                return gl.dotlm(x,y,lmax=self.para[0],mmax=self.para[1])
        else:
            return self._dotlm(x,y)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,**kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                Target space to which the transformation shall map
                (default: self).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        self.check_codomain(codomain) ## a bit pointless

        if(self==codomain):
            return x ## T == id

        elif(isinstance(codomain,gl_space)):
            ## transform
            if(self.datatype==np.complex64):
                Tx = gl.alm2map_f(x,nlat=codomain.para[0],nlon=codomain.para[1],lmax=self.para[0],mmax=self.para[1],cl=False)
            else:
                Tx = gl.alm2map(x,nlat=codomain.para[0],nlon=codomain.para[1],lmax=self.para[0],mmax=self.para[1],cl=False)
            ## weight if discrete
            if(codomain.discrete):
                Tx = codomain.calc_weight(Tx,power=0.5)

        elif(isinstance(codomain,hp_space)):
            ## transform
            Tx =  hp.alm2map(x.astype(np.complex128),codomain.para[0],lmax=self.para[0],mmax=self.para[1],pixwin=False,fwhm=0.0,sigma=None,pol=True,inplace=False) ## FIXME: `verbose` kwarg
            ## weight if discrete
            if(codomain.discrete):
                Tx = codomain.calc_weight(Tx,power=0.5)

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        return Tx.astype(codomain.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
        """
            Smoothes an array of field values by convolution with a Gaussian
            kernel in position space.

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
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## check sigma
        if(sigma==0):
            return x
        elif(sigma==-1):
            about.infos.cprint("INFO: invalid sigma reset.")
            sigma = 4.5/(self.para[0]+1) ## sqrt(2)*pi/(lmax+1)
        elif(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        ## smooth
        if(_hp_available): ## default
            return hp.smoothalm(x,fwhm=0.0,sigma=sigma,pol=True,mmax=self.para[1],verbose=False,inplace=False) ## no overwrite
        else:
            return gl.smoothalm(x,lmax=self.para[0],mmax=self.para[1],fwhm=0.0,sigma=sigma,overwrite=False) ## no overwrite


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
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## power spectrum
        if(self.datatype==np.complex64):
            if(_gl_available): ## default
                return gl.anaalm_f(x,lmax=self.para[0],mmax=self.para[1])
            else:
                return hp.alm2cl(x.astype(np.complex128),alms2=None,lmax=self.para[0],mmax=self.para[1],lmax_out=self.para[0],nspec=None).astype(np.float32)
        else:
            if(_hp_available): ## default
                return hp.alm2cl(x,alms2=None,lmax=self.para[0],mmax=self.para[1],lmax_out=self.para[0],nspec=None)
            else:
                return gl.anaalm(x,lmax=self.para[0],mmax=self.para[1])


    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,power=True,norm=None,cmap=None,cbar=True,other=None,legend=False,mono=True,**kwargs):
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
                values themselves (default: True).
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

        """
        if(not pl.isinteractive())and(not bool(kwargs.get("save",False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(power):
            x = self.calc_power(x)

            fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

            xaxes = np.arange(self.para[0]+1,dtype=np.int)
            if(vmin is None):
                vmin = np.min(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*x)[1:],color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
            if(mono):
                ax0.scatter(0.5*(xaxes[1]+xaxes[2]),x[0],s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=1)

            if(other is not None):
                if(isinstance(other,tuple)):
                    other = list(other)
                    for ii in xrange(len(other)):
                        if(isinstance(other[ii],field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii])
                elif(isinstance(other,field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other)]
                imax = max(1,len(other)-1)
                for ii in xrange(len(other)):
                    ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*other[ii])[1:],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5*(xaxes[1]+xaxes[2]),other[ii][0],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1],xaxes[-1])
            ax0.set_xlabel(r"$\ell$")
            ax0.set_ylim(vmin,vmax)
            ax0.set_ylabel(r"$\ell(2\ell+1) C_\ell$")
            ax0.set_title(title)

        else:
            x = self.enforce_shape(np.array(x))
            if(np.iscomplexobj(x)):
                if(title):
                    title += " "
                if(bool(kwargs.get("save",False))):
                    save_ = os.path.splitext(os.path.basename(str(kwargs.get("save"))))
                    kwargs.update(save=save_[0]+"_absolute"+save_[1])
                self.get_plot(np.absolute(x),title=title+"(absolute)",vmin=vmin,vmax=vmax,power=False,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
#                self.get_plot(np.real(x),title=title+"(real part)",vmin=vmin,vmax=vmax,power=False,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
#                self.get_plot(np.imag(x),title=title+"(imaginary part)",vmin=vmin,vmax=vmax,power=False,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
                if(cmap is None):
                    cmap = pl.cm.hsv_r
                if(bool(kwargs.get("save",False))):
                    kwargs.update(save=save_[0]+"_phase"+save_[1])
                self.get_plot(np.angle(x,deg=False),title=title+"(phase)",vmin=-3.1416,vmax=3.1416,power=False,norm=None,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs) ## values in [-pi,pi]
                return None ## leave method
            else:
                if(vmin is None):
                    vmin = np.min(x,axis=None,out=None)
                if(vmax is None):
                    vmax = np.max(x,axis=None,out=None)
                if(norm=="log")and(vmin<=0):
                    raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

                xmesh = np.nan*np.empty(self.para[::-1]+1,dtype=np.float16,order='C') ## not a number
                xmesh[4,1] = None
                xmesh[1,4] = None
                lm = 0
                for mm in xrange(self.para[1]+1):
                    xmesh[mm][mm:] = x[lm:lm+self.para[0]+1-mm]
                    lm += self.para[0]+1-mm

                s_ = np.array([1,self.para[1]/self.para[0]*(1.0+0.159*bool(cbar))])
                fig = pl.figure(num=None,figsize=(6.4*s_[0],6.4*s_[1]),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
                ax0 = fig.add_axes([0.06/s_[0],0.06/s_[1],1.0-0.12/s_[0],1.0-0.12/s_[1]])
                ax0.set_axis_bgcolor([0.0,0.0,0.0,0.0])

                xaxes = np.arange(self.para[0]+2,dtype=np.int)-0.5
                yaxes = np.arange(self.para[1]+2,dtype=np.int)-0.5
                if(norm=="log"):
                    n_ = ln(vmin=vmin,vmax=vmax)
                else:
                    n_ = None
                sub = ax0.pcolormesh(xaxes,yaxes,np.ma.masked_where(np.isnan(xmesh),xmesh),cmap=cmap,norm=n_,vmin=vmin,vmax=vmax,clim=(vmin,vmax))
                ax0.set_xlim(xaxes[0],xaxes[-1])
                ax0.set_xticks([0],minor=False)
                ax0.set_xlabel(r"$\ell$")
                ax0.set_ylim(yaxes[0],yaxes[-1])
                ax0.set_yticks([0],minor=False)
                ax0.set_ylabel(r"$m$")
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
                    fig.colorbar(sub,ax=ax0,orientation="horizontal",fraction=0.1,pad=0.05,shrink=0.75,aspect=20,ticks=[vmin,vmax],format=f_,drawedges=False,boundaries=b_,values=v_)
                ax0.set_title(title)

        if(bool(kwargs.get("save",False))):
            fig.savefig(str(kwargs.get("save")),dpi=None,facecolor="none",edgecolor="none",orientation="portrait",papertype=None,format=None,transparent=False,bbox_inches=None,pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_lm.lm_space>"

    def __str__(self):
        return "nifty_lm.lm_space instance\n- lmax     = "+str(self.para[0])+"\n- mmax     = "+str(self.para[1])+"\n- datatype = numpy."+str(np.result_type(self.datatype))

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class gl_space(space):
    """
        ..                 __
        ..               /  /
        ..     ____ __  /  /
        ..   /   _   / /  /
        ..  /  /_/  / /  /_
        ..  \___   /  \___/  space class
        .. /______/

        NIFTY subclass for Gauss-Legendre pixelizations [#]_ of the two-sphere.

        Parameters
        ----------
        nlat : int
            Number of latitudinal bins, or rings.
        nlon : int, *optional*
            Number of longitudinal bins (default: ``2*nlat - 1``).
        datatype : numpy.dtype, *optional*
            Data type of the field values (default: numpy.float64).

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        lm_space : A class for spherical harmonic components.

        Notes
        -----
        Only real-valued fields on the two-sphere are supported, i.e.
        `datatype` has to be either numpy.float64 or numpy.float32.

        References
        ----------
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing the two numbers `nlat` and `nlon`.
        datatype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for spherical spaces.
        vol : numpy.ndarray
            An array containing the pixel sizes.
    """
    def __init__(self,nlat,nlon=None,datatype=None):
        """
            Sets the attributes for a gl_space class instance.

            Parameters
            ----------
            nlat : int
                Number of latitudinal bins, or rings.
            nlon : int, *optional*
                Number of longitudinal bins (default: ``2*nlat - 1``).
            datatype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.float64).

            Returns
            -------
            None

            Raises
            ------
            ImportError
                If the libsharp_wrapper_gl module is not available.
            ValueError
                If input `nlat` is invaild.

        """
        ## check imports
        if(not _gl_available):
            raise ImportError(about._errors.cstring("ERROR: libsharp_wrapper_gl not available."))
        ## check parameters
        if(nlat<1):
            raise ValueError(about._errors.cstring("ERROR: nonpositive number."))
        if(nlat%2!=0):
            raise ValueError(about._errors.cstring("ERROR: invalid parameter ( nlat <> 2*n )."))
        if(nlon is None):
            nlon = 2*nlat-1
        elif(nlon<1):
            about.warnings.cprint("WARNING: parameter set to default.")
            nlon = 2*nlat-1
        if(nlon!=2*nlat-1):
            about.warnings.cprint("WARNING: unrecommended parameter ( nlon <> 2*nlat-1 ).")
        self.para = np.array([nlat,nlon],dtype=np.int)

        ## check data type
        if(datatype is None):
            datatype = np.float64
        elif(datatype not in [np.float32,np.float64]):
            about.warnings.cprint("WARNING: data type set to default.")
            datatype = np.float64
        self.datatype = datatype

        self.discrete = False
        self.vol = gl.vol(self.para[0],nlon=self.para[1]).astype(self.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def nlat(self):
        """
            Returns the number of latitudinal bins.

            Returns
            -------
            nlat : int
                Number of latitudinal bins, or rings.
        """
        return self.para[0]

    def nlon(self):
        """
            Returns the number of longitudinal bins.

            Returns
            -------
            nlon : int
                Number of longitudinal bins.
        """
        return self.para[1]

    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of pixels.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension as an array with one component
                or as a scalar (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension of the space.
        """
        ## dim = nlat*nlon
        if(split):
            return np.array([self.para[0]*self.para[1]],dtype=np.int)
        else:
            return self.para[0]*self.para[1]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.

            Notes
            -----
            Since the :py:class:`gl_space` class only supports real-valued
            fields, the number of degrees of freedom is the number of pixels.
        """
        ## dof = dim
        return self.para[0]*self.para[1]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
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
        """
        if(isinstance(spec,field)):
            spec = spec.val.astype(self.datatype)
        elif(callable(spec)):
            try:
                spec = np.array(spec(np.arange(self.para[0],dtype=np.int)),dtype=self.datatype)
            except:
                raise TypeError(about._errors.cstring("ERROR: invalid power spectra function.")) ## exception in ``spec(kindex)``
        elif(np.isscalar(spec)):
            spec = np.array([spec],dtype=self.datatype)
        else:
            spec = np.array(spec,dtype=self.datatype)

        ## check finiteness
        if(not np.all(np.isfinite(spec))):
            about.warnings.cprint("WARNING: infinite value(s).")
        ## check positivity (excluding null)
        if(np.any(spec<0)):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))
        elif(np.any(spec==0)):
            about.warnings.cprint("WARNING: nonpositive value(s).")

        size = self.para[0] ## nlat
        ## extend
        if(np.size(spec)==1):
            spec = spec*np.ones(size,dtype=spec.dtype,order='C')
        ## size check
        elif(np.size(spec)<size):
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(spec))+" < "+str(size)+" )."))
        elif(np.size(spec)>size):
            about.warnings.cprint("WARNING: power spectrum cut to size ( == "+str(size)+" ).")
            spec = spec[:size]

        return spec

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_power_indices(self,**kwargs):
        """
            Raises
            ------
            AttributeError
                Always. -- The power spectrum for a field on the sphere
            is defined by its spherical harmonics components and not its
            position space representation.

        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra indexing ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters.

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
            spec : {scalar, list, numpy.array, nifty.field, function}, *optional*
                Power spectrum (default: 1).
            codomain : nifty.lm_space, *optional*
                A compatible codomain for power indexing (default: None).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.arguments(self,**kwargs)

        if(arg is None):
            x = np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="syn"):
            lmax = self.para[0]-1 ## nlat-1
            if(self.datatype==np.float32):
                x = gl.synfast_f(arg[1],nlat=self.para[0],nlon=self.para[1],lmax=lmax,mmax=lmax,alm=False)
            else:
                x = gl.synfast(arg[1],nlat=self.para[0],nlon=self.para[1],lmax=lmax,mmax=lmax,alm=False)
            ## weight if discrete
            if(self.discrete):
                x = self.calc_weight(x,power=0.5)

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        return x

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

            Notes
            -----
            Compatible codomains are instances of :py:class:`gl_space` and
            :py:class:`lm_space`.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        if(isinstance(codomain,lm_space)):
            ##         nlon==2*lat-1                          lmax==nlat-1                         lmax==mmax
            if(self.para[1]==2*self.para[0]-1)and(codomain.para[0]==self.para[0]-1)and(codomain.para[0]==codomain.para[1]):
                return True
            else:
                about.warnings.cprint("WARNING: unrecommended codomain.")

        return False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Returns
            -------
            codomain : nifty.lm_space
                A compatible codomain.
        """
        if(self.datatype==np.float32):
            return lm_space(self.para[0]-1,mmax=self.para[0]-1,datatype=np.complex64) ## lmax,mmax = nlat-1,nlat-1
        else:
            return lm_space(self.para[0]-1,mmax=self.para[0]-1,datatype=np.complex128) ## lmax,mmax = nlat-1,nlat-1

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each field component (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the field components or the complete space.

            Notes
            -----
            For Gauss-Legendre pixelizations, the meta volumes are the pixel
            sizes.
        """
        if(total):
            return self.datatype(4*pi)
        else:
            mol = np.ones(self.dim(split=True),dtype=self.datatype,order='C')
            return self.calc_weight(mol,power=1)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_weight(self,x,power=1):
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
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## weight
        if(self.datatype==np.float32):
            return gl.weight_f(x,self.vol,p=np.float32(power),nlat=self.para[0],nlon=self.para[1],overwrite=False)
        else:
            return gl.weight(x,self.vol,p=np.float64(power),nlat=self.para[0],nlon=self.para[1],overwrite=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,**kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                Target space to which the transformation shall map
                (default: self).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array

            Notes
            -----
            Only instances of the :py:class:`lm_space` or :py:class:`gl_space`
            classes are allowed as `codomain`.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        self.check_codomain(codomain) ## a bit pointless

        if(self==codomain):
            return x ## T == id

        if(isinstance(codomain,lm_space)):
            ## weight if discrete
            if(self.discrete):
                x = self.calc_weight(x,power=-0.5)
            ## transform
            if(self.datatype==np.float32):
                Tx = gl.map2alm_f(x,nlat=self.para[0],nlon=self.para[1],lmax=codomain.para[0],mmax=codomain.para[1])
            else:
                Tx = gl.map2alm(x,nlat=self.para[0],nlon=self.para[1],lmax=codomain.para[0],mmax=codomain.para[1])

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        return Tx.astype(codomain.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
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
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## check sigma
        if(sigma==0):
            return x
        elif(sigma==-1):
            about.infos.cprint("INFO: invalid sigma reset.")
            sigma = 4.5/self.para[0] ## sqrt(2)*pi/(lmax+1)
        elif(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        ## smooth
        return gl.smoothmap(x,nlat=self.para[0],nlon=self.para[1],lmax=self.para[0]-1,mmax=self.para[0]-1,fwhm=0.0,sigma=sigma)

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
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## weight if discrete
        if(self.discrete):
            x = self.calc_weight(x,power=-0.5)
        ## power spectrum
        if(self.datatype==np.float32):
            return gl.anafast_f(x,nlat=self.para[0],nlon=self.para[1],lmax=self.para[0]-1,mmax=self.para[0]-1,alm=False)
        else:
            return gl.anafast(x,nlat=self.para[0],nlon=self.para[1],lmax=self.para[0]-1,mmax=self.para[0]-1,alm=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,power=False,unit="",norm=None,cmap=None,cbar=True,other=None,legend=False,mono=True,**kwargs):
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

        """
        if(not pl.isinteractive())and(not bool(kwargs.get("save",False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(power):
            x = self.calc_power(x)

            fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

            xaxes = np.arange(self.para[0],dtype=np.int)
            if(vmin is None):
                vmin = np.min(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*x)[1:],color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
            if(mono):
                ax0.scatter(0.5*(xaxes[1]+xaxes[2]),x[0],s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=1)

            if(other is not None):
                if(isinstance(other,tuple)):
                    other = list(other)
                    for ii in xrange(len(other)):
                        if(isinstance(other[ii],field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii])
                elif(isinstance(other,field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other)]
                imax = max(1,len(other)-1)
                for ii in xrange(len(other)):
                    ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*other[ii])[1:],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5*(xaxes[1]+xaxes[2]),other[ii][0],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1],xaxes[-1])
            ax0.set_xlabel(r"$l$")
            ax0.set_ylim(vmin,vmax)
            ax0.set_ylabel(r"$l(2l+1) C_l$")
            ax0.set_title(title)

        else:
            x = self.enforce_shape(np.array(x,dtype=self.datatype))
            if(vmin is None):
                vmin = np.min(x,axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x,axis=None,out=None)
            if(norm=="log")and(vmin<=0):
                raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

            fig = pl.figure(num=None,figsize=(8.5,5.4),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.02,0.05,0.96,0.9])

            lon,lat = gl.bounds(self.para[0],nlon=self.para[1])
            lon = (lon-pi)*180/pi
            lat = (lat-pi/2)*180/pi
            if(norm=="log"):
                n_ = ln(vmin=vmin,vmax=vmax)
            else:
                n_ = None
            sub = ax0.pcolormesh(lon,lat,np.roll(x.reshape((self.para[0],self.para[1]),order='C'),self.para[1]//2,axis=1)[::-1,::-1],cmap=cmap,norm=n_,vmin=vmin,vmax=vmax)
            ax0.set_xlim(-180,180)
            ax0.set_ylim(-90,90)
            ax0.set_aspect("equal")
            ax0.axis("off")
            if(cbar):
                if(norm=="log"):
                    f_ = lf(10,labelOnlyBase=False)
                    b_ = sub.norm.inverse(np.linspace(0,1,sub.cmap.N+1))
                    v_ = np.linspace(sub.norm.vmin,sub.norm.vmax,sub.cmap.N)
                else:
                    f_ = None
                    b_ = None
                    v_ = None
                cb0 = fig.colorbar(sub,ax=ax0,orientation="horizontal",fraction=0.1,pad=0.05,shrink=0.5,aspect=25,ticks=[vmin,vmax],format=f_,drawedges=False,boundaries=b_,values=v_)
                cb0.ax.text(0.5,-1.0,unit,fontdict=None,withdash=False,transform=cb0.ax.transAxes,horizontalalignment="center",verticalalignment="center")
            ax0.set_title(title)

        if(bool(kwargs.get("save",False))):
            fig.savefig(str(kwargs.get("save")),dpi=None,facecolor="none",edgecolor="none",orientation="portrait",papertype=None,format=None,transparent=False,bbox_inches=None,pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_lm.gl_space>"

    def __str__(self):
        return "nifty_lm.gl_space instance\n- nlat     = "+str(self.para[0])+"\n- nlon     = "+str(self.para[1])+"\n- datatype = numpy."+str(np.result_type(self.datatype))

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class hp_space(space):
    """
        ..        __
        ..      /  /
        ..     /  /___    ______
        ..    /   _   | /   _   |
        ..   /  / /  / /  /_/  /
        ..  /__/ /__/ /   ____/  space class
        ..           /__/

        NIFTY subclass for HEALPix discretizations of the two-sphere [#]_.

        Parameters
        ----------
        nside : int
            Resolution parameter for the HEALPix discretization, resulting in
            ``12*nside**2`` pixels.

        See Also
        --------
        gl_space : A class for the Gauss-Legendre discretization of the
            sphere [#]_.
        lm_space : A class for spherical harmonic components.

        Notes
        -----
        Only powers of two are allowed for `nside`.

        References
        ----------
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

        Attributes
        ----------
        para : numpy.ndarray
            Array containing the number `nside`.
        datatype : numpy.dtype
            Data type of the field values, which is always numpy.float64.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for spherical spaces.
        vol : numpy.ndarray
            An array with one element containing the pixel size.
    """
    niter = 0 ## default number of iterations used for transformations

    def __init__(self,nside):
        """
            Sets the attributes for a hp_space class instance.

            Parameters
            ----------
            nside : int
                Resolution parameter for the HEALPix discretization, resulting
                in ``12*nside**2`` pixels.

            Returns
            -------
            None

            Raises
            ------
            ImportError
                If the healpy module is not available.
            ValueError
                If input `nside` is invaild.

        """
        ## check imports
        if(not _hp_available):
            raise ImportError(about._errors.cstring("ERROR: healpy not available."))
        ## check parameters
        if(nside<1):
            raise ValueError(about._errors.cstring("ERROR: nonpositive number."))
        if(not hp.isnsideok(nside)):
            raise ValueError(about._errors.cstring("ERROR: invalid parameter ( nside <> 2**n )."))
        self.para = np.array([nside],dtype=np.int)

        self.datatype = np.float64
        self.discrete = False
        self.vol = np.array([4*pi/(12*self.para[0]**2)],dtype=self.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def nside(self):
        """
            Returns the resolution parameter.

            Returns
            -------
            nside : int
                HEALPix resolution parameter.
        """
        return self.para[0]


    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of pixels.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension as an array with one component
                or as a scalar (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension of the space.
        """
        ## dim = 12*nside**2
        if(split):
            return np.array([12*self.para[0]**2],dtype=np.int)
        else:
            return 12*self.para[0]**2

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.

            Notes
            -----
            Since the :py:class:`hp_space` class only supports real-valued
            fields, the number of degrees of freedom is the number of pixels.
        """
        ## dof = dim
        return 12*self.para[0]**2

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
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
        """
        if(isinstance(spec,field)):
            spec = spec.val.astype(self.datatype)
        elif(callable(spec)):
            try:
                spec = np.array(spec(np.arange(3*self.para[0],dtype=np.int)),dtype=self.datatype)
            except:
                raise TypeError(about._errors.cstring("ERROR: invalid power spectra function.")) ## exception in ``spec(kindex)``
        elif(np.isscalar(spec)):
            spec = np.array([spec],dtype=self.datatype)
        else:
            spec = np.array(spec,dtype=self.datatype)

        ## check finiteness
        if(not np.all(np.isfinite(spec))):
            about.warnings.cprint("WARNING: infinite value(s).")
        ## check positivity (excluding null)
        if(np.any(spec<0)):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))
        elif(np.any(spec==0)):
            about.warnings.cprint("WARNING: nonpositive value(s).")

        size = 3*self.para[0] ## 3*nside
        ## extend
        if(np.size(spec)==1):
            spec = spec*np.ones(size,dtype=spec.dtype,order='C')
        ## size check
        elif(np.size(spec)<size):
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(spec))+" < "+str(size)+" )."))
        elif(np.size(spec)>size):
            about.warnings.cprint("WARNING: power spectrum cut to size ( == "+str(size)+" ).")
            spec = spec[:size]

        return spec

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_power_indices(self,**kwargs):
        """
            Raises
            ------
            AttributeError
                Always. -- The power spectrum for a field on the sphere
            is defined by its spherical harmonics components and not its
            position space representation.

        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra indexing ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters.

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

                - "pm1" (uniform distribution over {+1,-1}
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
            spec : {scalar, list, numpy.array, nifty.field, function}, *optional*
                Power spectrum (default: 1).
            codomain : nifty.lm_space, *optional*
                A compatible codomain for power indexing (default: None).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.arguments(self,**kwargs)

        if(arg is None):
            x = np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="syn"):
            lmax = 3*self.para[0]-1 ## 3*nside-1
            x = hp.synfast(arg[1],self.para[0],lmax=lmax,mmax=lmax,alm=False,pol=True,pixwin=False,fwhm=0.0,sigma=None) ## FIXME: `verbose` kwarg
            ## weight if discrete
            if(self.discrete):
                x = self.calc_weight(x,power=0.5)

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        return x

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

            Notes
            -----
            Compatible codomains are instances of :py:class:`hp_space` and
            :py:class:`lm_space`.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        if(isinstance(codomain,lm_space)):
            ##        3*nside-1==lmax                             lmax==mmax
            if(3*self.para[0]-1==codomain.para[0])and(codomain.para[0]==codomain.para[1]):
                return True
            else:
                about.warnings.cprint("WARNING: unrecommended codomain.")

        return False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Returns
            -------
            codomain : nifty.lm_space
                A compatible codomain.
        """
        return lm_space(3*self.para[0]-1,mmax=3*self.para[0]-1,datatype=np.complex128) ## lmax,mmax = 3*nside-1,3*nside-1

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each field component (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the field components or the complete space.

            Notes
            -----
            For HEALpix discretizations, the meta volumes are the pixel sizes.
        """
        if(total):
            return self.datatype(4*pi)
        else:
            mol = np.ones(self.dim(split=True),dtype=self.datatype,order='C')
            return self.calc_weight(mol,power=1)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,**kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                Target space to which the transformation shall map
                (default: self).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations performed in the HEALPix basis
                transformation.

            Notes
            -----
            Only instances of the :py:class:`lm_space` or :py:class:`hp_space`
            classes are allowed as `codomain`.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        self.check_codomain(codomain) ## a bit pointless

        if(self==codomain):
            return x ## T == id

        if(isinstance(codomain,lm_space)):
            ## weight if discrete
            if(self.discrete):
                x = self.calc_weight(x,power=-0.5)
            ## transform
            Tx = hp.map2alm(x.astype(np.float64),lmax=codomain.para[0],mmax=codomain.para[1],iter=kwargs.get("iter",self.niter),pol=True,use_weights=False,datapath=None)

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        return Tx.astype(codomain.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
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

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations performed in the HEALPix basis
                transformation.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## check sigma
        if(sigma==0):
            return x
        elif(sigma==-1):
            about.infos.cprint("INFO: invalid sigma reset.")
            sigma = 1.5/self.para[0] ## sqrt(2)*pi/(lmax+1)
        elif(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        ## smooth
        return hp.smoothing(x,fwhm=0.0,sigma=sigma,pol=True,iter=kwargs.get("iter",self.niter),lmax=3*self.para[0]-1,mmax=3*self.para[0]-1,use_weights=False,datapath=None)

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
            iter : int, *optional*
                Number of iterations performed in the HEALPix basis
                transformation.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## weight if discrete
        if(self.discrete):
            x = self.calc_weight(x,power=-0.5)
        ## power spectrum
        return hp.anafast(x,map2=None,nspec=None,lmax=3*self.para[0]-1,mmax=3*self.para[0]-1,iter=kwargs.get("iter",self.niter),alm=False,pol=True,use_weights=False,datapath=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,power=False,unit="",norm=None,cmap=None,cbar=True,other=None,legend=False,mono=True,**kwargs):
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
            iter : int, *optional*
                Number of iterations performed in the HEALPix basis
                transformation.
        """
        if(not pl.isinteractive())and(not bool(kwargs.get("save",False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(power):
            x = self.calc_power(x,**kwargs)

            fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

            xaxes = np.arange(3*self.para[0],dtype=np.int)
            if(vmin is None):
                vmin = np.min(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*x)[1:],color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
            if(mono):
                ax0.scatter(0.5*(xaxes[1]+xaxes[2]),x[0],s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=1)

            if(other is not None):
                if(isinstance(other,tuple)):
                    other = list(other)
                    for ii in xrange(len(other)):
                        if(isinstance(other[ii],field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii])
                elif(isinstance(other,field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other)]
                imax = max(1,len(other)-1)
                for ii in xrange(len(other)):
                    ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*other[ii])[1:],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5*(xaxes[1]+xaxes[2]),other[ii][0],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1],xaxes[-1])
            ax0.set_xlabel(r"$\ell$")
            ax0.set_ylim(vmin,vmax)
            ax0.set_ylabel(r"$\ell(2\ell+1) C_\ell$")
            ax0.set_title(title)

        else:
            x = self.enforce_shape(np.array(x,dtype=self.datatype))
            if(norm=="log"):
                if(vmin is not None):
                    if(vmin<=0):
                        raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))
                elif(np.min(x,axis=None,out=None)<=0):
                    raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))
            if(cmap is None):
                cmap = pl.cm.jet ## default
            cmap.set_under(color='k',alpha=0.0) ## transparent box
            hp.mollview(x,fig=None,rot=None,coord=None,unit=unit,xsize=800,title=title,nest=False,min=vmin,max=vmax,flip="astro",remove_dip=False,remove_mono=False,gal_cut=0,format="%g",format2="%g",cbar=cbar,cmap=cmap,notext=False,norm=norm,hold=False,margins=None,sub=None)
            fig = pl.gcf()

        if(bool(kwargs.get("save",False))):
            fig.savefig(str(kwargs.get("save")),dpi=None,facecolor="none",edgecolor="none",orientation="portrait",papertype=None,format=None,transparent=False,bbox_inches=None,pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_lm.hp_space>"

    def __str__(self):
        return "nifty_lm.hp_space instance\n- nside = "+str(self.para[0])

##-----------------------------------------------------------------------------

