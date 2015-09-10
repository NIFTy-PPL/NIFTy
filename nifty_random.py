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

import numpy as np
from keepers import about
from nifty_mpi_data import distributed_data_object

##-----------------------------------------------------------------------------

class random(object):
    """
        ..                                          __
        ..                                        /  /
        ..       _____   ____ __   __ ___    ____/  /  ______    __ ____ ___
        ..     /   __/ /   _   / /   _   | /   _   / /   _   | /   _    _   |
        ..    /  /    /  /_/  / /  / /  / /  /_/  / /  /_/  / /  / /  / /  /
        ..   /__/     \______| /__/ /__/  \______|  \______/ /__/ /__/ /__/  class

        NIFTY (static) class for pseudo random number generators.

    """
    __init__ = None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def parse_arguments(domain,**kwargs):
        """
            Analyses the keyword arguments for supported or necessary ones.

            Parameters
            ----------
            domain : space
                Space wherein the random field values live.
            random : string, *optional*
                Specifies a certain distribution to be drawn from using a
                pseudo random number generator. Supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given
                    standard deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

            dev : {scalar, list, ndarray, field}, *optional*
                Standard deviation of the normal distribution if
                ``random == "gau"`` (default: None).
            var : {scalar, list, ndarray, field}, *optional*
                Variance of the normal distribution (outranks the standard
                deviation) if ``random == "gau"`` (default: None).
            spec : {scalar, list, array, field, function}, *optional*
                Power spectrum for ``random == "syn"`` (default: 1).
            size : integer, *optional*
                Number of irreducible bands for ``random == "syn"``
                (default: None).
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index of each band
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each irreducible band (default: None).
            vmax : {scalar, list, ndarray, field}, *optional*
                Upper limit of the uniform distribution if ``random == "uni"``
                (default: 1).

            Returns
            -------
            arg : list
                Ordered list of arguments (to be processed in
                ``get_random_values`` of the domain).

            Other Parameters
            ----------------
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

            Raises
            ------
            KeyError
                If the `random` key is not supporrted.

        """
        if "random" in kwargs:
            key = kwargs.get("random")
        else:
            return None

        if key == "pm1":
            return [key]

        elif key == "gau":
            """
            if("mean" in kwargs):
                mean = domain.enforce_values(kwargs.get("mean"),extend=False)
            else:
                mean = None
            if("dev" in kwargs):
                dev = domain.enforce_values(kwargs.get("dev"),extend=False)
            else:
                dev = None
            if("var" in kwargs):
                var = domain.enforce_values(kwargs.get("var"),extend=False)
            else:
                var = None
            """
            mean = kwargs.get('mean', None)
            dev = kwargs.get('dev', None)
            var = kwargs.get('var', None)
            return [key,mean,dev,var]

        elif key == "syn":
            pindex = kwargs.get('pindex', None)
            kindex = kwargs.get('kindex', None)
            size = kwargs.get('size', None)
            log = kwargs.get('log', False)
            nbin = kwargs.get('nbin', None)
            binbounds = kwargs.get('binbounds', None)
            spec = kwargs.get('spec', 1)
            codomain = kwargs.get('codomain', None)

            ## check which domain should be taken for powerindexing
            if domain.check_codomain(codomain) == True and\
                codomain.fourier == True:
                fourier_domain = codomain
            elif domain.fourier == True:
                fourier_domain = domain
            else:
                fourier_domain = domain.get_codomain()

            ## building kpack
            if pindex != None and kindex != None:
                pindex = distributed_data_object(pindex,
                                                 distribution_strategy='fftw')
                kpack = [pindex, kindex]
            else:
                kpack = None

            ## simply put size and kindex into enforce_power
            ## if one or both are None, enforce power will fix that
            spec = fourier_domain.enforce_power(spec, size=size, kindex=kindex)



            return [key, spec, kpack, fourier_domain, log, nbin, binbounds]


            '''
            ## explicit power indices
            if ("pindex" in kwargs) and ("kindex" in kwargs):
                kindex = kwargs.get("kindex")
                if(kindex is None):
                    spec = domain.enforce_power(kwargs.get("spec",1),size=kwargs.get("size",None))
                    kpack = None
                else:
                    spec = domain.enforce_power(kwargs.get("spec",1),size=len(kindex),kindex=kindex)
                    pindex = kwargs.get("pindex",None)
                    if(pindex is None):
                        kpack = None
                    else:
                        kpack = [pindex,kindex]



            ## implicit power indices
            else:
                try:
                    domain.set_power_indices(**kwargs)
                except:
                    codomain = kwargs.get("codomain",None)
                    if(codomain is None):
                        spec = domain.enforce_power(kwargs.get("spec",1),size=kwargs.get("size",None))
                        kpack = None
                    else:
                        domain.check_codomain(codomain)
                        codomain.set_power_indices(**kwargs)
                        kindex = codomain.power_indices.get("kindex")
                        spec = domain.enforce_power(kwargs.get("spec",1),size=len(kindex),kindex=kindex,codomain=codomain)
                        kpack = [codomain.power_indices.get("pindex"),kindex]
                else:
                    kindex = domain.power_indices.get("kindex")
                    spec = domain.enforce_power(kwargs.get("spec",1),size=len(kindex),kindex=kindex)
                    kpack = [domain.power_indices.get("pindex"),kindex]
            '''


        elif key == "uni":
            """
            if("vmin" in kwargs):
                vmin = domain.enforce_values(kwargs.get("vmin"),extend=False)
            else:
                vmin = 0
            if("vmax" in kwargs):
                vmax = domain.enforce_values(kwargs.get("vmax"),extend=False)
            else:
                vmax = 1
            """
            vmin = domain.datatype(kwargs.get('vmin', 0))
            vmax = domain.datatype(kwargs.get('vmax', 1))
            return [key,vmin,vmax]

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(key)+"'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def pm1(datatype=np.int, shape=1):
        """
            Generates random field values according to an uniform distribution
            over {+1,-1} or {+1,+i,-1,-i}, respectively.

            Parameters
            ----------
            datatype : type, *optional*
                Data type of the field values (default: np.int).
            shape : {integer, tuple, list, ndarray}, *optional*
                Split up dimension of the space (default: 1).

            Returns
            -------
            x : ndarray
                Random field values (with correct dtype and shape).

        """
        size = np.prod(shape,axis=0,dtype=np.int,out=None)

        if(issubclass(datatype,np.complexfloating)):
            x = np.array([1+0j,0+1j,-1+0j,0-1j],dtype=datatype)[np.random.randint(4,high=None,size=size)]
        else:
            x = 2*np.random.randint(2,high=None,size=size)-1

        return x.astype(datatype).reshape(shape,order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def gau(datatype=np.float64, shape=1, mean=None, dev=None, var=None):
        """
            Generates random field values according to a normal distribution.

            Parameters
            ----------
            datatype : type, *optional*
                Data type of the field values (default: np.float64).
            shape : {integer, tuple, list, ndarray}, *optional*
                Split up dimension of the space (default: 1).
            mean : {scalar, ndarray}, *optional*
                Mean of the normal distribution (default: 0).
            dev : {scalar, ndarray}, *optional*
                Standard deviation of the normal distribution (default: 1).
            var : {scalar, ndarray}, *optional*
                Variance of the normal distribution (outranks the standard
                deviation) (default: None).

            Returns
            -------
            x : ndarray
                Random field values (with correct dtype and shape).

            Raises
            ------
            ValueError
                If the array dimension of `mean`, `dev` or `var` mismatch with
                `shape`.

        """
        size = np.prod(shape,axis=0,dtype=np.int,out=None)

        if(issubclass(datatype,np.complexfloating)):
            x = np.empty(size,dtype=datatype,order='C')
            x.real = np.random.normal(loc=0,scale=np.sqrt(0.5),size=size)
            x.imag = np.random.normal(loc=0,scale=np.sqrt(0.5),size=size)
        else:
            x = np.random.normal(loc=0,scale=1,size=size)

        if(var is not None):
            if(np.size(var)==1):
                x *= np.sqrt(np.abs(var))
            elif(np.size(var)==size):
                x *= np.sqrt(np.absolute(var).flatten(order='C'))
            else:
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(var))+" <> "+str(size)+" )."))
        elif(dev is not None):
            if(np.size(dev)==1):
                x *= np.abs(dev)
            elif(np.size(dev)==size):
                x *= np.absolute(dev).flatten(order='C')
            else:
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(dev))+" <> "+str(size)+" )."))
        if(mean is not None):
            if(np.size(mean)==1):
                x += mean
            elif(np.size(mean)==size):
                x += np.array(mean).flatten(order='C')
            else:
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(mean))+" <> "+str(size)+" )."))

        return x.astype(datatype).reshape(shape,order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def uni(datatype=np.float64,shape=1,vmin=0,vmax=1):
        """
            Generates random field values according to an uniform distribution
            over [vmin,vmax[.

            Parameters
            ----------
            datatype : type, *optional*
                Data type of the field values (default: np.float64).
            shape : {integer, tuple, list, ndarray}, *optional*
                Split up dimension of the space (default: 1).

            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution (default: 0).
            vmax : {scalar, list, ndarray, field}, *optional*
                Upper limit of the uniform distribution (default: 1).

            Returns
            -------
            x : ndarray
                Random field values (with correct dtype and shape).

        """
        size = np.prod(shape,axis=0,dtype=np.int,out=None)
        if(np.size(vmin)>1):
            vmin = np.array(vmin).flatten(order='C')
        if(np.size(vmax)>1):
            vmax = np.array(vmax).flatten(order='C')

        if(datatype in [np.complex64,np.complex128]):
            x = np.empty(size,dtype=datatype,order='C')
            x.real = (vmax-vmin)*np.random.random(size=size)+vmin
            x.imag = (vmax-vmin)*np.random.random(size=size)+vmin
        elif(datatype in [np.int8,np.int16,np.int32,np.int64]):
            x = np.random.randint(min(vmin,vmax),high=max(vmin,vmax),size=size)
        else:
            x = (vmax-vmin)*np.random.random(size=size)+vmin

        return x.astype(datatype).reshape(shape,order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.random>"

##-----------------------------------------------------------------------------

