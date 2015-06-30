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

"""
    ..                  __   ____   __
    ..                /__/ /   _/ /  /_
    ..      __ ___    __  /  /_  /   _/  __   __
    ..    /   _   | /  / /   _/ /  /   /  / /  /
    ..   /  / /  / /  / /  /   /  /_  /  /_/  /
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  core
    ..                               /______/

    .. The NIFTY project homepage is http://www.mpa-garching.mpg.de/ift/nifty/

    NIFTY [#]_, "Numerical Information Field Theory", is a versatile
    library designed to enable the development of signal inference algorithms
    that operate regardless of the underlying spatial grid and its resolution.
    Its object-oriented framework is written in Python, although it accesses
    libraries written in Cython, C++, and C for efficiency.

    NIFTY offers a toolkit that abstracts discretized representations of
    continuous spaces, fields in these spaces, and operators acting on fields
    into classes. Thereby, the correct normalization of operations on fields is
    taken care of automatically without concerning the user. This allows for an
    abstract formulation and programming of inference algorithms, including
    those derived within information field theory. Thus, NIFTY permits its user
    to rapidly prototype algorithms in 1D and then apply the developed code in
    higher-dimensional settings of real world problems. The set of spaces on
    which NIFTY operates comprises point sets, n-dimensional regular grids,
    spherical spaces, their harmonic counterparts, and product spaces
    constructed as combinations of those.

    References
    ----------
    .. [#] Selig et al., "NIFTY -- Numerical Information Field Theory --
        a versatile Python library for signal inference",
        `A&A, vol. 554, id. A26 <http://dx.doi.org/10.1051/0004-6361/201321236>`_,
        2013; `arXiv:1301.4499 <http://www.arxiv.org/abs/1301.4499>`_

    Class & Feature Overview
    ------------------------
    The NIFTY library features three main classes: **spaces** that represent
    certain grids, **fields** that are defined on spaces, and **operators**
    that apply to fields.

    .. Overview of all (core) classes:
    ..
    .. - switch
    .. - notification
    .. - _about
    .. - random
    .. - space
    ..     - point_space
    ..     - rg_space
    ..     - lm_space
    ..     - gl_space
    ..     - hp_space
    ..     - nested_space
    .. - field
    .. - operator
    ..     - diagonal_operator
    ..         - power_operator
    ..     - projection_operator
    ..     - vecvec_operator
    ..     - response_operator
    .. - probing
    ..     - trace_probing
    ..     - diagonal_probing

    Overview of the main classes and functions:

    .. automodule:: nifty

    - :py:class:`space`
        - :py:class:`point_space`
        - :py:class:`rg_space`
        - :py:class:`lm_space`
        - :py:class:`gl_space`
        - :py:class:`hp_space`
        - :py:class:`nested_space`
    - :py:class:`field`
    - :py:class:`operator`
        - :py:class:`diagonal_operator`
            - :py:class:`power_operator`
        - :py:class:`projection_operator`
        - :py:class:`vecvec_operator`
        - :py:class:`response_operator`

        .. currentmodule:: nifty.nifty_tools

        - :py:class:`invertible_operator`
        - :py:class:`propagator_operator`

        .. currentmodule:: nifty.nifty_explicit

        - :py:class:`explicit_operator`

    .. automodule:: nifty

    - :py:class:`probing`
        - :py:class:`trace_probing`
        - :py:class:`diagonal_probing`

        .. currentmodule:: nifty.nifty_explicit

        - :py:class:`explicit_probing`

    .. currentmodule:: nifty.nifty_tools

    - :py:class:`conjugate_gradient`
    - :py:class:`steepest_descent`

    .. currentmodule:: nifty.nifty_explicit

    - :py:func:`explicify`

    .. currentmodule:: nifty.nifty_power

    - :py:func:`weight_power`,
      :py:func:`smooth_power`,
      :py:func:`infer_power`,
      :py:func:`interpolate_power`

"""
from __future__ import division
import numpy as np
import pylab as pl
from nifty_paradict import space_paradict,\
                            point_space_paradict,\
                            nested_space_paradict

from nifty_about import about
from nifty_random import random


pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679




##=============================================================================

class space(object):
    """
        ..     _______   ______    ____ __   _______   _______
        ..   /  _____/ /   _   | /   _   / /   ____/ /   __  /
        ..  /_____  / /  /_/  / /  /_/  / /  /____  /  /____/
        .. /_______/ /   ____/  \______|  \______/  \______/  class
        ..          /__/

        NIFTY base class for spaces and their discretizations.

        The base NIFTY space class is an abstract class from which other
        specific space subclasses, including those preimplemented in NIFTY
        (e.g. the regular grid class) must be derived.

        Parameters
        ----------
        para : {single object, list of objects}, *optional*
            This is a freeform list of parameters that derivatives of the space
            class can use (default: 0).
        datatype : numpy.dtype, *optional*
            Data type of the field values for a field defined on this space
            (default: numpy.float64).

        See Also
        --------
        point_space :  A class for unstructured lists of numbers.
        rg_space : A class for regular cartesian grids in arbitrary dimensions.
        hp_space : A class for the HEALPix discretization of the sphere
            [#]_.
        gl_space : A class for the Gauss-Legendre discretization of the sphere
            [#]_.
        lm_space : A class for spherical harmonic components.
        nested_space : A class for product spaces.

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
        para : {single object, list of objects}
            This is a freeform list of parameters that derivatives of the space class can use.
        datatype : numpy.dtype
            Data type of the field values for a field defined on this space.
        discrete : bool
            Whether the space is inherently discrete (true) or a discretization
            of a continuous space (false).
        vol : numpy.ndarray
            An array of pixel volumes, only one component if the pixels all
            have the same volume.
    """
    def __init__(self,para=0,datatype=None):
        """
            Sets the attributes for a space class instance.

            Parameters
            ----------
            para : {single object, list of objects}, *optional*
                This is a freeform list of parameters that derivatives of the
                space class can use (default: 0).
            datatype : numpy.dtype, *optional*
                Data type of the field values for a field defined on this space
                (default: numpy.float64).

            Returns
            -------
            None
        """
        self.paradict = space_paradict(default=para)        

        ## check data type
        if(datatype is None):
            datatype = np.float64
        elif(datatype not in [np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64,np.complex64,np.complex128]):
            about.warnings.cprint("WARNING: data type set to default.")
            datatype = np.float64
        self.datatype = datatype

        self.discrete = True
        self.vol = np.real(np.array([1],dtype=self.datatype))
        
    @property
    def para(self):
        return self.paradict['default']
    
    @para.setter
    def para(self, x):
        self.paradict['default'] = x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _freeze_config(self, dictionary):
        """
            a helper function which forms a hashable identifying object from 
            a dictionary which can be used as key of a dict
        """        
        return frozenset(dictionary.items())



    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def getitem(self, data, key):
        raise NotImplementedError(about._errors.cstring(\
            "ERROR: no generic instance method 'getitem'."))
        

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setitem(self, data, key):
        raise NotImplementedError(about._errors.cstring(\
            "ERROR: no generic instance method 'getitem'."))
        
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++            
    def apply_scalar_function(self, x, function, inplace=False):
        raise NotImplementedError(about._errors.cstring(\
            "ERROR: no generic instance method 'apply_scalar_function'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++            
    def unary_operation(self, x, op=None):
        raise NotImplementedError(about._errors.cstring(\
            "ERROR: no generic instance method 'unary_operation'."))
    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++            
    def binary_operation(self, x, y, op=None):
        raise NotImplementedError(about._errors.cstring(\
            "ERROR: no generic instance method 'binary_operation'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++            
    def norm(self, x, q):
        raise NotImplementedError(about._errors.cstring(\
            "ERROR: no generic instance method 'norm'."))

    def shape(self):
        raise NotImplementedError(about._errors.cstring(\
            "ERROR: no generic instance method 'shape'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of pixels.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension split up, i.e. the numbers of
                pixels in each direction, or not (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension(s) of the space.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'dim'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'dof'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
        """
            Provides a valid power spectrum array from a given object.

            Parameters
            ----------
            spec : {scalar, list, numpy.ndarray, nifty.field, function}
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
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'enforce_power'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_power_indices(self,**kwargs):
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

        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'set_power_indices'."))

    def get_power_indices(self,**kwargs):
        """
            Provides the (un)indexing objects for spectral indexing.

            Provides one-dimensional arrays containing the scales of the
            spectral bands and the numbers of modes per scale, and an array
            giving for each component of a field the corresponding index of a
            power spectrum as well as an Unindexing array.

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
            kindex : numpy.ndarray
                Scale of each spectral band.
            rho : numpy.ndarray
                Number of modes per scale represented in the discretization.
            pindex : numpy.ndarray
                Indexing array giving the power spectrum index for each
                represented mode.
            pundex : numpy.ndarray
                Unindexing array undoing power spectrum indexing.

            Notes
            -----
            The ``kindex`` and ``rho`` are each one-dimensional arrays.
            The indexing array is of the same shape as a field living in this
            space and contains the indices of the associated bands.
            Indexing with the unindexing array undoes the indexing with the
            indexing array; i.e., ``power == power[pindex].flatten()[pundex]``.

            See Also
            --------
            set_power_indices

        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'get_power_indices'."))

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
        return self.enforce_values(x, extend=True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_shape(self,x):
        """
            Shapes an array of valid field values correctly, according to the
            specifications of the space instance.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values to be put into shape.

            Returns
            -------
            y : numpy.ndarray
                Correctly shaped array.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'enforce_shape'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_values(self,x,extend=True):
        """
            Computes valid field values from a given object, according to the
            constraints from the space instance.

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
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'enforce_values'."))


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
            spec : {scalar, list, numpy.ndarray, nifty.field, function}, *optional*
                Power spectrum (default: 1).
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index of each band
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band (default: None).
            codomain : nifty.space, *optional*
                A compatible codomain with power indices (default: None).
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
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'get_random_values'."))

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
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'check_codomain'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, usually either the position basis or the basis of
            harmonic eigenmodes.

            Parameters
            ----------
            coname : string, *optional*
                String specifying a desired codomain (default: None).
            cozerocenter : {bool, numpy.ndarray}, *optional*
                Whether or not the grid is zerocentered for each axis or not
                (default: None).
            conest : list, *optional*
                List of nested spaces of the codomain (default: None).
            coorder : list, *optional*
                Permutation of the list of nested spaces (default: None).

            Returns
            -------
            codomain : nifty.space
                A compatible codomain.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'get_codomain'."))

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
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'get_meta_volume'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_weight(self,x,power=1):
        """
            Weights a given array of field values with the pixel volumes (not
            the meta volumes) to a given power.

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
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'calc_weight'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
        raise NotImplementedError(about._errors.cstring(\
            "ERROR: no generic instance method 'dot'."))



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
                Number of iterations performed in specific transformations.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'calc_transform'."))

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
                of length in position space (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations (default: 0).
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'calc_smooth'."))

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
            kindex : numpy.ndarray, *optional*
                Scale corresponding to each band in the power spectrum
                (default: None).
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
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'calc_power'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,**kwargs):
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
            iter : int, *optional*
                Number of iterations (default: 0).

        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'get_plot'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.space>"

    def __str__(self):
        return "nifty_core.space instance\n- para     = "+str(self.para)+"\n- datatype = numpy."+str(np.result_type(self.datatype))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __len__(self):
        return int(self.dim(split=False))

    ## _identiftier returns an object which contains all information needed 
    ## to uniquely idetnify a space. It returns a (immutable) tuple which therefore
    ## can be compored. 
    def _identifier(self):
        return tuple(sorted(vars(self).items()))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _meta_vars(self): ## > captures all nonstandard properties
        mars = np.array([ii[1] for ii in vars(self).iteritems() if ii[0] not in ["para","datatype","discrete","vol","power_indices"]],dtype=np.object)
        if(np.size(mars)==0):
            return None
        else:
            return mars

    def __eq__(self,x): ## __eq__ : self == x
        if isinstance(x, type(self)):
            return self._identifier() == x._identifier()
        else:
            return False

    def __ne__(self,x): ## __ne__ : self <> x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.any(self.para!=x.para))or(self.discrete!=x.discrete)or(np.any(self.vol!=x.vol))or(np.any(self._meta_vars()!=x._meta_vars())): ## data types are ignored
                return True
        return False

    def __lt__(self,x): ## __lt__ : self < x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.size(self.para)!=np.size(x.para))or(np.size(self.vol)!=np.size(x.vol)):
                raise ValueError(about._errors.cstring("ERROR: incomparable spaces."))
            elif(self.discrete==x.discrete): ## data types are ignored
                for ii in xrange(np.size(self.para)):
                    if(self.para[ii]<x.para[ii]):
                        return True
                    elif(self.para[ii]>x.para[ii]):
                        return False
                for ii in xrange(np.size(self.vol)):
                    if(self.vol[ii]<x.vol[ii]):
                        return True
                    elif(self.vol[ii]>x.vol[ii]):
                        return False
                s_mars = self._meta_vars()
                x_mars = x._meta_vars()
                for ii in xrange(np.size(s_mars)):
                    if(np.all(s_mars[ii]<x_mars[ii])):
                        return True
                    elif(np.any(s_mars[ii]>x_mars[ii])):
                        break
        return False

    def __le__(self,x): ## __le__ : self <= x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.size(self.para)!=np.size(x.para))or(np.size(self.vol)!=np.size(x.vol)):
                raise ValueError(about._errors.cstring("ERROR: incomparable spaces."))
            elif(self.discrete==x.discrete): ## data types are ignored
                for ii in xrange(np.size(self.para)):
                    if(self.para[ii]<x.para[ii]):
                        return True
                    if(self.para[ii]>x.para[ii]):
                        return False
                for ii in xrange(np.size(self.vol)):
                    if(self.vol[ii]<x.vol[ii]):
                        return True
                    if(self.vol[ii]>x.vol[ii]):
                        return False
                s_mars = self._meta_vars()
                x_mars = x._meta_vars()
                for ii in xrange(np.size(s_mars)):
                    if(np.all(s_mars[ii]<x_mars[ii])):
                        return True
                    elif(np.any(s_mars[ii]>x_mars[ii])):
                        return False
                return True
        return False

    def __gt__(self,x): ## __gt__ : self > x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.size(self.para)!=np.size(x.para))or(np.size(self.vol)!=np.size(x.vol)):
                raise ValueError(about._errors.cstring("ERROR: incomparable spaces."))
            elif(self.discrete==x.discrete): ## data types are ignored
                for ii in xrange(np.size(self.para)):
                    if(self.para[ii]>x.para[ii]):
                        return True
                    elif(self.para[ii]<x.para[ii]):
                        break
                for ii in xrange(np.size(self.vol)):
                    if(self.vol[ii]>x.vol[ii]):
                        return True
                    elif(self.vol[ii]<x.vol[ii]):
                        break
                s_mars = self._meta_vars()
                x_mars = x._meta_vars()
                for ii in xrange(np.size(s_mars)):
                    if(np.all(s_mars[ii]>x_mars[ii])):
                        return True
                    elif(np.any(s_mars[ii]<x_mars[ii])):
                        break
        return False

    def __ge__(self,x): ## __ge__ : self >= x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.size(self.para)!=np.size(x.para))or(np.size(self.vol)!=np.size(x.vol)):
                raise ValueError(about._errors.cstring("ERROR: incomparable spaces."))
            elif(self.discrete==x.discrete): ## data types are ignored
                for ii in xrange(np.size(self.para)):
                    if(self.para[ii]>x.para[ii]):
                        return True
                    if(self.para[ii]<x.para[ii]):
                        return False
                for ii in xrange(np.size(self.vol)):
                    if(self.vol[ii]>x.vol[ii]):
                        return True
                    if(self.vol[ii]<x.vol[ii]):
                        return False
                s_mars = self._meta_vars()
                x_mars = x._meta_vars()
                for ii in xrange(np.size(s_mars)):
                    if(np.all(s_mars[ii]>x_mars[ii])):
                        return True
                    elif(np.any(s_mars[ii]<x_mars[ii])):
                        return False
                return True
        return False

##=============================================================================



##-----------------------------------------------------------------------------

class point_space(space):
    """
        ..                            __             __
        ..                          /__/           /  /_
        ..      ______    ______    __   __ ___   /   _/
        ..    /   _   | /   _   | /  / /   _   | /  /
        ..   /  /_/  / /  /_/  / /  / /  / /  / /  /_
        ..  /   ____/  \______/ /__/ /__/ /__/  \___/  space class
        .. /__/

        NIFTY subclass for unstructured spaces.

        Unstructured spaces are lists of values without any geometrical
        information.

        Parameters
        ----------
        num : int
            Number of points.
        datatype : numpy.dtype, *optional*
            Data type of the field values (default: None).

        Attributes
        ----------
        para : numpy.ndarray
            Array containing the number of points.
        datatype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Parameter captioning the fact that a :py:class:`point_space` is
            always discrete.
        vol : numpy.ndarray
            Pixel volume of the :py:class:`point_space`, which is always 1.
    """
    def __init__(self,num,datatype=None):
        """
            Sets the attributes for a point_space class instance.

            Parameters
            ----------
            num : int
                Number of points.
            datatype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.float64).

            Returns
            -------
            None.
        """
        self.paradict = point_space_paradict(num=num)       
        
        ## check datatype
        if(datatype is None):
            datatype = np.float64
        elif(datatype not in [np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64,np.complex64,np.complex128]):
            about.warnings.cprint("WARNING: data type set to default.")
            datatype = np.float64
        self.datatype = datatype

        self.discrete = True
        self.vol = np.real(np.array([1],dtype=self.datatype))


    @property
    def para(self):
        temp = np.array([self.paradict['num']], dtype=int)
        return temp
    
    @para.setter
    def para(self, x):
        self.paradict['num'] = x
        
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def getitem(self, data, key):
        return data[key]
        

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setitem(self, data, update, key):
        data[key]=update

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def apply_scalar_function(self, x, function, inplace=False):
        if inplace == False:        
            try: 
                return function(x)
            except:
                return np.vectorize(function)(x)
        else:
            try:
                x[:] = function(x)
            except:
                x[:] = np.vectorize(function)(x)
            return x
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++          
    
    
    def unary_operation(self, x, op='None', **kwargs):
        """
        x must be a numpy array which is compatible with the space!
        Valid operations are
        
        """
                                
        def _argmin(z, **kwargs):
            ind = np.argmin(z, **kwargs)
            if np.isscalar(ind):
                ind = np.unravel_index(ind, z.shape, order='C')
                if(len(ind)==1):
                    return ind[0]
            return ind         

        def _argmax(z, **kwargs):
            ind = np.argmax(z, **kwargs)
            if np.isscalar(ind):
                ind = np.unravel_index(ind, z.shape, order='C')
                if(len(ind)==1):
                    return ind[0]
            return ind         
        
        
        translation = {"pos" : lambda y: getattr(y, '__pos__')(),
                        "neg" : lambda y: getattr(y, '__neg__')(),
                        "abs" : lambda y: getattr(y, '__abs__')(),
                        "nanmin" : np.nanmin,  
                        "min" : np.amin,
                        "nanmax" : np.nanmax,
                        "max" : np.amax,
                        "med" : np.median,
                        "mean" : np.mean,
                        "std" : np.std,
                        "var" : np.var,
                        "argmin" : _argmin,
                        "argmin_flat" : np.argmin,
                        "argmax" : _argmax, 
                        "argmax_flat" : np.argmax,
                        "conjugate" : np.conjugate,
                        "None" : lambda y: y}

                
        return translation[op](x, **kwargs)      

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++            
    def binary_operation(self, x, y, op='None', cast=0):
        
        translation = {"add" : lambda z: getattr(z, '__add__'),
                        "radd" : lambda z: getattr(z, '__radd__'),
                        "iadd" : lambda z: getattr(z, '__iadd__'),
                        "sub" : lambda z: getattr(z, '__sub__'),
                        "rsub" : lambda z: getattr(z, '__rsub__'),
                        "isub" : lambda z: getattr(z, '__isub__'),
                        "mul" : lambda z: getattr(z, '__mul__'),
                        "rmul" : lambda z: getattr(z, '__rmul__'),
                        "imul" : lambda z: getattr(z, '__imul__'),
                        "div" : lambda z: getattr(z, '__div__'),
                        "rdiv" : lambda z: getattr(z, '__rdiv__'),
                        "idiv" : lambda z: getattr(z, '__idiv__'),
                        "pow" : lambda z: getattr(z, '__pow__'),
                        "rpow" : lambda z: getattr(z, '__rpow__'),
                        "ipow" : lambda z: getattr(z, '__ipow__'),
                        "None" : lambda z: lambda u: u}
        
        if (cast & 1) != 0:
            x = self.cast(x)
        if (cast & 2) != 0:
            y = self.cast(y)        
        
        return translation[op](x)(y)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++            
    def norm(self, x, q=2):
        """
            Computes the Lq-norm of field values.

            Parameters
            ----------
            x : np.ndarray 
                The data array 
            q : scalar
                Parameter q of the Lq-norm (default: 2).

            Returns
            -------
            norm : scalar
                The Lq-norm of the field values.

        """

        
        if(q == 2):
            result = self.calc_dot(x,x)
        else:
            y = x**(q-1)        
            result = self.calc_dot(x,y)
        
        result = result**(1./q)
        return result 



    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def num(self):
        """
            Returns the number of points.

            Returns
            -------
            num : int
                Number of points.
        """
        return self.para[0]

    def shape(self):
        return np.array([self.paradict['num']])

        
    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of points.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension as an array with one component
                or as a scalar (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension(s) of the space.
        """
        ## dim = num
        if(split):
            return self.shape()
            #return np.array([self.para[0]],dtype=np.int)
        else:
            return np.prod(self.shape())
            #return self.para[0]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space, i.e./  the
            number of points for real-valued fields and twice that number for
            complex-valued fields.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        ## dof ~ dim
        if(issubclass(self.datatype,np.complexfloating)):
            return 2*self.para[0]
        else:
            return self.para[0]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_power_indices(self,**kwargs):
        """
            Provides the (un)indexing objects for spectral indexing.

            Provides one-dimensional arrays containing the scales of the
            spectral bands and the numbers of modes per scale, and an array
            giving for each component of a field the corresponding index of a
            power spectrum as well as an Unindexing array.

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
            kindex : numpy.ndarray
                Scale of each spectral band.
            rho : numpy.ndarray
                Number of modes per scale represented in the discretization.
            pindex : numpy.ndarray
                Indexing array giving the power spectrum index for each
                represented mode.
            pundex : numpy.ndarray
                Unindexing array undoing power spectrum indexing.

            Notes
            -----
            The ``kindex`` and ``rho`` are each one-dimensional arrays.
            The indexing array is of the same shape as a field living in this
            space and contains the indices of the associated bands.
            Indexing with the unindexing array undoes the indexing with the
            indexing array; i.e., ``power == power[pindex].flatten()[pundex]``.

            See Also
            --------
            set_power_indices

        """
        self.set_power_indices(**kwargs)
        return self.power_indices.get("kindex"),\
                self.power_indices.get("rho"),\
                self.power_indices.get("pindex"),\
                self.power_indices.get("pundex")

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_shape(self,x):
        """
            Shapes an array of valid field values correctly, according to the
            specifications of the space instance.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values to be put into shape.

            Returns
            -------
            y : numpy.ndarray
                Correctly shaped array.
        """
        x = np.array(x)

        if(np.size(x)!=self.dim(split=False)):
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(x))+" <> "+str(self.dim(split=False))+" )."))
#        elif(not np.all(np.array(np.shape(x))==self.dim(split=True))):
#            about.warnings.cprint("WARNING: reshaping forced.")

        return x.reshape(self.dim(split=True),order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_values(self,x,extend=True):
        """
            Computes valid field values from a given object, according to the
            constraints from the space instance.

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

        ## check finiteness
        if(not np.all(np.isfinite(x))):
            about.warnings.cprint("WARNING: infinite value(s).")

        return x

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
            spec : {scalar, list, numpy.ndarray, nifty.field, function}, *optional*
                Power spectrum (default: 1).
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index of each band
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band (default: None).
            codomain : nifty.space, *optional*
                A compatible codomain with power indices (default: None).
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
        arg = random.parse_arguments(self,**kwargs)

        if(arg is None):
            x = np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

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
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        return False
        
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
    def enforce_power(self,spec,**kwargs):
        """
            Raises an error since the power spectrum is ill-defined for point
            spaces.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra ill-defined for (unstructured) point spaces."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_power_indices(self,**kwargs):
        """
            Raises
            ------
            AttributeError
                Always. -- The power spectrum is ill-defined for point spaces.

        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra indexing ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, in this case another instance of
            :py:class:`point_space` with the same properties.

            Returns
            -------
            codomain : nifty.point_space
                A compatible codomain.
        """
        return point_space(self.para[0],datatype=self.datatype) ## == self

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
            Since point spaces are unstructured, the meta volume of each
            component is one, the total meta volume of the space is the number
            of points.
        """
        if(total):
            return self.dim(split=False)
        else:
            return np.ones(self.dim(split=True),dtype=self.vol.dtype,order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_weight(self,x,power=1):
        """
            Weights a given array of field values with the pixel volumes (not
            the meta volumes) to a given power.

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
        return x*self.vol**power

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
        result = np.vdot(x, y)
        if np.isreal(result):
            result = np.asscalar(np.real(result))            
        return result

    '''
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
        dot = np.dot(np.conjugate(x),y,out=None)
        if(np.isreal(dot)):
            return np.asscalar(np.real(dot))
        else:
            return dot
     
     '''

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
                Number of iterations performed in specific transformations.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        assert(self.check_codomain(codomain))

        if self == codomain:
            return x

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))


    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,**kwargs):
        """
            Raises an error since smoothing is ill-defined on an unstructured
            space.
        """
        raise AttributeError(about._errors.cstring("ERROR: smoothing ill-defined for (unstructured) point space."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Raises an error since the power spectrum is ill-defined for point
            spaces.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra ill-defined for (unstructured) point space."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,unit="",norm=None,other=None,legend=False,**kwargs):
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
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            other : {single object, tuple of objects}, *optional*
                Object or tuple of objects to be added, where objects can be
                scalars, arrays, or fields (default: None).
            legend : bool, *optional*
                Whether to show the legend or not (default: False).
            save : string, *optional*
                Valid file name where the figure is to be stored, by default
                the figure is not saved (default: False).

        """
        if(not pl.isinteractive())and(not bool(kwargs.get("save",False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
        ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

        xaxes = np.arange(self.para[0],dtype=np.int)
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

        if(norm=="log")and(vmin<=0):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

        if(np.iscomplexobj(x)):
            ax0.scatter(xaxes,np.absolute(x),s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph (absolute)",linewidths=None,verts=None,zorder=1)
            ax0.scatter(xaxes,np.real(x),s=20,color=[0.0,0.5,0.0],marker='s',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph (real part)",linewidths=None,verts=None,facecolor="none",zorder=1)
            ax0.scatter(xaxes,np.imag(x),s=20,color=[0.0,0.5,0.0],marker='D',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph (imaginary part)",linewidths=None,verts=None,facecolor="none",zorder=1)
            if(legend):
                ax0.legend()
        elif(other is not None):
            ax0.scatter(xaxes,x,s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph 0",linewidths=None,verts=None,zorder=1)
            if(isinstance(other,tuple)):
                other = [self.enforce_values(xx,extend=True) for xx in other]
            else:
                other = [self.enforce_values(other,extend=True)]
            imax = max(1,len(other)-1)
            for ii in xrange(len(other)):
                ax0.scatter(xaxes,other[ii],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph "+str(ii),linewidths=None,verts=None,zorder=-ii)
            if(legend):
                ax0.legend()
        else:
            ax0.scatter(xaxes,x,s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph 0",linewidths=None,verts=None,zorder=1)

        ax0.set_xlim(xaxes[0],xaxes[-1])
        ax0.set_xlabel("index")
        ax0.set_ylim(vmin,vmax)
        if(norm=="log"):
            ax0.set_yscale('log')

        if(unit):
            unit = " ["+unit+"]"
        ax0.set_ylabel("values"+unit)
        ax0.set_title(title)

        if(bool(kwargs.get("save",False))):
            fig.savefig(str(kwargs.get("save")),dpi=None,facecolor="none",edgecolor="none",orientation="portrait",papertype=None,format=None,transparent=False,bbox_inches=None,pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.point_space>"

    def __str__(self):
        return "nifty_core.point_space instance\n- num      = "+str(self.para[0])+"\n- datatype = numpy."+str(np.result_type(self.datatype))





##-----------------------------------------------------------------------------
class nested_space(space):
    """
        ..                                      __                    __
        ..                                    /  /_                 /  /
        ..      __ ___    _______   _______  /   _/  _______   ____/  /
        ..    /   _   | /   __  / /  _____/ /  /   /   __  / /   _   /
        ..   /  / /  / /  /____/ /_____  / /  /_  /  /____/ /  /_/  /
        ..  /__/ /__/  \______/ /_______/  \___/  \______/  \______|  space class

        NIFTY subclass for product spaces

        Parameters
        ----------
        nest : list
            A list of space instances that are to be combined into a product
            space.

        Notes
        -----
        Note that the order of the spaces is important for some of the methods.

        Attributes
        ----------
        nest : list
            List of the space instances that are combined into the product space, any instances of the :py:class:`nested_space` class itself are further unraveled.
        para : numpy.ndarray
            One-dimensional array containing the dimensions of all the space instances (split up into their axes when applicable) that are contained in the nested space.
        datatype : numpy.dtype
            Data type of the field values, inherited from the innermost space, i.e. that last entry in the `nest` list.
        discrete : bool
            Whether or not the product space is discrete, ``True`` only if all subspaces are discrete.
    """
    def __init__(self,nest):
        """
            Sets the attributes for a nested_space class instance.

            Parameters
            ----------
            nest : list
                A list of space instances that are to be combined into a product
                space.

            Returns
            -------
            None
        """
        if(not isinstance(nest,list)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        ## check nest
        purenest = []
        pre_para = []
        for nn in nest:
            if(not isinstance(nn,space)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            elif(isinstance(nn,nested_space)): ## no 2nd level nesting
                for nn_ in nn.nest:
                    purenest.append(nn_)
                    pre_para = pre_para + [nn_.dim(split=True)]
            else:
                purenest.append(nn)
                pre_para = pre_para + [nn.dim(split=True)]
        if(len(purenest)<2):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        self.nest = purenest
        
        self.paradict = nested_space_paradict(ndim=len(pre_para))                
        for i in range(len(pre_para)):
            self.paradict[i]=pre_para[i]
            

        ## check data type
        for nn in self.nest[:-1]:
            if(nn.datatype!=self.nest[-1].datatype): ## may conflict permutability
                about.infos.cprint("INFO: ambiguous data type.")
                break
        self.datatype = self.nest[-1].datatype

        self.discrete = np.prod([nn.discrete for nn in self.nest],axis=0,dtype=np.bool,out=None)
        self.vol = np.prod([nn.get_meta_volume(total=True) for nn in self.nest],axis=0,dtype=None,out=None) ## total volume


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
                
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def shape(self):
        temp = []
        for i in range(self.paradict.ndim):
            temp = np.append(temp, self.paradict[i])
        return temp
        
    def dim(self,split=False):
        """
            Computes the dimension of the product space.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension split up into the dimensions of
                each subspace, each one of these split up into the number of
                pixels along each axis when applicable, or not
                (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension(s) of the space.
        """
        if(split):
            return self.shape()
            #return self.para
        else:
            return np.prod(self.shape())
            #return np.prod(self.para,axis=0,dtype=None,out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the product space, as
            the product of the degrees of freedom of each subspace.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        return np.prod([nn.dof() for nn in self.nest],axis=0,dtype=None,out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
        """
            Raises an error since there is no canonical definition for the
            power spectrum on a generic product space.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_power_indices(self,**kwargs):
        """
            Raises
            ------
            AttributeError
                Always. -- There is no canonical definition for the power
                spectrum on a generic product space.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra indexing ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_values(self,x,extend=True):
        """
            Computes valid field values from a given object, according to the
            constraints from the space instances that make up the product
            space.

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
            elif(self.nest[-1]==x.domain):
                if(self.datatype is not x.domain.datatype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '"+str(np.result_type(self.datatype))+"' <> '"+str(np.result_type(x.domain.datatype))+"' )."))
                else:
                    subshape = self.para[:-np.size(self.nest[-1].dim(split=True))]
                    x = np.tensordot(np.ones(subshape,dtype=self.datatype,order='C'),x.val,axes=0)
            elif(isinstance(x.domain,nested_space)):
                if(self.datatype is not x.domain.datatype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '"+str(np.result_type(self.datatype))+"' <> '"+str(np.result_type(x.domain.datatype))+"' )."))
                else:
                    if(np.all(self.nest[-len(x.domain.nest):]==x.domain.nest)):
                        subshape = self.para[:np.sum([np.size(nn.dim(split=True)) for nn in self.nest[:-len(x.domain.nest)]],axis=0,dtype=np.int,out=None)]
                        x = np.tensordot(np.ones(subshape,dtype=self.datatype,order='C'),x.val,axes=0)
                    else:
                        raise ValueError(about._errors.cstring("ERROR: inequal domains."))
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            if(np.size(x)==1):
                if(extend):
                    x = self.datatype(x)*np.ones(self.para,dtype=self.datatype,order='C')
                else:
                    if(np.isscalar(x)):
                        x = np.array([x],dtype=self.datatype)
                    else:
                        x = np.array(x,dtype=self.datatype)
            else:
                x = np.array(x,dtype=self.datatype)
                if(np.ndim(x)<np.size(self.para)):
                    subshape = np.array([],dtype=np.int)
                    for ii in range(len(self.nest))[::-1]:
                        subshape = np.append(self.nest[ii].dim(split=True),subshape,axis=None)
                        if(np.all(np.array(np.shape(x))==subshape)):
                            subshape = self.para[:np.sum([np.size(nn.dim(split=True)) for nn in self.nest[:ii]],axis=0,dtype=np.int,out=None)]
                            x = np.tensordot(np.ones(subshape,dtype=self.datatype,order='C'),x,axes=0)
                            break
                else:
                    x = self.enforce_shape(x)

        if(np.size(x)!=1):
            subdim = np.prod(self.para[:-np.size(self.nest[-1].dim(split=True))],axis=0,dtype=np.int,out=None)
            ## enforce special properties
            x = x.reshape([subdim]+self.nest[-1].dim(split=True).tolist(),order='C')
            x = np.array([self.nest[-1].enforce_values(xx,extend=True) for xx in x],dtype=self.datatype).reshape(self.para,order='C')

        ## check finiteness
        if(not np.all(np.isfinite(x))):
            about.warnings.cprint("WARNING: infinite value(s).")

        return x

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
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.parse_arguments(self,**kwargs)

        if(arg is None):
            return np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        subdim = np.prod(self.para[:-np.size(self.nest[-1].dim(split=True))],axis=0,dtype=np.int,out=None)
        ## enforce special properties
        x = x.reshape([subdim]+self.nest[-1].dim(split=True).tolist(),order='C')
        x = np.array([self.nest[-1].enforce_values(xx,extend=True) for xx in x],dtype=self.datatype).reshape(self.para,order='C')

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,conest=None,coorder=None,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable.

            Parameters
            ----------
            conest : list, *optional*
                List of nested spaces of the codomain (default: None).
            coorder : list, *optional*
                Permutation of the list of nested spaces (default: None).

            Returns
            -------
            codomain : nifty.nested_space
                A compatible codomain.

            Notes
            -----
            By default, the codomain of the innermost subspace (i.e. the last
            entry of the `nest` list) is generated and the outer subspaces are
            left unchanged. If `conest` is given, this nested space is checked
            for compatibility and returned as codomain. If `conest` is not
            given but `coorder` is, the codomain is a reordered version of the
            original :py:class:`nested_space` instance.
        """
        if(conest is None):
            if(coorder is None):
                return nested_space(self.nest[:-1]+[self.nest[-1].get_codomain(**kwargs)])
            else:
                ## check coorder
                coorder = np.array(coorder,dtype=np.int).reshape(len(self.nest),order='C')
                if(np.any(np.sort(coorder,axis=0,kind="quicksort",order=None)!=np.arange(len(self.nest)))):
                    raise ValueError(about._errors.cstring("ERROR: invalid input."))
                ## check data type
                if(self.nest[np.argmax(coorder,axis=0)]!=self.datatype):
                    about.warnings.cprint("WARNING: ambiguous data type.")
                return nested_space(np.array(self.nest)[coorder].tolist()) ## list

        else:
            codomain = nested_space(conest)
            if(self.check_codomain(codomain)):
                return codomain
            else:
                raise ValueError(about._errors.cstring("ERROR: unsupported or incompatible input."))

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
            Only instances of the :py:class:`nested_space` class can be valid
            codomains.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        elif(isinstance(codomain,nested_space)):
            ##                nest'[:-1]==nest[:-1]
            if(np.all(codomain.nest[:-1]==self.nest[:-1])):
                return self.nest[-1].check_codomain(codomain.nest[-1])
            ##   len(nest')==len(nest)
            elif(len(codomain.nest)==len(self.nest)):
                ## check permutability
                unpaired = range(len(self.nest))
                ambiguous = False
                for ii in xrange(len(self.nest)):
                    for jj in xrange(len(self.nest)):
                        if(codomain.nest[ii]==self.nest[jj]):
                            if(jj in unpaired):
                                unpaired.remove(jj)
                                break
                            else:
                                ambiguous = True
                if(len(unpaired)!=0):
                    return False
                elif(ambiguous):
                    about.infos.cprint("INFO: ambiguous permutation.")
                return True

        return False

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
        """
        if(total):
            ## product
            return self.vol ## == np.prod([nn.get_meta_volume(total=True) for nn in self.nest],axis=0,dtype=None,out=None)
        else:
            mol = self.nest[0].get_meta_volume(total=False)
            ## tensor product
            for nn in self.nest[1:]:
                mol = np.tensordot(mol,nn.get_meta_volume(total=False),axes=0)
            return mol

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_weight(self,x,power=1):
        """
            Weights a given array of field values with the pixel volumes (not
            the meta volumes) to a given power.

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
        return x*self.get_meta_volume(total=False)**power

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
        dot = np.sum(np.conjugate(x)*y,axis=None,dtype=None,out=None)
        if(np.isreal(dot)):
            return np.asscalar(np.real(dot))
        else:
            return dot

    def calc_pseudo_dot(self,x,y,**kwargs):
        """
            Computes the (correctly weighted) inner product in the innermost
            subspace (i.e.\  the last one in the `nest` list).

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values for a field on the product space.
            y : numpy.ndarray
                Array of field values for a field on the innermost subspace.

            Returns
            -------
            pot : numpy.ndarray
                Array containing the field values of the outcome of the pseudo
                inner product.

            Other parameters
            ----------------
            target : nifty.space, *optional*
                Space in which the transform of the output field lives
                (default: None).

            Notes
            -----
            The outcome of the pseudo inner product calculation is a field
            defined on a nested space that misses the innermost subspace.
            Instead of a field on the innermost subspace, a field on the
            complete nested space can be provided as `y`, in which case the
            regular inner product is calculated and the output is a scalar.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        ## analyse (sub)array
        dotspace = None
        subspace = None
        if(np.size(y)==1)or(np.all(np.array(np.shape(y))==self.nest[-1].dim(split=True))):
            dotspace = self.nest[-1]
            if(len(self.nest)==2):
                subspace = self.nest[0]
            else:
                subspace = nested_space(self.nest[:-1])
        elif(np.all(np.array(np.shape(y))==self.para)):
            about.warnings.cprint("WARNING: computing (normal) inner product.")
            return self.calc_dot(x,self.enforce_values(y,extend=True))
        else:
            dotshape = self.nest[-1].dim(split=True)
            for ii in range(len(self.nest)-1)[::-1]:
                dotshape = np.append(self.nest[ii].dim(split=True),dotshape,axis=None)
                if(np.all(np.array(np.shape(y))==dotshape)):
                    dotspace = nested_space(self.nest[ii:])
                    if(ii<2):
                        subspace = self.nest[0]
                    else:
                        subspace = nested_space(self.nest[:ii])
                    break

        if(dotspace is None):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        y = dotspace.enforce_values(y,extend=True)

        ## weight if ...
        if(not dotspace.discrete):
            y = dotspace.calc_weight(y,power=1)
        ## pseudo inner product(s)
        x = x.reshape([subspace.dim(split=False)]+dotspace.dim(split=True).tolist(),order='C')
        pot = np.array([dotspace.calc_dot(xx,y) for xx in x],dtype=subspace.datatype).reshape(subspace.dim(split=True),order='C')
        return field(subspace,val=pot,**kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,coorder=None,**kwargs):
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
            coorder : list, *optional*
                Permutation of the subspaces.

            Notes
            -----
            Possible transformations are reorderings of the subspaces or any
            transformations acting on a single subspace.2
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        assert(self.check_codomain(codomain))

        if(self==codomain)and(coorder is None):
            return x ## T == id

        elif(isinstance(codomain,nested_space)):
            if(np.all(codomain.nest[:-1]==self.nest[:-1]))and(coorder is None):
                ## reshape
                subdim = np.prod(self.para[:-np.size(self.nest[-1].dim(split=True))],axis=0,dtype=np.int,out=None)
                x = x.reshape([subdim]+self.nest[-1].dim(split=True).tolist(),order='C')
                ## transform
                Tx = np.array([self.nest[-1].calc_transform(xx,codomain=codomain.nest[-1],**kwargs) for xx in x],dtype=codomain.datatype).reshape(codomain.dim(split=True),order='C')
            elif(len(codomain.nest)==len(self.nest)):#and(np.all([nn in self.nest for nn in codomain.nest]))and(np.all([nn in codomain.nest for nn in self.nest])):
                ## check coorder
                if(coorder is None):
                    coorder = -np.ones(len(self.nest),dtype=np.int,order='C')
                    for ii in xrange(len(self.nest)):
                        for jj in xrange(len(self.nest)):
                            if(codomain.nest[ii]==self.nest[jj]):
                                if(ii not in coorder):
                                    coorder[jj] = ii
                                    break
                    if(np.any(coorder==-1)):
                        raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))
                else:
                    coorder = np.array(coorder,dtype=np.int).reshape(len(self.nest),order='C')
                    if(np.any(np.sort(coorder,axis=0,kind="quicksort",order=None)!=np.arange(len(self.nest)))):
                        raise ValueError(about._errors.cstring("ERROR: invalid input."))
                    for ii in xrange(len(self.nest)):
                        if(codomain.nest[coorder[ii]]!=self.nest[ii]):
                            raise ValueError(about._errors.cstring("ERROR: invalid input."))
                ## compute axes permutation
                lim = np.zeros((len(self.nest),2),dtype=np.int)
                for ii in xrange(len(self.nest)):
                    lim[ii] = np.array([lim[ii-1][1],lim[ii-1][1]+np.size(self.nest[coorder[ii]].dim(split=True))])
                lim = lim[coorder]
                reorder = []
                for ii in xrange(len(self.nest)):
                    reorder += range(lim[ii][0],lim[ii][1])
                ## permute
                Tx = np.copy(x)
                for ii in xrange(len(reorder)):
                    while(reorder[ii]!=ii):
                        Tx = np.swapaxes(Tx,ii,reorder[ii])
                        ii_ = reorder[reorder[ii]]
                        reorder[reorder[ii]] = reorder[ii]
                        reorder[ii] = ii_
                ## check data type
                if(codomain.datatype!=self.datatype):
                    about.warnings.cprint("WARNING: ambiguous data type.")
            else:
                raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        return Tx.astype(codomain.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
        """
            Smoothes an array of field values by convolution with a Gaussian
            kernel, acting on the innermost subspace only (i.e.\  on the last
            entry of the `nest` list).

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values to be smoothed.
            sigma : float, *optional*
                Standard deviation of the Gaussian kernel, specified in units
                of length in position space of the innermost subspace; for
                testing: a sigma of -1 will be reset to a reasonable value
                (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations (default: 0).
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## check sigma
        if(sigma==0):
            return x
        else:
            ## reshape
            subdim = np.prod(self.para[:-np.size(self.nest[-1].dim(split=True))],axis=0,dtype=np.int,out=None)
            x = x.reshape([subdim]+self.nest[-1].dim(split=True).tolist(),order='C')
            ## smooth
            return np.array([self.nest[-1].calc_smooth(xx,sigma=sigma,**kwargs) for xx in x],dtype=self.datatype).reshape(self.dim(split=True),order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Raises an error since there is no canonical definition for the
            power spectrum on a generic product space.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.nested_space>"

    def __str__(self):
        return "nifty_core.nested_space instance\n- nest = "+str(self.nest)

##-----------------------------------------------------------------------------





##=============================================================================

class field(object):
    """
        ..         ____   __             __          __
        ..       /   _/ /__/           /  /        /  /
        ..      /  /_   __   _______  /  /    ____/  /
        ..     /   _/ /  / /   __  / /  /   /   _   /
        ..    /  /   /  / /  /____/ /  /_  /  /_/  /
        ..   /__/   /__/  \______/  \___/  \______|  class

        Basic NIFTy class for fields.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        val : {scalar, ndarray}, *optional*
            Defines field values, either to be given by a number interpreted
            as a constant array, or as an arbitrary array consistent with the
            space defined in domain or to be drawn from a random distribution
            controlled by kwargs.

        target : space, *optional*
            The space wherein the operator output lives (default: domain).


        Other Parameters
        ----------------
        random : string
            Indicates that the field values should be drawn from a certain
            distribution using a pseudo-random number generator.
            Supported distributions are:

            - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
            - "gau" (normal distribution with zero-mean and a given standard
                deviation or variance)
            - "syn" (synthesizes from a given power spectrum)
            - "uni" (uniform distribution over [vmin,vmax[)

        dev : scalar
            Sets the standard deviation of the Gaussian distribution
            (default=1).

        var : scalar
            Sets the variance of the Gaussian distribution, outranking the dev
            parameter (default=1).

        spec : {scalar, list, array, field, function}
            Specifies a power spectrum from which the field values should be
            synthesized (default=1). Can be given as a constant, or as an
            array with indvidual entries per mode.
        log : bool
            Flag specifying if the spectral binning is performed on logarithmic
            scale or not; if set, the number of used bins is set
            automatically (if not given otherwise); by default no binning
            is done (default: None).
        nbin : integer
            Number of used spectral bins; if given `log` is set to ``False``;
            integers below the minimum of 3 induce an automatic setting;
            by default no binning is done (default: None).
        binbounds : {list, array}
            User specific inner boundaries of the bins, which are preferred
            over the above parameters; by default no binning is done
            (default: None).

        vmin : scalar
            Sets the lower limit for the uniform distribution.
        vmax : scalar
            Sets the upper limit for the uniform distribution.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.

        val : {scalar, ndarray}, *optional*
            Defines field values, either to be given by a number interpreted
            as a constant array, or as an arbitrary array consistent with the
            space defined in domain or to be drawn from a random distribution
            controlled by the keyword arguments.

        target : space, *optional*
            The space wherein the operator output lives (default: domain).

    """
    def __init__(self,domain,val=None,target=None,**kwargs):
        """
            Sets the attributes for a field class instance.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        val : {scalar,ndarray}, *optional*
            Defines field values, either to be given by a number interpreted
            as a constant array, or as an arbitrary array consistent with the
            space defined in domain or to be drawn from a random distribution
            controlled by the keyword arguments.

        target : space, *optional*
            The space wherein the operator output lives (default: domain).

        Returns
        -------
        Nothing

        """
        ## check domain
        if not isinstance(domain,space):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain
        ## check codomain
        if target is None:
            target = domain.get_codomain()
        else:
            assert(self.domain.check_codomain(target))
        self.target = target
        
        if val == None:
            if kwargs == {}:
                self.val = self.domain.cast(0)
            else:
                self.val = self.domain.get_random_values(codomain=self.target, 
                                                         **kwargs)
        else:
            self.val = val
        
    
    @property
    def val(self):
        return self.__val
    
    @val.setter
    def val(self, x):
        self.__val = self.domain.cast(x)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def copy(self, domain=None, target=None):
        new_field = self.copy_empty(domain=domain, target=target)
        new_field.val = new_field.domain.cast(self.val.copy())
        return new_field
    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def copy_empty(self, domain=None, target=None, **kwargs):
        if domain == None:
            domain = self.domain
        if target == None:
            target = self.target
        new_field = field(domain=domain, target=target, **kwargs) 
        return new_field

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dim(self, split=False):
        """
            Computes the (array) dimension of the underlying space.

            Parameters
            ----------
            split : bool
                Sets the output to be either split up per axis or
                in form of total number of field entries in all
                dimensions (default=False)

            Returns
            -------
            dim : {scalar, ndarray}
                Dimension of space.

        """
        return self.domain.dim(split=split)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def cast_domain(self, newdomain, new_target=None, force=True):
        """
            Casts the domain of the field. 

            Parameters
            ----------
            newdomain : space
                New space wherein the field should live.

            new_target : space, *optional*
                Space wherein the transform of the field should live.
                When not given, target will automatically be the codomain
                of the newly casted domain (default=None).

            force : bool, *optional*
                Whether to force reshaping of the field if necessary or not
                (default=True)

            Returns
            -------
            Nothing

        """
        ## Check if the newdomain is a space
        if not isinstance(newdomain,space):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        ## Check if the datatypes match
        elif newdomain.datatype != self.domain.datatype:
            raise TypeError(about._errors.cstring(
                "ERROR: inequal data types '" +
                str(np.result_type(newdomain.datatype)) +
                "' and '" + str(np.result_type(self.domain.datatype)) +
                "'."))
        ## Check if the total dimensions match
        elif newdomain.dim() != self.domain.dim():
            raise ValueError(about._errors.cstring(
            "ERROR: dimension mismatch ( " + str(newdomain.dim()) + 
            " <> " + str(self.domain.dim()) + " )."))

        if force == True:
            self.set_domain(new_domain = newdomain, force = True)
        else:
            if not np.all(newdomain.dim(split=True) == \
                    self.domain.dim(split=True)):
                raise ValueError(about._errors.cstring(
                "ERROR: shape mismatch ( " + str(newdomain.dim(split=True)) + 
                " <> " + str(self.domain.dim(split=True)) + " )."))
            else:
                self.domain = newdomain
        ## Use the casting of the new domain in order to make the old data fit.
        self.set_val(new_val = self.val)

        ## set the target 
        if new_target == None:
            if not self.domain.check_codomain(self.target):
                if(force):
                    about.infos.cprint("INFO: codomain set to default.")
                else:
                    about.warnings.cprint("WARNING: codomain set to default.")
                self.set_target(new_target = self.domain.get_codomain())
        else:
            self.set_target(new_target = new_target, force = force)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_val(self, new_val):
        """
            Resets the field values.

            Parameters
            ----------
            new_val : {scalar, ndarray}
                New field values either as a constant or an arbitrary array.

        """
        '''
        if(new_val is None):
            self.val = np.zeros(self.dim(split=True),dtype=self.domain.datatype,order='C')
        else:
            self.val = self.domain.enforce_values(new_val,extend=True)
        '''
        self.val = self.domain.cast(new_val)
        return self.val
        
    def get_val(self):
        return self.val
        
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_domain(self, new_domain=None, force=False):
        if new_domain is None:
            new_domain = self.target.get_codomain()
        elif force == False:
            assert(self.target.check_codomain(new_domain))
        self.domain = new_domain
        return self.domain
        

    def set_target(self, new_target=None, force=False):
        """
            Resets the codomain of the field.

            Parameters
            ----------
            new_target : space
                 The new space wherein the transform of the field should live.
                 (default=None).

        """
        ## check codomain
        if new_target is None:
            new_target = self.domain.get_codomain()
        elif force == False:
            assert(self.domain.check_codomain(new_target))
        self.target = new_target
        return self.target
        
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def weight(self, power=1, overwrite=False):
        """
            Returns the field values, weighted with the volume factors to a
            given power. The field values will optionally be overwritten.

            Parameters
            ----------
            power : scalar, *optional*
                Specifies the optional power coefficient to which the field
                values are taken (default=1).

            overwrite : bool, *optional*
                Whether to overwrite the field values or not (default: False).

            Returns
            -------
            field   : field, *optional*
                If overwrite is False, the weighted field is returned.
                Otherwise, nothing is returned.

        """
        if overwrite == True:
            new_field = self
        else:
            new_field = self.copy_empty()
            
        new_field.set_val(new_val = self.domain.calc_weight(self.get_val(), 
                                                            power = power))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dot(self, x=None):
        """
            Computes the inner product of the field with a given object
            implying the correct volume factor needed to reflect the
            discretization of the continuous fields.

            Parameters
            ----------
            x : {scalar, ndarray, field}, *optional*
                The object with which the inner product is computed
                (default=None).

            Returns
            -------
            dot : scalar
                The result of the inner product.

        """
        ## Case 1: x equals None        
        if x == None:
            return None

        ## Case 2: x is a field         
        elif isinstance(x, field):
            ## if x lives in the cospace, transform it an make a
            ## recursive call
            if self.domain.fourier != x.domain.fourier:
                return self.dot(x = x.transform())
            else:
            ## whether the domain matches exactly or not:
            ## extract the data from x and try to dot with this
                return self.dot(x = x.get_val())

        ## Case 3: x is something else
        else:
            ## Cast the input in order to cure datatype and shape differences
            casted_x = self.cast(x)
            ## Compute the dot respecting the fact of discrete/continous spaces             
            if self.domain.discrete == True:
                return self.domain.calc_dot(self.get_val(), casted_x)
            else:
                return self.domain.calc_dot(self.get_val(), 
                                            self.domain.calc_weight(
                                                casted_x,
                                                power=1))

    def norm(self, q=0.5):
        """
            Computes the Lq-norm of the field values.

            Parameters
            ----------
            q : scalar
                Parameter q of the Lq-norm (default: 2).

            Returns
            -------
            norm : scalar
                The Lq-norm of the field values.

        """
        if q == 0.5:
            return (self.dot(x = self))**(1/2)
        else:
            return self.dot(x = self**(q-1))**(1/q)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## TODO: rework the nested space semantics in order to become compatible 
    ## with the usual space interface

    def pseudo_dot(self,x=1,**kwargs):
        """
            Computes the pseudo inner product of the field with a given object
            implying the correct volume factor needed to reflect the
            discretization of the continuous fields. This method specifically
            handles the inner products of fields defined over a
            :py:class:`nested_space`.

            Parameters
            ----------
            x : {scalar, ndarray, field}, *optional*
                The object with which the inner product is computed
                (default=None).

            Other Parameters
            ----------------
            target : space, *optional*
                space wherein the transform of the output field should live
                (default: None).

            Returns
            -------
            pot : ndarray
                The result of the pseudo inner product.

            Examples
            --------
            Pseudo inner product of a field defined over a nested space with
            a simple field defined over a rg_space.

            >>> from nifty import *
            >>> space = rg_space(2)
            >>> nspace = nested_space([space,space])
            >>> nval = array([[1,2],[3,4]])
            >>> nfield = nifty.field(domain = nspace, val = nval)
            >>> val = array([1,1])
            >>> nfield.pseudo_dot(x=val).val
            array([ 1.5,  3.5])

        """
        ## check attribute
        if(not hasattr(self.domain,"calc_pseudo_dot")):
            if(isinstance(x,field)):
                if(hasattr(x.domain,"calc_pseudo_dot")):
                    return x.pseudo_dot(x=self,**kwargs)
            about.warnings.cprint("WARNING: computing (normal) inner product.")
            return self.dot(x=x)
        ## strip field (calc_pseudo_dot handles subspace)
        if(isinstance(x,field)):
            if(np.size(x.dim(split=True))>np.size(self.dim(split=True))): ## switch
                return x.pseudo_dot(x=self,**kwargs)
            else:
                try:
                    return self.pseudo_dot(x=x.val,**kwargs)
                except(TypeError,ValueError):
                    try:
                        return self.pseudo_dot(x=x.transform(target=x.target,overwrite=False).val,**kwargs)
                    except(TypeError,ValueError):
                        raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        ## pseudo inner product (calc_pseudo_dot handles weights)
        else:
            if(np.isscalar(x)):
                x = np.array([x],dtype=self.domain.datatype)
            else:
                x = np.array(x,dtype=self.domain.datatype)

            if(np.size(x)>self.dim(split=False)):
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(x))+" <> "+str(self.dim(split=False))+" )."))
            elif(np.size(x)==self.dim(split=False)):
                about.warnings.cprint("WARNING: computing (normal) inner product.")
                return self.dot(x=x)
            else:
                return self.domain.calc_pseudo_dot(self.val,x,**kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## TODO: rework the nested space semantics in order to become compatible 
    ## with the usual space interface

    def tensor_dot(self,x=None,**kwargs):
        """
            Computes the tensor product of a field defined on a arbitrary domain
            with a given object defined on another arbitrary domain.

            Parameters
            ----------
            x : {scalar, ndarray, field}, *optional*
                The object with which the inner product is computed
                (default=None).

            Other Parameters
            ----------------
            target : space, *optional*
                space wherein the transform of the output field should live
                (default: None).

            Returns
            -------
            tot : field
                The result of the tensor product, a field defined over a nested
                space.

        """
        if(x is None):
            return self
        elif(isinstance(x,field)):
            return field(nested_space([self.domain,x.domain]),val=np.tensordot(self.val,x.val,axes=0),**kwargs)
        else:
            return field(nested_space([self.domain,self.domain]),val=np.tensordot(self.val,self.domain.enforce_values(x,extend=True),axes=0),**kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def conjugate(self, inplace=False):
        """
            Computes the complex conjugate of the field.

            Returns
            -------
            cc : field
                The complex conjugated field.

        """
        if inplace == True:
            work_field = self
        else:
            work_field = self.copy_empty()
        work_field.set_val(new_val = self.val.conjugate())
        
        return work_field
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def transform(self, target=None, overwrite=False,  **kwargs):
        """
            Computes the transform of the field using the appropriate conjugate
            transformation.

            Parameters
            ----------
            target : space, *optional*
                Domain of the transform of the field (default:self.target)

            overwrite : bool, *optional*
                Whether to overwrite the field or not (default: False).

            Other Parameters
            ----------------
            iter : scalar
                Number of iterations (default: 0)

            Returns
            -------
            field : field, *optional*
                If overwrite is False, the transformed field is returned.
                Otherwise, nothing is returned.

        """
        if(target is None):
            target = self.target
        else:
            assert(self.domain.check_codomain(target))
        
        new_val = self.domain.calc_transform(self.val, 
                                                  codomain=target,
                                                  **kwargs)        
                                                  
        if overwrite == True:
            return_field = self
            return_field.set_target(new_target = self.domain, force = True)
            return_field.set_domain(new_domain = target, force = True)
        else:
            return_field = self.copy_empty(domain = self.target, 
                                           target = self.domain)
        return_field.set_val(new_val = new_val)
        
        return return_field
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def smooth(self, sigma=0, overwrite=False, **kwargs):
        """
            Smoothes the field by convolution with a Gaussian kernel.

            Parameters
            ----------
            sigma : scalar, *optional*
                standard deviation of the Gaussian kernel specified in units of
                length in position space (default: 0)

            overwrite : bool, *optional*
                Whether to overwrite the field or not (default: False).

            Other Parameters
            ----------------
            iter : scalar
                Number of iterations (default: 0)

            Returns
            -------
            field : field, *optional*
                If overwrite is False, the transformed field is returned.
                Otherwise, nothing is returned.

        """
        if overwrite == True:
            new_field = self
        else:
            new_field = self.copy_empty()
            
        new_field.set_val(new_val = self.domain.calc_smooth(self.get_val(),
                                                            sigma = sigma,
                                                            **kwargs))
        return new_field                                                            

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def power(self, **kwargs):
        """
            Computes the power spectrum of the field values.

            Other Parameters
            ----------------
            pindex : ndarray, *optional*
                Specifies the indexing array for the distribution of
                indices in conjugate space (default: None).
            kindex : numpy.ndarray, *optional*
                Scale corresponding to each band in the power spectrum
                (default: None).
            rho : scalar
                Number of degrees of freedom per irreducible band
                (default=None).
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
            iter : scalar
                Number of iterations (default: 0)

            Returns
            -------
            spec : ndarray
                Returns the power spectrum.

        """
        if("codomain" in kwargs):
            kwargs.__delitem__("codomain")
            about.warnings.cprint("WARNING: codomain was removed from kwargs.")

        return self.domain.calc_power(self.get_val(),
                                      codomain = self.target,
                                      **kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def hat(self):
        """
            Translates the field into a diagonal operator.

            Returns
            -------
            D : operator
                The new diagonal operator instance.

        """
        from nifty.operators.nifty_operators import diagonal_operator
        return diagonal_operator(domain=self.domain,
                                 diag=self.get_val(),
                                 bare=False)

    def inverse_hat(self):
        """
            Translates the inverted field into a diagonal operator.

            Returns
            -------
            D : operator
                The new diagonal operator instance.

        """
        if(np.any(self.val==0)):
            raise AttributeError(
                about._errors.cstring("ERROR: singular operator."))
        else:
            from nifty.operators.nifty_operators import diagonal_operator
            return diagonal_operator(domain=self.domain,
                                     diag=(1/self).get_val(),
                                     bare=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def plot(self,**kwargs):
        """
            Plots the field values using matplotlib routines.

            Other Parameters
            ----------------
            title : string
                Title of the plot (default= "").
            vmin : scalar
                Minimum value displayed (default=min(x)).
            vmax : scalar
                Maximum value displayed (default=max(x)).
            power : bool
                Whether to plot the power spectrum or the array (default=None).
            unit : string
                The unit of the field values (default="").
            norm : scalar
                A normalization (default=None).
            cmap : cmap
                A color map (default=None).
            cbar : bool
                Whether to show the color bar or not (default=True).
            other : {scalar, ndarray, field}
                Object or tuple of objects to be added (default=None).
            legend : bool
                Whether to show the legend or not (default=False).
            mono : bool
                Whether to plot the monopol of the power spectrum or not
                (default=True).
            save : string, *optional*
                Valid file name where the figure is to be stored, by default
                the figure is not saved (default: False).
            error : {scalar, ndarray, field}
                object indicating some confidence intervall (default=None).
            iter : scalar
                Number of iterations (default: 0).
            kindex : scalar
                The spectral index per irreducible band (default=None).
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

            Notes
            -----
            The applicability of the keyword arguments depends on the
            respective space on which the field is defined. Confer to the
            corresponding :py:meth:`get_plot` method.

        """
        ## if a save path is given, set pylab to not-interactive
        remember_interactive = pl.isinteractive()
        pl.matplotlib.interactive(not bool(kwargs.get("save", False)))

        if "codomain" in kwargs:
            kwargs.__delitem__("codomain")
            about.warnings.cprint("WARNING: codomain was removed from kwargs.")

        ## draw/save the plot(s)
        self.domain.get_plot(self.val, codomain=self.target, **kwargs)
        
        ## restore the pylab interactiveness
        pl.matplotlib.interactive(remember_interactive)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.field>"

    def __str__(self):
        minmax = [self.val.amin(), self.val.amax()]
        mean = self.val.mean()
        return "nifty_core.field instance\n- domain      = " + \
                repr(self.domain) + \
                "\n- val         = [...]" + \
                "\n  - min.,max. = " + str(minmax) + \
                "\n  - mean = " + str(mean) + \
                "\n- target      = " + repr(self.target)


    def __len__(self):
        return int(self.dim(split=True)[0])

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __getitem__(self,key):
        return self.domain.getitem(self.val, key)

    def __setitem__(self,key,value):
        self.domain.setitem(self.val, value, key)


    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def apply_scalar_function(self, function, inplace=False):
        if inplace == True:
            temp = self
        else:
            temp = self.copy_empty()
        data_object = self.domain.apply_scalar_function(self.val,\
                                                        function, inplace)
        temp.set_val(data_object)
        return temp
        
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def min(self,ignore=False,**kwargs):
        """
            Returns the minimum of the field values.

            Parameters
            ----------
            ignore : bool
                Whether to ignore NANs or not (default: False).

            Returns
            -------
            amin : {scalar, ndarray}
                Minimum field value.

            See Also
            --------
            np.amin, np.nanmin

        """
        if ignore == True:
            return self.domain.unary_operation(self.val, op='nanmin', **kwargs)
        else:
            return self.domain.unary_operation(self.val, op='min', **kwargs)
        
    def max(self,ignore=False,**kwargs):
        """
            Returns the maximum of the field values.

            Parameters
            ----------
            ignore : bool
                Whether to ignore NANs or not (default: False).

            Returns
            -------
            amax : {scalar, ndarray}
                Maximum field value.

            See Also
            --------
            np.amax, np.nanmax

        """
        if ignore == True:
            return self.domain.unary_operation(self.val, op='nanmax', **kwargs)
        else:
            return self.domain.unary_operation(self.val, op='max', **kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def med(self,**kwargs):
        """
            Returns the median of the field values.

            Returns
            -------
            med : scalar
                Median field value.

            See Also
            --------
            np.median

        """
        return self.domain.unary_operation(self.val, op='median', **kwargs)

    def mean(self,**kwargs):
        """
            Returns the mean of the field values.

            Returns
            -------
            mean : scalar
                Mean field value.

            See Also
            --------
            np.mean

        """
        return self.domain.unary_operation(self.val, op='mean', **kwargs)

    def std(self,**kwargs):
        """
            Returns the standard deviation of the field values.

            Returns
            -------
            std : scalar
                Standard deviation of the field values.

            See Also
            --------
            np.std

        """
        return self.domain.unary_operation(self.val, op='std', **kwargs)

    def var(self,**kwargs):
        """
            Returns the variance of the field values.

            Returns
            -------
            var : scalar
                Variance of the field values.

            See Also
            --------
            np.var

        """
        return self.domain.unary_operation(self.val, op='var', **kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def argmin(self,split=True,**kwargs):
        """
            Returns the index of the minimum field value.

            Parameters
            ----------
            split : bool
                Whether to split (unravel) the flat index or not; does not
                apply to multiple indices along some axis (default: True).

            Returns
            -------
            ind : {integer, tuple, array}
                Index of the minimum field value being an integer for
                one-dimensional fields, a tuple for multi-dimensional fields,
                and an array in case minima along some axis are requested.

            See Also
            --------
            np.argmax, np.argmin

        """
        if split == True:
            return self.domain.unary_operation(self.val, op='argmin', **kwargs)
        else:
            return self.domain.unary_operation(self.val, 
                                               op='argmin_flat', **kwargs)
        
    def argmax(self,split=True,**kwargs):
        """
            Returns the index of the maximum field value.

            Parameters
            ----------
            split : bool
                Whether to split (unravel) the flat index or not; does not
                apply to multiple indices along some axis (default: True).

            Returns
            -------
            ind : {integer, tuple, array}
                Index of the maximum field value being an integer for
                one-dimensional fields, a tuple for multi-dimensional fields,
                and an array in case maxima along some axis are requested.

            See Also
            --------
            np.argmax, np.argmin

        """
        if split == True:
            return self.domain.unary_operation(self.val, op='argmax', **kwargs)
        else:
            return self.domain.unary_operation(self.val, 
                                               op='argmax_flat', **kwargs)



    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

    def __pos__(self):
        new_field = self.copy_empty()
        new_field.val = self.domain.unary_operation(self.val, op='pos')
        return new_field

    def __neg__(self):
        new_field = self.copy_empty()
        new_field.val = self.domain.unary_operation(self.val, op='neg')
        return new_field

    def __abs__(self):
        new_field = self.copy_empty()
        new_field.val = self.domain.unary_operation(self.val, op='abs')
        return new_field

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __binary_helper__(self, other, op='None'):
        new_val = self.domain.binary_operation(self.val, other, op=op, cast=0)
        new_field = self.copy_empty()        
        new_field.val = new_val
        return new_field
    
    def __inplace_binary_helper__(self, other, op='None'):
        self.val = self.domain.binary_operation(self.val, other, op=op, 
                                                cast=0)
        return self
    
    def __add__(self, other):
        return self.__binary_helper__(other, op='add')
    __radd__ = __add__
    def __iadd__(self, other):
        return self.__inplace_binary_helper__(other, op='iadd')
    
    
    def __sub__(self, other):
        return self.__binary_helper__(other, op='sub')
    def __rsub__(self, other):
        return self.__binary_helper__(other, op='rsub')
    def __isub__(self, other):
        return self.__inplace_binary_helper__(other, op='isub')
        
    def __mul__(self, other):
        return self.__binary_helper__(other, op='mul')
    __rmul__ = __mul__
    def __imul__(self, other):
        return self.__inplace_binary_helper__(other, op='imul')
        
    def __div__(self, other):
        return self.__binary_helper__(other, op='div')
    def __rdiv__(self, other):
        return self.__binary_helper__(other, op='rdiv')
    def __idiv__(self, other):
        return self.__inplace_binary_helper__(other, op='idiv')    
    __truediv__ = __div__    
    __itruediv__ = __idiv__
    
    def __pow__(self, other):
        return self.__binary_helper__(other, op='pow')
    def __rpow__(self, other):
        return self.__binary_helper__(other, op='rpow')
    def __ipow__(self, other):
        return self.__inplace_binary_helper__(other, op='ipow')    
    


##=============================================================================



