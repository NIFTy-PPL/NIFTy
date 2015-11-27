# NIFTY (Numerical Information Field Theory) has been developed at the
# Max-Planck-Institute for Astrophysics.
##
# Copyright (C) 2013 Max-Planck-Society
##
# Author: Marco Selig
# Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
##
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

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
    point_space_paradict

from keepers import about,\
    global_configuration as gc,\
    global_dependency_injector as gdi

from nifty_random import random
from nifty.nifty_mpi_data import distributed_data_object,\
    STRATEGIES as DISTRIBUTION_STRATEGIES

import nifty.nifty_utilities as utilities

POINT_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


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
        dtype : numpy.dtype, *optional*
            Data type of the field values for a field defined on this space
            (default: numpy.float64).
        datamodel :

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
            This is a freeform list of parameters that derivatives of the space
            class can use.
        dtype : numpy.dtype
            Data type of the field values for a field defined on this space.
        discrete : bool
            Whether the space is inherently discrete (true) or a discretization
            of a continuous space (false).
        vol : numpy.ndarray
            An array of pixel volumes, only one component if the pixels all
            have the same volume.
    """

    def __init__(self):
        """
            Sets the attributes for a space class instance.

            Parameters
            ----------
            dtype : numpy.dtype, *optional*
                Data type of the field values for a field defined on this space
                (default: numpy.float64).
            datamodel :

            Returns
            -------
            None
        """
        self.paradict = space_paradict()

    @property
    def para(self):
        return self.paradict['default']

    @para.setter
    def para(self, x):
        self.paradict['default'] = x

    def __hash__(self):
        return hash(())

    def _identifier(self):
        """
        _identiftier returns an object which contains all information needed
        to uniquely idetnify a space. It returns a (immutable) tuple which
        therefore can be compared.
        """
        return tuple(sorted(vars(self).items()))

    def __eq__(self, x):
        if isinstance(x, type(self)):
            return self._identifier() == x._identifier()
        else:
            return False

    def __ne__(self, x):
        return not self.__eq__(x)

    def __len__(self):
        return int(self.get_dim(split=False))

    def copy(self):
        return space(para=self.para,
                     dtype=self.dtype)

    def getitem(self, data, key):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'getitem'."))

    def setitem(self, data, key):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'getitem'."))

    def apply_scalar_function(self, x, function, inplace=False):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'apply_scalar_function'."))

    def unary_operation(self, x, op=None):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'unary_operation'."))

    def binary_operation(self, x, y, op=None):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'binary_operation'."))

    def get_shape(self):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'shape'."))

    def get_dim(self, split=False):
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
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'dim'."))

    def get_dof(self):
        """
            Computes the number of degrees of freedom of the space.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'dof'."))

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
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'cast'."))

    # TODO: Move enforce power into power_indices class
    def enforce_power(self, spec, **kwargs):
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
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'enforce_power'."))

    def check_codomain(self, codomain):
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
        if codomain is None:
            return False
        else:
            raise NotImplementedError(about._errors.cstring(
                "ERROR: no generic instance method 'check_codomain'."))

    def get_codomain(self, **kwargs):
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
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'get_codomain'."))

    def get_random_values(self, **kwargs):
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
                - "gau" (normal distribution with zero-mean and a given
                    standard deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            spec : {scalar, list, numpy.ndarray, nifty.field, function},
                    *optional*
                Power spectrum (default: 1).
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index of each band
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band (default: None).
            codomain : nifty.space, *optional*
                A compatible codomain with power indices (default: None).
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'get_random_values'."))

    def calc_weight(self, x, power=1):
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
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'calc_weight'."))

    def get_weight(self, power=1):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'get_weight'."))

    def calc_norm(self, x, q):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'norm'."))

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
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'dot'."))

    def calc_transform(self, x, codomain=None, **kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                codomain space to which the transformation shall map
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
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'calc_transform'."))

    def calc_smooth(self, x, sigma=0, **kwargs):
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
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'calc_smooth'."))

    def calc_power(self, x, **kwargs):
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
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'calc_power'."))

    def calc_real_Q(self, x):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'calc_real_Q'."))

    def calc_bincount(self, x, weights=None, minlength=None):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'calc_bincount'."))

    def get_plot(self, x, **kwargs):
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
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).
            iter : int, *optional*
                Number of iterations (default: 0).

        """
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'get_plot'."))

    def __repr__(self):
        string = ""
        string += str(type(self)) + "\n"
        string += "paradict: " + str(self.paradict) + "\n"
        return string

    def __str__(self):
        return self.__repr__()


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
        dtype : numpy.dtype, *optional*
            Data type of the field values (default: None).

        Attributes
        ----------
        para : numpy.ndarray
            Array containing the number of points.
        dtype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Parameter captioning the fact that a :py:class:`point_space` is
            always discrete.
        vol : numpy.ndarray
            Pixel volume of the :py:class:`point_space`, which is always 1.
    """

    def __init__(self, num, dtype=np.dtype('float'), datamodel='fftw',
                 comm=gc['default_comm']):
        """
            Sets the attributes for a point_space class instance.

            Parameters
            ----------
            num : int
                Number of points.
            dtype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.float64).

            Returns
            -------
            None.
        """
        self._cache_dict = {'check_codomain': {}}
        self.paradict = point_space_paradict(num=num)

        # parse dtype
        dtype = np.dtype(dtype)
        if dtype not in [np.dtype('bool'),
                         np.dtype('int16'),
                         np.dtype('int32'),
                         np.dtype('int64'),
                         np.dtype('float32'),
                         np.dtype('float64'),
                         np.dtype('complex64'),
                         np.dtype('complex128')]:
            raise ValueError(about._errors.cstring(
                             "WARNING: incompatible dtype: " + str(dtype)))
        self.dtype = dtype

        if datamodel not in ['np'] + POINT_DISTRIBUTION_STRATEGIES:
            about._errors.cstring("WARNING: datamodel set to default.")
            self.datamodel = \
                gc['default_distribution_strategy']
        else:
            self.datamodel = datamodel

        self.comm = self._parse_comm(comm)
        self.discrete = True
#        self.harmonic = False
        self.distances = (np.float(1),)

    @property
    def para(self):
        temp = np.array([self.paradict['num']], dtype=int)
        return temp

    @para.setter
    def para(self, x):
        self.paradict['num'] = x[0]

    def __hash__(self):
        # Extract the identifying parts from the vars(self) dict.
        result_hash = 0
        for (key, item) in vars(self).items():
            if key in ['_cache_dict']:
                continue
            result_hash ^= item.__hash__() * hash(key)
        return result_hash

    def _identifier(self):
        # Extract the identifying parts from the vars(self) dict.
        temp = [(ii[0],
                 ((lambda x: x[1].__hash__() if x[0] == 'comm' else x)(ii)))
                for ii in vars(self).iteritems()
                if ii[0] not in ['_cache_dict']
                ]
        # Return the sorted identifiers as a tuple.
        return tuple(sorted(temp))

    def _parse_comm(self, comm):
        # check if comm is a string -> the name of comm is given
        # -> Extract it from the mpi_module
        if isinstance(comm, str):
            if gc.validQ('default_comm', comm):
                result_comm = getattr(gdi[gc['mpi_module']], comm)
            else:
                raise ValueError(about._errors.cstring(
                    "ERROR: The given communicator-name is not supported."))
        # check if the given comm object is an instance of default Intracomm
        else:
            if isinstance(comm, gdi[gc['mpi_module']].Intracomm):
                result_comm = comm
            else:
                raise ValueError(about._errors.cstring(
                    "ERROR: The given comm object is not an instance of the " +
                    "default-MPI-module's Intracomm Class."))
        return result_comm

    def copy(self):
        return point_space(num=self.paradict['num'],
                           dtype=self.dtype,
                           datamodel=self.datamodel,
                           comm=self.comm)

    def getitem(self, data, key):
        return data[key]

    def setitem(self, data, update, key):
        data[key] = update

    def apply_scalar_function(self, x, function, inplace=False):
        if self.datamodel == 'np':
            if not inplace:
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

        elif self.datamodel in POINT_DISTRIBUTION_STRATEGIES:
            return x.apply_scalar_function(function, inplace=inplace)
        else:
            raise NotImplementedError(about._errors.cstring(
                "ERROR: function is not implemented for given datamodel."))

    def unary_operation(self, x, op='None', **kwargs):
        """
        x must be a numpy array which is compatible with the space!
        Valid operations are

        """
        if self.datamodel == 'np':
            def _argmin(z, **kwargs):
                ind = np.argmin(z, **kwargs)
                if np.isscalar(ind):
                    ind = np.unravel_index(ind, z.shape, order='C')
                    if(len(ind) == 1):
                        return ind[0]
                return ind

            def _argmax(z, **kwargs):
                ind = np.argmax(z, **kwargs)
                if np.isscalar(ind):
                    ind = np.unravel_index(ind, z.shape, order='C')
                    if(len(ind) == 1):
                        return ind[0]
                return ind

            translation = {'pos': lambda y: getattr(y, '__pos__')(),
                           'neg': lambda y: getattr(y, '__neg__')(),
                           'abs': lambda y: getattr(y, '__abs__')(),
                           'real': lambda y: getattr(y, 'real'),
                           'imag': lambda y: getattr(y, 'imag'),
                           'nanmin': np.nanmin,
                           'amin': np.amin,
                           'nanmax': np.nanmax,
                           'amax': np.amax,
                           'median': np.median,
                           'mean': np.mean,
                           'std': np.std,
                           'var': np.var,
                           'argmin': _argmin,
                           'argmin_flat': np.argmin,
                           'argmax': _argmax,
                           'argmax_flat': np.argmax,
                           'conjugate': np.conjugate,
                           'sum': np.sum,
                           'prod': np.prod,
                           'unique': np.unique,
                           'copy': np.copy,
                           'copy_empty': np.empty_like,
                           'isnan': np.isnan,
                           'isinf': np.isinf,
                           'isfinite': np.isfinite,
                           'nan_to_num': np.nan_to_num,
                           'all': np.all,
                           'any': np.any,
                           'None': lambda y: y}

        elif self.datamodel in POINT_DISTRIBUTION_STRATEGIES:
            translation = {'pos': lambda y: getattr(y, '__pos__')(),
                           'neg': lambda y: getattr(y, '__neg__')(),
                           'abs': lambda y: getattr(y, '__abs__')(),
                           'real': lambda y: getattr(y, 'real'),
                           'imag': lambda y: getattr(y, 'imag'),
                           'nanmin': lambda y: getattr(y, 'nanmin')(),
                           'amin': lambda y: getattr(y, 'amin')(),
                           'nanmax': lambda y: getattr(y, 'nanmax')(),
                           'amax': lambda y: getattr(y, 'amax')(),
                           'median': lambda y: getattr(y, 'median')(),
                           'mean': lambda y: getattr(y, 'mean')(),
                           'std': lambda y: getattr(y, 'std')(),
                           'var': lambda y: getattr(y, 'var')(),
                           'argmin': lambda y: getattr(y, 'argmin_nonflat')(),
                           'argmin_flat': lambda y: getattr(y, 'argmin')(),
                           'argmax': lambda y: getattr(y, 'argmax_nonflat')(),
                           'argmax_flat': lambda y: getattr(y, 'argmax')(),
                           'conjugate': lambda y: getattr(y, 'conjugate')(),
                           'sum': lambda y: getattr(y, 'sum')(),
                           'prod': lambda y: getattr(y, 'prod')(),
                           'unique': lambda y: getattr(y, 'unique')(),
                           'copy': lambda y: getattr(y, 'copy')(),
                           'copy_empty': lambda y: getattr(y, 'copy_empty')(),
                           'isnan': lambda y: getattr(y, 'isnan')(),
                           'isinf': lambda y: getattr(y, 'isinf')(),
                           'isfinite': lambda y: getattr(y, 'isfinite')(),
                           'nan_to_num': lambda y: getattr(y, 'nan_to_num')(),
                           'all': lambda y: getattr(y, 'all')(),
                           'any': lambda y: getattr(y, 'any')(),
                           'None': lambda y: y}
        else:
            raise NotImplementedError(about._errors.cstring(
                "ERROR: function is not implemented for given datamodel."))

        return translation[op](x, **kwargs)

    def binary_operation(self, x, y, op='None', cast=0):

        translation = {'add': lambda z: getattr(z, '__add__'),
                       'radd': lambda z: getattr(z, '__radd__'),
                       'iadd': lambda z: getattr(z, '__iadd__'),
                       'sub': lambda z: getattr(z, '__sub__'),
                       'rsub': lambda z: getattr(z, '__rsub__'),
                       'isub': lambda z: getattr(z, '__isub__'),
                       'mul': lambda z: getattr(z, '__mul__'),
                       'rmul': lambda z: getattr(z, '__rmul__'),
                       'imul': lambda z: getattr(z, '__imul__'),
                       'div': lambda z: getattr(z, '__div__'),
                       'rdiv': lambda z: getattr(z, '__rdiv__'),
                       'idiv': lambda z: getattr(z, '__idiv__'),
                       'pow': lambda z: getattr(z, '__pow__'),
                       'rpow': lambda z: getattr(z, '__rpow__'),
                       'ipow': lambda z: getattr(z, '__ipow__'),
                       'ne': lambda z: getattr(z, '__ne__'),
                       'lt': lambda z: getattr(z, '__lt__'),
                       'le': lambda z: getattr(z, '__le__'),
                       'eq': lambda z: getattr(z, '__eq__'),
                       'ge': lambda z: getattr(z, '__ge__'),
                       'gt': lambda z: getattr(z, '__gt__'),
                       'None': lambda z: lambda u: u}

        if (cast & 1) != 0:
            x = self.cast(x)
        if (cast & 2) != 0:
            y = self.cast(y)

        return translation[op](x)(y)

    def get_shape(self):
        return (self.paradict['num'],)

    def get_dim(self):
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
        return np.prod(self.get_shape())

    def get_dof(self, split=False):
        """
            Computes the number of degrees of freedom of the space, i.e./  the
            number of points for real-valued fields and twice that number for
            complex-valued fields.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        if split:
            dof = self.get_shape()
            if issubclass(self.dtype.type, np.complexfloating):
                dof = tuple(np.array(dof)*2)
        else:
            dof = self.get_dim()
            if issubclass(self.dtype.type, np.complexfloating):
                dof = dof * 2
        return dof

    def get_vol(self, split=False):
        if split:
            return self.distances
        else:
            return np.prod(self.distances)

    def get_meta_volume(self, split=False):
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
        if not split:
            return self.get_dim() * self.get_vol()
        else:
            mol = self.cast(1, dtype=np.dtype('float'))
            return self.calc_weight(mol, power=1)

    def cast(self, x=None, dtype=None, **kwargs):
        if dtype is not None:
            dtype = np.dtype(dtype)

        # If x is a field, extract the data and do a recursive call
        if isinstance(x, field):
            # Check if the domain matches
            if self != x.domain:
                about.warnings.cflush(
                    "WARNING: Getting data from foreign domain!")
            # Extract the data, whatever it is, and cast it again
            return self.cast(x.val,
                             dtype=dtype,
                             **kwargs)

        if self.datamodel in POINT_DISTRIBUTION_STRATEGIES:
            return self._cast_to_d2o(x=x,
                                     dtype=dtype,
                                     **kwargs)
        elif self.datamodel == 'np':
            return self._cast_to_np(x=x,
                                    dtype=dtype,
                                    **kwargs)
        else:
            raise NotImplementedError(about._errors.cstring(
                "ERROR: function is not implemented for given datamodel."))

    def _cast_to_d2o(self, x, dtype=None, **kwargs):
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
        if dtype is None:
            dtype = self.dtype

        # Case 1: x is a distributed_data_object
        if isinstance(x, distributed_data_object):
            to_copy = False

            # Check the shape
            if np.any(np.array(x.shape) != np.array(self.get_shape())):
                # Check if at least the number of degrees of freedom is equal
                if x.get_dim() == self.get_dim():
                    try:
                        temp = x.copy_empty(global_shape=self.get_shape())
                        temp.set_local_data(x.get_local_data(), copy=False)
                    except:
                        # If the number of dof is equal or 1, use np.reshape...
                        about.warnings.cflush(
                            "WARNING: Trying to reshape the data. This " +
                            "operation is expensive as it consolidates the " +
                            "full data!\n")
                        temp = x.get_full_data()
                        temp = np.reshape(temp, self.get_shape())
                    # ... and cast again
                    return self._cast_to_d2o(temp,
                                             dtype=dtype,
                                             **kwargs)

                else:
                    raise ValueError(about._errors.cstring(
                        "ERROR: Data has incompatible shape!"))

            # Check the dtype
            if x.dtype != dtype:
                if x.dtype > dtype:
                    about.warnings.cflush(
                        "WARNING: Datatypes are of conflicting precision " +
                        "(own: " + str(dtype) + " <> foreign: " +
                        str(x.dtype) + ") and will be casted! Potential " +
                        "loss of precision!\n")
                to_copy = True

            # Check the distribution_strategy
            if x.distribution_strategy != self.datamodel:
                to_copy = True

            if to_copy:
                temp = x.copy_empty(dtype=dtype,
                                    distribution_strategy=self.datamodel)
                temp.inject((slice(None),), x, (slice(None),))
                temp.hermitian = x.hermitian
                x = temp

            return x

        # Case 2: x is something else
        # Use general d2o casting
        else:
            x = distributed_data_object(x,
                                        global_shape=self.get_shape(),
                                        dtype=dtype,
                                        distribution_strategy=self.datamodel)
            # Cast the d2o
            return self.cast(x, dtype=dtype)

    def _cast_to_np(self, x, dtype=None, **kwargs):
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
        if dtype is None:
            dtype = self.dtype

        # Case 1: x is a distributed_data_object
        if isinstance(x, distributed_data_object):
            # Extract the data
            temp = x.get_full_data()
            # Cast the resulting numpy array again
            return self._cast_to_np(temp,
                                    dtype=dtype,
                                    **kwargs)

        # Case 2: x is a distributed_data_object
        elif isinstance(x, np.ndarray):
            # Check the shape
            if np.any(np.array(x.shape) != np.array(self.get_shape())):
                # Check if at least the number of degrees of freedom is equal
                if x.size == self.get_dim():
                    # If the number of dof is equal or 1, use np.reshape...
                    temp = x.reshape(self.get_shape())
                    # ... and cast again
                    return self._cast_to_np(temp,
                                            dtype=dtype,
                                            **kwargs)
                elif x.size == 1:
                    temp = np.empty(shape=self.get_shape(),
                                    dtype=dtype)
                    temp[:] = x
                    return self._cast_to_np(temp,
                                            dtype=dtype,
                                            **kwargs)
                else:
                    raise ValueError(about._errors.cstring(
                        "ERROR: Data has incompatible shape!"))

            # Check the dtype
            if x.dtype != dtype:
                if x.dtype > dtype:
                    about.warnings.cflush(
                        "WARNING: Datatypes are of conflicting precision " +
                        " (own: " + str(dtype) + " <> foreign: " +
                        str(x.dtype) + ") and will be casted! Potential " +
                        "loss of precision!\n")
                # Fix the datatype...
                temp = x.astype(dtype)
                # ... and cast again
                return self._cast_to_np(temp,
                                        dtype=dtype,
                                        **kwargs)

            return x

        # Case 3: x is something else
        # Use general numpy casting
        else:
            temp = np.empty(self.get_shape(), dtype=dtype)
            if x is not None:
                temp[:] = x
            return self._cast_to_np(temp, dtype=dtype)

    def enforce_power(self, spec, **kwargs):
        """
            Raises an error since the power spectrum is ill-defined for point
            spaces.
        """
        raise AttributeError(about._errors.cstring(
            "ERROR: the definition of power spectra is ill-defined for " +
            "(unstructured) point spaces."))

    def _enforce_power_helper(self, spec, size, kindex):
        # Now it's about to extract a powerspectrum from spec
        # First of all just extract a numpy array. The shape is cared about
        # later.
        dtype = np.dtype('float')
        # Case 1: spec is a function
        if callable(spec):
            # Try to plug in the kindex array in the function directly
            try:
                spec = np.array(spec(kindex), dtype=dtype)
            except:
                # Second try: Use a vectorized version of the function.
                # This is slower, but better than nothing
                try:
                    spec = np.array(np.vectorize(spec)(kindex),
                                    dtype=dtype)
                except:
                    raise TypeError(about._errors.cstring(
                        "ERROR: invalid power spectra function."))

        # Case 2: spec is a field:
        elif isinstance(spec, field):
            try:
                spec = spec.val.get_full_data()
            except AttributeError:
                spec = spec.val
            spec = spec.astype(dtype).flatten()

        # Case 3: spec is a scalar or something else:
        else:
            spec = np.array(spec, dtype=dtype).flatten()

        # Make some sanity checks
        # check finiteness
        if not np.all(np.isfinite(spec)):
            about.warnings.cprint("WARNING: infinite value(s).")

        # check positivity (excluding null)
        if np.any(spec < 0):
            raise ValueError(about._errors.cstring(
                "ERROR: nonpositive value(s)."))
        if np.any(spec == 0):
            about.warnings.cprint("WARNING: nonpositive value(s).")

        # Set the size parameter
        if size is None:
            size = len(kindex)

        # Fix the size of the spectrum
        # If spec is singlevalued, expand it
        if np.size(spec) == 1:
            spec = spec * np.ones(size, dtype=spec.dtype)
        # If the size does not fit at all, throw an exception
        elif np.size(spec) < size:
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( " +
                                                   str(np.size(spec)) + " < " +
                                                   str(size) + " )."))
        elif np.size(spec) > size:
            about.warnings.cprint("WARNING: power spectrum cut to size ( == " +
                                  str(size) + " ).")
            spec = spec[:size]

        return spec

    def check_codomain(self, codomain):
        check_dict = self._cache_dict['check_codomain']
        temp_id = id(codomain)
        if temp_id in check_dict:
            return check_dict[temp_id]
        else:
            temp_result = self._check_codomain(codomain)
            check_dict[temp_id] = temp_result
            return temp_result

    def _check_codomain(self, codomain):
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
        if codomain is None:
            return False

        if not isinstance(codomain, space):
            raise TypeError(about._errors.cstring(
                "ERROR: invalid input. The given input is not a nifty space."))

        if codomain == self:
            return True
        else:
            return False

    def get_codomain(self, **kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, in this case another instance of
            :py:class:`point_space` with the same properties.

            Returns
            -------
            codomain : nifty.point_space
                A compatible codomain.
        """
        return self.copy()

    def get_random_values(self, **kwargs):
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
                - "gau" (normal distribution with zero-mean and a given
                standard
                    deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            spec : {scalar, list, numpy.ndarray, nifty.field, function},
            *optional*
                Power spectrum (default: 1).
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index of each band
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band (default: None).
            codomain : nifty.space, *optional*
                A compatible codomain with power indices (default: None).
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
                vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """

        arg = random.parse_arguments(self, **kwargs)

        if arg is None:
            return self.cast(0)

        if self.datamodel == 'np':
            if arg['random'] == "pm1":
                x = random.pm1(dtype=self.dtype,
                               shape=self.get_shape())
            elif arg['random'] == "gau":
                x = random.gau(dtype=self.dtype,
                               shape=self.get_shape(),
                               mean=arg['mean'],
                               std=arg['std'])
            elif arg['random'] == "uni":
                x = random.uni(dtype=self.dtype,
                               shape=self.get_shape(),
                               vmin=arg['vmin'],
                               vmax=arg['vmax'])
            else:
                raise KeyError(about._errors.cstring(
                    "ERROR: unsupported random key '" +
                    str(arg['random']) + "'."))
            return x

        elif self.datamodel in POINT_DISTRIBUTION_STRATEGIES:
            # Prepare the empty distributed_data_object
            sample = distributed_data_object(
                                        global_shape=self.get_shape(),
                                        dtype=self.dtype,
                                        distribution_strategy=self.datamodel)

            # Case 1: uniform distribution over {-1,+1}/{1,i,-1,-i}
            if arg['random'] == 'pm1':
                sample.apply_generator(lambda s: random.pm1(dtype=self.dtype,
                                                            shape=s))

            # Case 2: normal distribution with zero-mean and a given standard
            #         deviation or variance
            elif arg['random'] == 'gau':
                std = arg['std']
                if np.isscalar(std) or std is None:
                    processed_std = std
                else:
                    try:
                        processed_std = sample.distributor.\
                                                        extract_local_data(std)
                    except(AttributeError):
                        processed_std = std

                sample.apply_generator(lambda s: random.gau(dtype=self.dtype,
                                                            shape=s,
                                                            mean=arg['mean'],
                                                            std=processed_std))

            # Case 3: uniform distribution
            elif arg['random'] == 'uni':
                sample.apply_generator(lambda s: random.uni(dtype=self.dtype,
                                                            shape=s,
                                                            vmin=arg['vmin'],
                                                            vmax=arg['vmax']))
            return sample

        else:
            raise NotImplementedError(about._errors.cstring(
                "ERROR: function is not implemented for given datamodel."))

    def calc_weight(self, x, power=1):
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
        # weight
        return x * self.get_weight(power=power)

    def get_weight(self, power=1, split=False):
        splitted_weight = tuple(np.array(self.distances)**np.array(power))
        if not split:
            return np.prod(splitted_weight)
        else:
            return splitted_weight

    def calc_norm(self, x, q=2):
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
        if q == 2:
            result = self.calc_dot(x, x)
        else:
            y = x**(q - 1)
            result = self.calc_dot(x, y)

        result = result**(1. / q)
        return result

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

        if self.datamodel == 'np':
            result = np.vdot(x, y)
        elif self.datamodel in POINT_DISTRIBUTION_STRATEGIES:
            result = x.vdot(y)
        else:
            raise NotImplementedError(about._errors.cstring(
                "ERROR: function is not implemented for given datamodel."))

        if np.isreal(result):
            result = np.asscalar(np.real(result))
        return result

    def calc_transform(self, x, codomain=None, **kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                codomain space to which the transformation shall map
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
        raise AttributeError(about._errors.cstring(
            "ERROR: fourier-transformation is ill-defined for " +
            "(unstructured) point space."))

    def calc_smooth(self, x, **kwargs):
        """
            Raises an error since smoothing is ill-defined on an unstructured
            space.
        """
        raise AttributeError(about._errors.cstring(
            "ERROR: smoothing ill-defined for (unstructured) point space."))

    def calc_power(self, x, **kwargs):
        """
            Raises an error since the power spectrum is ill-defined for point
            spaces.
        """
        raise AttributeError(about._errors.cstring(
            "ERROR: power spectra ill-defined for (unstructured) " +
            "point space."))

    def calc_real_Q(self, x):
        try:
            return x.isreal().all()
        except AttributeError:
            return np.all(np.isreal(x))

    def calc_bincount(self, x, weights=None, minlength=None):
        try:
            complex_weights_Q = issubclass(weights.dtype.type,
                                           np.complexfloating)
        except AttributeError:
            complex_weights_Q = False

        if self.datamodel == 'np':
            if complex_weights_Q:
                    real_bincount = np.bincount(x, weights=weights.real,
                                                minlength=minlength)
                    imag_bincount = np.bincount(x, weights=weights.imag,
                                                minlength=minlength)
                    return real_bincount + imag_bincount
            else:
                return np.bincount(x, weights=weights, minlength=minlength)
        elif self.datamodel in POINT_DISTRIBUTION_STRATEGIES:
            if complex_weights_Q:
                real_bincount = x.bincount(weights=weights.real,
                                           minlength=minlength)
                imag_bincount = x.bincount(weights=weights.imag,
                                           minlength=minlength)
                return real_bincount + imag_bincount
            else:
                return x.bincount(weights=weights, minlength=minlength)
        else:
            raise NotImplementedError(about._errors.cstring(
                "ERROR: function is not implemented for given datamodel."))

    def get_plot(self, x, title="", vmin=None, vmax=None, unit=None,
                 norm=None, other=None, legend=False, save=None, **kwargs):
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
        if not pl.isinteractive() and save is not None:
            about.warnings.cprint("WARNING: interactive mode off.")

        x = self.cast(x)

        fig = pl.figure(num=None,
                        figsize=(6.4, 4.8),
                        dpi=None,
                        facecolor="none",
                        edgecolor="none",
                        frameon=False,
                        FigureClass=pl.Figure)

        ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])
        xaxes = np.arange(self.para[0], dtype=np.dtype('int'))

        if(norm == "log")and(vmin <= 0):
            raise ValueError(about._errors.cstring(
                "ERROR: nonpositive value(s)."))

        if issubclass(self.dtype.type, np.complexfloating):
            if vmin is None:
                vmin = min(x.real.min(), x.imag.min(), abs(x).min())
            if vmax is None:
                vmax = min(x.real.max(), x.imag.max(), abs(x).max())
        else:
            if vmin is None:
                vmin = x.min()
            if vmax is None:
                vmax = x.max()

        ax0.set_xlim(xaxes[0], xaxes[-1])
        ax0.set_xlabel("index")
        ax0.set_ylim(vmin, vmax)

        if(norm == "log"):
            ax0.set_yscale('log')

        if issubclass(self.dtype.type, np.complexfloating):
            ax0.scatter(xaxes, self.unary_operation(x, op='abs'),
                        color=[0.0, 0.5, 0.0], marker='o',
                        label="graph (absolute)", facecolor="none", zorder=1)
            ax0.scatter(xaxes, self.unary_operation(x, op='real'),
                        color=[0.0, 0.5, 0.0], marker='s',
                        label="graph (real part)", facecolor="none", zorder=1)
            ax0.scatter(xaxes, self.unary_operation(x, op='imag'),
                        color=[0.0, 0.5, 0.0], marker='D',
                        label="graph (imaginary part)", facecolor="none",
                        zorder=1)
        else:
            ax0.scatter(xaxes, x, color=[0.0, 0.5, 0.0], marker='o',
                        label="graph 0", zorder=1)

        if other is not None:
            if not isinstance(other, tuple):
                other = (other, )
            imax = max(1, len(other) - 1)
            for ii in xrange(len(other)):
                ax0.scatter(xaxes, self.dtype(other[ii]),
                            color=[max(0.0, 1.0 - (2 * ii / imax)**2),
                                   0.5 * ((2 * ii - imax) / imax)**2,
                                   max(0.0, 1.0 -
                                       (2 * (ii - imax) / imax)**2)],
                            marker='o', label="'other' graph " + str(ii),
                            zorder=-ii)

        if legend:
            ax0.legend()

        if unit is not None:
            unit = " [" + unit + "]"
        else:
            unit = ""

        ax0.set_ylabel("values" + unit)
        ax0.set_title(title)

        if save is not None:
            fig.savefig(str(save), dpi=None,
                        facecolor="none", edgecolor="none")
            pl.close(fig)
        else:
            fig.canvas.draw()

    def __repr__(self):
        string = ""
        string += str(type(self)) + "\n"
        string += "paradict: " + str(self.paradict) + "\n"
        string += 'dtype: ' + str(self.dtype) + "\n"
        string += 'datamodel: ' + str(self.datamodel) + "\n"
        string += 'comm: ' + self.comm.name + "\n"
        string += 'discrete: ' + str(self.discrete) + "\n"
        string += 'distances: ' + str(self.distances) + "\n"
        return string


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

        codomain : space, *optional*
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

        codomain : space, *optional*
            The space wherein the operator output lives (default: domain).

    """

    def __init__(self, domain=None, val=None, codomain=None, ishape=None,
                 copy=False, **kwargs):
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

        codomain : space, *optional*
            The space wherein the operator output lives (default: domain).

        Returns
        -------
        Nothing

        """
        # If the given val was a field, try to cast it accordingly to the given
        # domain and codomain, etc...
        if isinstance(val, field):
            self._init_from_field(f=val,
                                  domain=domain,
                                  codomain=codomain,
                                  ishape=ishape,
                                  copy=copy,
                                  **kwargs)
        else:
            self._init_from_array(val=val,
                                  domain=domain,
                                  codomain=codomain,
                                  ishape=ishape,
                                  copy=copy,
                                  **kwargs)

    def _init_from_field(self, f, domain, codomain, ishape, copy, **kwargs):
        # check domain
        if domain is None:
            domain = f.domain

        # check codomain
        if codomain is None:
            if domain.check_codomain(f.codomain):
                codomain = f.codomain
            else:
                codomain = domain.get_codomain()

        # check for ishape
        if ishape is None:
            ishape = f.ishape

        # Check if the given field lives in a space which is compatible to the
        # given domain
        if f.domain != domain:
            # Try to transform the given field to the given domain/codomain
            f = f.transform(new_domain=domain,
                            new_codomain=codomain)

        self._init_from_array(domain=domain,
                              val=f.val,
                              codomain=codomain,
                              ishape=ishape,
                              copy=copy,
                              **kwargs)

    def _init_from_array(self, val, domain, codomain, ishape, copy, **kwargs):
        # check domain
        if not isinstance(domain, space):
            raise TypeError(about._errors.cstring(
                "ERROR: Given domain is not a space."))
        self.domain = domain

        # check codomain
        if codomain is None:
            codomain = domain.get_codomain()
        elif not self.domain.check_codomain(codomain):
            raise ValueError(about._errors.cstring(
                "ERROR: The given codomain is not compatible to the domain."))
        self.codomain = codomain

        if ishape is not None:
            ishape = tuple(np.array(ishape, dtype=np.uint).flatten())
        elif val is not None:
            try:
                if val.dtype.type == np.object_:
                    ishape = val.shape
                else:
                    ishape = ()
            except(AttributeError):
                try:
                    ishape = val.ishape
                except(AttributeError):
                    ishape = ()
        else:
            ishape = ()
        self.ishape = ishape

        if val is None:
            if kwargs == {}:
                val = self._map(lambda: self.domain.cast(0.))
            else:
                val = self._map(lambda: self.domain.get_random_values(
                    codomain=self.codomain,
                    **kwargs))
        self.set_val(new_val=val, copy=copy)

    def __len__(self):
        return int(self.get_dim(split=True)[0])

    def apply_scalar_function(self, function, inplace=False):
        if inplace:
            working_field = self
        else:
            working_field = self.copy_empty()

        data_object = self._map(
            lambda z: self.domain.apply_scalar_function(z, function, inplace),
            self.get_val())

        working_field.set_val(data_object)
        return working_field

    def copy(self, domain=None, codomain=None):
        copied_val = self._map(
            lambda z: self.domain.unary_operation(z, op='copy'),
            self.get_val())
        new_field = self.copy_empty(domain=domain, codomain=codomain)
        new_field.set_val(new_val=copied_val)
        return new_field

    def _fast_copy_empty(self):
        # make an empty field
        new_field = EmptyField()
        # repair its class
        new_field.__class__ = self.__class__
        # copy domain, codomain, ishape and val
        for key, value in self.__dict__.items():
            if key != 'val':
                new_field.__dict__[key] = value
            else:
                new_field.__dict__[key] = \
                    self.domain.unary_operation(self.val, op='copy_empty')
        return new_field

    def copy_empty(self, domain=None, codomain=None, ishape=None, **kwargs):
        if domain is None:
            domain = self.domain
        if codomain is None:
            codomain = self.codomain
        if ishape is None:
            ishape = self.ishape

        if (domain is self.domain and
                codomain is self.codomain and
                ishape == self.ishape and
                kwargs == {}):
            new_field = self._fast_copy_empty()
        else:
            new_field = field(domain=domain, codomain=codomain, ishape=ishape,
                              **kwargs)
        return new_field

    def set_val(self, new_val=None, copy=False):
        """
            Resets the field values.

            Parameters
            ----------
            new_val : {scalar, ndarray}
                New field values either as a constant or an arbitrary array.

        """
        if new_val is not None:
            if copy:
                new_val = self._map(
                            lambda z: self.domain.unary_operation(z, 'copy'),
                            new_val)
            self.val = self._map(lambda z: self.domain.cast(z), new_val)
        return self.val

    def get_val(self):
        return self.val

    # TODO: Add functionality for boolean indexing.

    def __getitem__(self, key):
        if np.isscalar(key) == True or isinstance(key, slice):
            key = (key, )
        if self.ishape == ():
            return self.domain.getitem(self.get_val(), key)
        else:
            gotten = self.get_val()[key[:len(self.ishape)]]
            try:
                is_data_container = (gotten.dtype.type == np.object_)
            except(AttributeError):
                is_data_container = False

            if len(key) > len(self.ishape):
                if is_data_container:
                    gotten = self._map(
                        lambda z: self.domain.getitem(
                                                    z, key[len(self.ishape):]),
                        gotten)
                else:
                    gotten = self.domain.getitem(gotten,
                                                 key[len(self.ishape):])
            return gotten

    def __setitem__(self, key, value):
        if np.isscalar(key) or isinstance(key, slice):
            key = (key, )
        if self.ishape == ():
            return self.domain.setitem(self.get_val(), value, key)
        else:
            if len(key) > len(self.ishape):
                gotten = self.get_val()[key[:len(self.ishape)]]
                try:
                    is_data_container = (gotten.dtype.type == np.object_)
                except(AttributeError):
                    is_data_container = False

                if is_data_container:
                    gotten = self._map(
                        lambda z1, z2: self.domain.setitem(
                                               z1, z2, key[len(self.ishape):]),
                        gotten, value)
                else:
                    gotten = self.domain.setitem(gotten, value,
                                                 key[len(self.ishape):])
            else:
                dummy = np.empty(self.ishape)
                gotten = self.val.__setitem__(key, self.cast(
                                           value, ishape=np.shape(dummy[key])))
            return gotten

    def get_shape(self):
        return self.domain.get_shape()

    def get_dim(self, split=False):
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
        return self.domain.get_dim(split=split)

    def get_dof(self, split=False):
        return self.domain.get_dof(split=split)

    def get_ishape(self):
        return self.ishape

    def _map(self, function, *args):
        return utilities.field_map(self.ishape, function, *args)

    def cast(self, x=None, ishape=None):
        if ishape is None:
            ishape = self.ishape
        casted_x = self._cast_to_ishape(x, ishape=ishape)
        if ishape == ():
            return self.domain.cast(casted_x)
        else:
            return self._map(lambda z: self.domain.cast(z),
                             casted_x)

    def _cast_to_ishape(self, x, ishape=None):
        if ishape is None:
            ishape = self.ishape

        if isinstance(x, field):
            x = x.get_val()
        if ishape == ():
            casted_x = self._cast_to_scalar_helper(x)
        else:
            casted_x = self._cast_to_tensor_helper(x, ishape)
        return casted_x

    def _cast_to_scalar_helper(self, x):
        # if x is already a scalar or does fit directly, return it
        self_shape = self.domain.get_shape()
        x_shape = np.shape(x)
        if np.isscalar(x) or x_shape == self_shape:
            return x

        # check if the given object is a 'container'
        try:
            container_Q = (x.dtype.type == np.object_)
        except(AttributeError):
            container_Q = False

        if container_Q:
            # extract the first element. This works on 0-d ndarrays, too.
            result = x[(0,) * len(x_shape)]
            return result

        # if x is no container-type, it could be that the needed shape
        # for self.domain is encapsulated in x
        if x_shape[len(x_shape) - len(self_shape):] == self_shape:
            if x_shape[:len(x_shape) - len(self_shape)] != (1,):
                about.warnings.cprint(
                    "WARNING: discarding all internal dimensions " +
                    "except for the first one.")
            result = x
            for i in xrange(len(x_shape) - len(self_shape)):
                result = result[0]
            return result

        # In all other cases, cast x directly
        return x

    def _cast_to_tensor_helper(self, x, ishape=None):
        if ishape is None:
            ishape = self.ishape

        # Check if x is a container of proper length
        # containing something which will then checked by the domain-space

        x_shape = np.shape(x)
        self_shape = self.domain.get_shape()
        try:
            container_Q = (x.dtype.type == np.object_)
        except(AttributeError):
            container_Q = False

        if container_Q:
            if x_shape == ishape:
                return x
            elif x_shape == ishape[:len(x_shape)]:
                return x.reshape(x_shape +
                                 (1,) * (len(ishape) - len(x_shape)))

        # Slow track: x could be a pure ndarray

        # Case 1 and 2:
        # 1: There are cases where np.shape will only find the container
        # although it was no np.object array; e.g. for [a,1].
        # 2: The overall shape is already the right one
        if x_shape == ishape or x_shape == (ishape + self_shape):
            # Iterate over the outermost dimension and cast the inner spaces
            result = np.empty(ishape, dtype=np.object)
            for i in xrange(np.prod(ishape)):
                ii = np.unravel_index(i, ishape)
                try:
                    result[ii] = x[ii]
                except(TypeError):
                    extracted = x
                    for j in xrange(len(ii)):
                        extracted = extracted[ii[j]]
                    result[ii] = extracted

        # Case 3: The overall shape does not match directly.
        # Check if the input has shape (1, self.domain.shape)
        # Iterate over the outermost dimension and cast the inner spaces
        elif x_shape == ((1,) + self_shape):
            result = np.empty(ishape, dtype=np.object)
            for i in xrange(np.prod(ishape)):
                ii = np.unravel_index(i, ishape)
                result[ii] = x[0]

        # Case 4: fallback: try to cast x with self.domain
        else:  # Iterate over the outermost dimension and cast the inner spaces
            result = np.empty(ishape, dtype=np.object)
            for i in xrange(np.prod(ishape)):
                ii = np.unravel_index(i, ishape)
                result[ii] = x

        return result

    def set_domain(self, new_domain=None, force=False):
        """
            Resets the codomain of the field.

            Parameters
            ----------
            new_codomain : space
                 The new space wherein the transform of the field should live.
                 (default=None).

        """
        # check codomain
        if new_domain is None:
            new_domain = self.codomain.get_codomain()
        elif not force:
            assert(self.codomain.check_codomain(new_domain))
        self.domain = new_domain
        return self.domain

    def set_codomain(self, new_codomain=None, force=False):
        """
            Resets the codomain of the field.

            Parameters
            ----------
            new_codomain : space
                 The new space wherein the transform of the field should live.
                 (default=None).

        """
        # check codomain
        if new_codomain is None:
            new_codomain = self.domain.get_codomain()
        elif not force:
            assert(self.domain.check_codomain(new_codomain))
        self.codomain = new_codomain
        return self.codomain

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
        if overwrite:
            new_field = self
        else:
            new_field = self.copy_empty()

        new_val = self._map(lambda y: self.domain.calc_weight(y, power=power),
                            self.get_val())

        new_field.set_val(new_val=new_val)
        return new_field

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
            return (self.dot(x=self))**(1 / 2)
        else:
            return self.dot(x=self**(q - 1))**(1 / q)

    def dot(self, x=None, axis=None, bare=False):
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
        # Case 1: x equals None
        if x is None:
            return None

        # Case 2: x is a field
        elif isinstance(x, field):
            # if x lives in the cospace, transform it an make a
            # recursive call
            try:
                if self.domain.harmonic != x.domain.harmonic:
                    return self.dot(x=x.transform(), axis=axis)
            except(AttributeError):
                pass

            # whether the domain matches exactly or not:
            # extract the data from x and try to dot with this
            return self.dot(x=x.get_val(), axis=axis, bare=bare)

        # Case 3: x is something else
        else:
            # Cast the input in order to cure dtype and shape differences
            casted_x = self._cast_to_ishape(x)
            # Compute the dot respecting the fact of discrete/continous spaces
            if self.domain.discrete or bare:
                result = self._map(
                    lambda z1, z2: self.domain.calc_dot(z1, z2),
                    self.get_val(),
                    casted_x)
            else:
                result = self._map(
                    lambda z1, z2: self.domain.calc_dot(
                        self.domain.calc_weight(z1, power=1),
                        z2),
                    self.get_val(), casted_x)

            return np.sum(result, axis=axis)

    def outer_dot(self, x=1, axis=None):

        # Use the fact that self.val is a numpy array of dtype np.object
        # -> The shape casting, etc... can be done by numpy
        # If ishape == (), self.val will be multiplied with x directly.
        if self.ishape == ():
            return self * x
        new_val = np.sum(self.get_val() * x, axis=axis)
        # if axis != None, the contraction was not overarching
        if np.dtype(new_val.dtype).type == np.object_:
            new_field = self.copy_empty(ishape=new_val.shape)
        else:
            new_field = self.copy_empty(ishape=())
        new_field.set_val(new_val=new_val)
        return new_field

    def tensor_product(self, x=None):
        if x is None:
            return self
        elif np.isscalar(x) == True:
            return self * x
        else:
            if self.ishape == ():
                temp_val = self.get_val()
                old_val = np.empty((1,), dtype=np.object)
                old_val[0] = temp_val
            else:
                old_val = self.get_val()

            new_val = np.tensordot(old_val, x, axes=0)

            if self.ishape == ():
                new_val = new_val[0]
            new_field = self.copy_empty(ishape=new_val.shape)
            new_field.set_val(new_val=new_val)

            return new_field

    def conjugate(self, inplace=False):
        """
            Computes the complex conjugate of the field.

            Returns
            -------
            cc : field
                The complex conjugated field.

        """
        if inplace:
            work_field = self
        else:
            work_field = self.copy_empty()

        new_val = self._map(
            lambda z: self.domain.unary_operation(z, 'conjugate'),
            self.get_val())
        work_field.set_val(new_val=new_val)

        return work_field

    def transform(self, new_domain=None, new_codomain=None, overwrite=False,
                  **kwargs):
        """
            Computes the transform of the field using the appropriate conjugate
            transformation.

            Parameters
            ----------
            codomain : space, *optional*
                Domain of the transform of the field (default:self.codomain)

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
        if new_domain is None:
            new_domain = self.codomain

        if new_codomain is None:
            # try to recycle the old domain
            if new_domain.check_codomain(self.domain):
                new_codomain = self.domain
            else:
                new_codomain = new_domain.get_codomain()
        else:
            assert(new_domain.check_codomain(new_codomain))

        new_val = self._map(
            lambda z: self.domain.calc_transform(
                z, codomain=new_domain, **kwargs),
            self.get_val())

        if overwrite:
            return_field = self
            return_field.set_codomain(new_codomain=new_codomain, force=True)
            return_field.set_domain(new_domain=new_domain, force=True)
        else:
            return_field = self.copy_empty(domain=new_domain,
                                           codomain=new_codomain)
        return_field.set_val(new_val=new_val, copy=False)

        return return_field

    def smooth(self, sigma=0, inplace=False, **kwargs):
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
        if inplace:
            new_field = self
        else:
            new_field = self.copy_empty()

        new_val = self._map(
            lambda z: self.domain.calc_smooth(z, sigma=sigma, **kwargs),
            self.get_val())

        new_field.set_val(new_val=new_val)
        return new_field

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
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
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

        power_spectrum = self._map(
            lambda z: self.domain.calc_power(z, codomain=self.codomain,
                                             **kwargs),
            self.get_val())

        return power_spectrum

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
                                 bare=False,
                                 ishape=self.ishape)

    def inverse_hat(self):
        """
            Translates the inverted field into a diagonal operator.

            Returns
            -------
            D : operator
                The new diagonal operator instance.

        """
        any_zero_Q = self._map(lambda z: (z == 0).any(), self.get_val())
        any_zero_Q = np.any(any_zero_Q)
        if any_zero_Q:
            raise AttributeError(
                about._errors.cstring("ERROR: singular operator."))
        else:
            from nifty.operators.nifty_operators import diagonal_operator
            return diagonal_operator(domain=self.domain,
                                     diag=(1 / self).get_val(),
                                     bare=False,
                                     ishape=self.ishape)

    def plot(self, **kwargs):
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
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).

            Notes
            -----
            The applicability of the keyword arguments depends on the
            respective space on which the field is defined. Confer to the
            corresponding :py:meth:`get_plot` method.

        """
        # if a save path is given, set pylab to not-interactive
        remember_interactive = pl.isinteractive()
        pl.matplotlib.interactive(not bool(kwargs.get("save", False)))

        if "codomain" in kwargs:
            kwargs.__delitem__("codomain")
            about.warnings.cprint("WARNING: codomain was removed from kwargs.")

        # draw/save the plot(s)
        self.domain.get_plot(self.val, codomain=self.codomain, **kwargs)

        # restore the pylab interactiveness
        pl.matplotlib.interactive(remember_interactive)

    def __repr__(self):
        return "<nifty_core.field>"

    def __str__(self):
        minmax = [self.min(), self.max()]
        mean = self.mean()
        return "nifty_core.field instance\n- domain      = " + \
            repr(self.domain) +\
            "\n- val         = " + repr(self.get_val()) + \
            "\n  - min.,max. = " + str(minmax) + \
            "\n  - mean = " + str(mean) + \
            "\n- codomain      = " + repr(self.codomain) + \
            "\n- ishape          = " + str(self.ishape)

    def _unary_helper(self, x, op, **kwargs):
        result = self._map(
            lambda z: self.domain.unary_operation(z, op=op, **kwargs),
            self.get_val())
        return result

    def min(self, ignore=False, **kwargs):
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
        return self._unary_helper(self.get_val(), op='amin', **kwargs)

    def nanmin(self, **kwargs):
        return self._unary_helper(self.get_val(), op='nanmin', **kwargs)

    def max(self, **kwargs):
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
        return self._unary_helper(self.get_val(), op='amax', **kwargs)

    def nanmax(self, **kwargs):
        return self._unary_helper(self.get_val(), op='nanmax', **kwargs)

    def median(self, **kwargs):
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
        return self._unary_helper(self.get_val(), op='median',
                                  **kwargs)

    def mean(self, **kwargs):
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
        return self._unary_helper(self.get_val(), op='mean',
                                  **kwargs)

    def std(self, **kwargs):
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
        return self._unary_helper(self.get_val(), op='std',
                                  **kwargs)

    def var(self, **kwargs):
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
        return self._unary_helper(self.get_val(), op='var',
                                  **kwargs)

    def argmin(self, split=True, **kwargs):
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
        if split:
            return self._unary_helper(self.get_val(), op='argmin',
                                      **kwargs)
        else:
            return self._unary_helper(self.get_val(), op='argmin_flat',
                                      **kwargs)

    def argmax(self, split=True, **kwargs):
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
        if split:
            return self._unary_helper(self.get_val(), op='argmax',
                                      **kwargs)
        else:
            return self._unary_helper(self.get_val(), op='argmax_flat',
                                      **kwargs)

    # TODO: Implement the full range of unary and binary operotions

    def __pos__(self):
        new_field = self.copy_empty()
        new_val = self._unary_helper(self.get_val(), op='pos')
        new_field.set_val(new_val=new_val)
        return new_field

    def __neg__(self):
        new_field = self.copy_empty()
        new_val = self._unary_helper(self.get_val(), op='neg')
        new_field.set_val(new_val=new_val)
        return new_field

    def __abs__(self):
        new_field = self.copy_empty()
        new_val = self._unary_helper(self.get_val(), op='abs')
        new_field.set_val(new_val=new_val)
        return new_field

    def _binary_helper(self, other, op='None', inplace=False):
        # if other is a field, make sure that the domains match
        if isinstance(other, field):
            other = field(domain=self.domain,
                          val=other,
                          codomain=self.codomain,
                          copy=False)
        try:
            other_val = other.get_val()
        except AttributeError:
            other_val = other

        # bring other_val into the right shape
        if self.ishape == ():
            other_val = self._cast_to_scalar_helper(other_val)
        else:
            other_val = self._cast_to_tensor_helper(other_val)

        new_val = self._map(
            lambda z1, z2: self.domain.binary_operation(z1, z2, op=op, cast=0),
            self.get_val(),
            other_val)

        if inplace:
            working_field = self
        else:
            working_field = self.copy_empty()

        working_field.set_val(new_val=new_val)
        return working_field

    def __add__(self, other):
        return self._binary_helper(other, op='add')
    __radd__ = __add__

    def __iadd__(self, other):
        return self._binary_helper(other, op='iadd', inplace=True)

    def __sub__(self, other):
        return self._binary_helper(other, op='sub')

    def __rsub__(self, other):
        return self._binary_helper(other, op='rsub')

    def __isub__(self, other):
        return self._binary_helper(other, op='isub', inplace=True)

    def __mul__(self, other):
        return self._binary_helper(other, op='mul')
    __rmul__ = __mul__

    def __imul__(self, other):
        return self._binary_helper(other, op='imul', inplace=True)

    def __div__(self, other):
        return self._binary_helper(other, op='div')

    def __rdiv__(self, other):
        return self._binary_helper(other, op='rdiv')

    def __idiv__(self, other):
        return self._binary_helper(other, op='idiv', inplace=True)
    __truediv__ = __div__
    __itruediv__ = __idiv__

    def __pow__(self, other):
        return self._binary_helper(other, op='pow')

    def __rpow__(self, other):
        return self._binary_helper(other, op='rpow')

    def __ipow__(self, other):
        return self._binary_helper(other, op='ipow', inplace=True)

    def __lt__(self, other):
        return self._binary_helper(other, op='lt')

    def __le__(self, other):
        return self._binary_helper(other, op='le')

    def __ne__(self, other):
        if other is None:
            return True
        else:
            return self._binary_helper(other, op='ne')

    def __eq__(self, other):
        if other is None:
            return False
        else:
            return self._binary_helper(other, op='eq')

    def __ge__(self, other):
        return self._binary_helper(other, op='ge')

    def __gt__(self, other):
        return self._binary_helper(other, op='gt')


class EmptyField(field):
    def __init__(self):
        pass