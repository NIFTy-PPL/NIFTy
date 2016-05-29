# NIFTY (Numerical Information Field Theory) has been developed at the
# Max-Planck-Institute for Astrophysics.
##
# Copyright (C) 2015 Max-Planck-Society
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
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  rg
    ..                               /______/

    NIFTY submodule for regular Cartesian grids.

"""
from __future__ import division

import itertools
import numpy as np
import os
from scipy.special import erf
import pylab as pl
from matplotlib.colors import LogNorm as ln
from matplotlib.ticker import LogFormatter as lf

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.nifty_core import point_space
from nifty.nifty_field import field

import nifty_fft
from nifty.config import about,\
                         nifty_configuration as gc,\
                         dependency_injector as gdi
from nifty.nifty_paradict import rg_space_paradict
from nifty.nifty_power_indices import rg_power_indices
from nifty.nifty_random import random
import nifty.nifty_utilities as utilities

MPI = gdi[gc['mpi_module']]
RG_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


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
        dtype : numpy.dtype
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
    epsilon = 0.0001  # relative precision for comparisons

    def __init__(self, shape, zerocenter=False, complexity=0, distances=None,
                 harmonic=False, fft_module=gc['fft_module']):
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
        self._cache_dict = {'check_codomain':{}}
        self.paradict = rg_space_paradict(shape=shape,
                                          complexity=complexity,
                                          zerocenter=zerocenter)
        # set dtype
        if self.paradict['complexity'] == 0:
            self.dtype = np.dtype('float64')
        else:
            self.dtype = np.dtype('complex128')

        # set volume/distances
        naxes = len(self.paradict['shape'])
        if distances is None:
            distances = 1 / np.array(self.paradict['shape'], dtype=np.float)
        elif np.isscalar(distances):
            distances = np.ones(naxes, dtype=np.float) * distances
        else:
            distances = np.array(distances, dtype=np.float)
            if np.size(distances) == 1:
                distances = distances * np.ones(naxes, dtype=np.float)
            if np.size(distances) != naxes:
                raise ValueError(about._errors.cstring(
                    "ERROR: size mismatch ( " + str(np.size(distances)) +
                    " <> " + str(naxes) + " )."))
        if np.any(distances <= 0):
            raise ValueError(about._errors.cstring(
                "ERROR: nonpositive distance(s)."))

        self.distances = tuple(distances)
        self.harmonic = bool(harmonic)
        self.discrete = False

        # Initializes the fast-fourier-transform machine, which will be used
        # to transform the space
        if not gc.validQ('fft_module', fft_module):
            about.warnings.cprint("WARNING: fft_module set to default.")
            fft_module = gc['fft_module']
        self.fft_machine = nifty_fft.fft_factory(fft_module)

        # Initialize the power_indices object which takes care of kindex,
        # pindex, rho and the pundex for a given set of parameters

        # TODO harmonic = True doesn't work yet
        if self.harmonic:
            self.power_indices = rg_power_indices(
                    shape=self.get_shape(),
                    dgrid=distances,
                    zerocentered=self.paradict['zerocenter'],
                    allowed_distribution_strategies=RG_DISTRIBUTION_STRATEGIES)

    @property
    def para(self):
        temp = np.array(self.paradict['shape'] +
                        [self.paradict['complexity']] +
                        self.paradict['zerocenter'], dtype=int)
        return temp

    @para.setter
    def para(self, x):
        self.paradict['shape'] = x[:(np.size(x) - 1) // 2]
        self.paradict['zerocenter'] = x[(np.size(x) + 1) // 2:]
        self.paradict['complexity'] = x[(np.size(x) - 1) // 2]

    def __hash__(self):
        result_hash = 0
        for (key, item) in vars(self).items():
            if key in ['_cache_dict', 'fft_machine', 'power_indices']:
                continue
            result_hash ^= item.__hash__() * hash(key)
        return result_hash

    # __identiftier__ returns an object which contains all information needed
    # to uniquely identify a space. It returns a (immutable) tuple which
    # therefore can be compared.
    # The rg_space version of __identifier__ filters out the vars-information
    # which is describing the rg_space's structure
    def _identifier(self):
        # Extract the identifying parts from the vars(self) dict.
        temp = [(ii[0],
                 ((lambda x: tuple(x) if
                  isinstance(x, np.ndarray) else x)(ii[1])))
                for ii in vars(self).iteritems()
                if ii[0] not in ['_cache_dict', 'fft_machine',
                                 'power_indices']]
        # Return the sorted identifiers as a tuple.
        return tuple(sorted(temp))

    def copy(self):
        return rg_space(shape=self.paradict['shape'],
                        complexity=self.paradict['complexity'],
                        zerocenter=self.paradict['zerocenter'],
                        distances=self.distances,
                        harmonic=self.harmonic,
                        fft_module=self.fft_machine.name)

    def get_shape(self):
        return tuple(self.paradict['shape'])

    def _complement_cast(self, x, axis=None, hermitianize=True):
        if axis is None:
            if x is not None and hermitianize and self.paradict['complexity']\
                    == 1 and not x.hermitian:
                about.warnings.cflush(
                     "WARNING: Data gets hermitianized. This operation is " +
                     "extremely expensive\n")
                x = utilities.hermitianize(x)
        else:
            # TODO hermitianize only on specific axis
            if x is not None and hermitianize and self.paradict['complexity']\
                    == 1 and not x.hermitian:
                about.warnings.cflush(
                     "WARNING: Data gets hermitianized. This operation is " +
                     "extremely expensive\n")
                x = utilities.hermitianize(x)
        return x

    def enforce_power(self, spec, size=None, kindex=None, codomain=None,
                      **kwargs):
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

        # Setting up the local variables: kindex
        # The kindex is only necessary if spec is a function or if
        # the size is not set explicitly
        if kindex is None and (size is None or callable(spec)):
            # Determine which space should be used to get the kindex
            if self.harmonic:
                kindex_supply_space = self
            else:
                # Check if the given codomain is compatible with the space
                try:
                    assert(self.check_codomain(codomain))
                    kindex_supply_space = codomain
                except(AssertionError):
                    about.warnings.cprint("WARNING: Supplied codomain is " +
                                          "incompatible. Generating a " +
                                          "generic codomain. This can " +
                                          "be expensive!")
                    kindex_supply_space = self.get_codomain()

            kindex = kindex_supply_space.\
                power_indices.get_index_dict(**kwargs)['kindex']

        return self._enforce_power_helper(spec=spec,
                                          size=size,
                                          kindex=kindex)

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

        if not isinstance(codomain, rg_space):
            raise TypeError(about._errors.cstring(
                "ERROR: The given codomain must be a nifty rg_space."))

        # check number of number and size of axes
        if not np.all(np.array(self.paradict['shape']) ==
                      np.array(codomain.paradict['shape'])):
            return False

        # check harmonic flag
        if self.harmonic == codomain.harmonic:
            return False

        # check complexity-type
        # prepare the shorthands
        dcomp = self.paradict['complexity']
        cocomp = codomain.paradict['complexity']

        # Case 1: if the domain is copmleteley complex
        # -> the codomain must be complex, too
        if dcomp == 2:
            if cocomp != 2:
                return False
        # Case 2: domain is hermitian
        # -> codmomain can be real. If it is marked as hermitian or even
        # fully complex, a warning is raised
        elif dcomp == 1:
            if cocomp > 0:
                about.warnings.cprint("WARNING: Unrecommended codomain! " +
                                      "The domain is hermitian, hence the " +
                                      "codomain should be restricted to " +
                                      "real values!")

        # Case 3: domain is real
        # -> codmain should be hermitian
        elif dcomp == 0:
            if cocomp == 2:
                about.warnings.cprint("WARNING: Unrecommended codomain! " +
                                      "The domain is real, hence the " +
                                      "codomain should be restricted to " +
                                      "hermitian configurations!")
            elif cocomp == 0:
                return False

        # Check if the distances match, i.e. dist'=1/(num*dist)
        if not np.all(
                np.absolute(np.array(self.paradict['shape']) *
                            np.array(self.distances) *
                            np.array(codomain.distances) - 1) < self.epsilon):
            return False

        return True

    def get_codomain(self, cozerocenter=None, **kwargs):
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
        naxes = len(self.get_shape())
        # Parse the cozerocenter input
        if(cozerocenter is None):
            cozerocenter = self.paradict['zerocenter']
        # if the input is something scalar, cast it to a boolean
        elif(np.isscalar(cozerocenter)):
            cozerocenter = bool(cozerocenter)
        # if it is not a scalar...
        else:
            # ...cast it to a numpy array of booleans
            cozerocenter = np.array(cozerocenter, dtype=np.bool)
            # if it was a list of length 1, extract the boolean
            if(np.size(cozerocenter) == 1):
                cozerocenter = np.asscalar(cozerocenter)
            # if the length of the input does not match the number of
            # dimensions, raise an exception
            elif(np.size(cozerocenter) != naxes):
                raise ValueError(about._errors.cstring(
                    "ERROR: size mismatch ( " +
                    str(np.size(cozerocenter)) + " <> " + str(naxes) + " )."))

        # Set up the initialization variables
        shape = self.paradict['shape']
        distances = 1 / (np.array(self.paradict['shape']) *
                         np.array(self.distances))
        fft_module = self.fft_machine.name
        complexity = {0: 1, 1: 0, 2: 2}[self.paradict['complexity']]
        harmonic = bool(not self.harmonic)

        new_space = rg_space(shape,
                             zerocenter=cozerocenter,
                             complexity=complexity,
                             distances=distances,
                             harmonic=harmonic,
                             fft_module=fft_module)
        return new_space

    def get_random_values(self, **kwargs):
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
            codomain : nifty.rg_space, *optional*
                A compatible codomain (default: None).
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
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        # Parse the keyword arguments
        arg = random.parse_arguments(self, **kwargs)

        if arg is None:
            return self.cast(0)

        # Should the output be hermitianized?
        hermitianizeQ = (self.paradict['complexity'] == 1)

        # Case 1: uniform distribution over {-1,+1}/{1,i,-1,-i}
        if arg['random'] == 'pm1' and not hermitianizeQ:
            sample = super(rg_space, self).get_random_values(**arg)

        elif arg['random'] == 'pm1' and hermitianizeQ:
            sample = self.get_random_values(random='uni', vmin=-1, vmax=1)

            if issubclass(sample.dtype.type, np.complexfloating):
                temp_data = sample.copy()
                sample[temp_data.real >= 0.5] = 1
                sample[(temp_data.real >= 0) * (temp_data.real < 0.5)] = -1
                sample[(temp_data.real < 0) * (temp_data.imag >= 0)] = 1j
                sample[(temp_data.real < 0) * (temp_data.imag < 0)] = -1j
                # Set the mirroring invariant points to real values
                product_list = []
                for s in self.get_shape():
                    # if the particular dimension has even length, set
                    # also the middle of the array to a real value
                    if s % 2 == 0:
                        product_list += [[0, s/2]]
                    else:
                        product_list += [[0]]

                for i in itertools.product(*product_list):
                    sample[i] = {1: 1,
                                 -1: -1,
                                 1j: 1,
                                 -1j: -1}[sample[i]]
            else:
                sample[sample >= 0] = 1
                sample[sample < 0] = -1

            try:
                sample.hermitian = True
            except(AttributeError):
                pass

        # Case 2: normal distribution with zero-mean and a given standard
        #         deviation or variance
        elif arg['random'] == 'gau':
            sample = super(rg_space, self).get_random_values(**arg)

            if hermitianizeQ:
                sample = utilities.hermitianize_gaussian(sample)

        # Case 3: uniform distribution
        elif arg['random'] == "uni" and not hermitianizeQ:
            sample = super(rg_space, self).get_random_values(**arg)

        elif arg['random'] == "uni" and hermitianizeQ:
            # For a hermitian uniform sample, generate a gaussian one
            # and then convert it to a uniform one
            sample = self.get_random_values(random='gau')
            # Use the cummulative of the gaussian, the error function in order
            # to transform it to a uniform distribution.
            if issubclass(sample.dtype.type, np.complexfloating):
                def temp_erf(x):
                    return erf(x.real) + 1j * erf(x.imag)
            else:
                def temp_erf(x):
                    return erf(x / np.sqrt(2))

            sample.apply_scalar_function(function=temp_erf, inplace=True)

            # Shift and stretch the uniform distribution into the given limits
            # sample = (sample + 1)/2 * (vmax-vmin) + vmin
            vmin = arg['vmin']
            vmax = arg['vmax']
            sample *= (vmax - vmin) / 2.
            sample += 1 / 2. * (vmax + vmin)

            try:
                sample.hermitian = True
            except(AttributeError):
                pass

        elif(arg['random'] == "syn"):
            spec = arg['spec']
            kpack = arg['kpack']
            harmonic_domain = arg['harmonic_domain']
            lnb_dict = {}
            for name in ('log', 'nbin', 'binbounds'):
                if arg[name] != 'default':
                    lnb_dict[name] = arg[name]

            # Check whether there is a kpack available or not.
            # kpack is only used for computing kdict and extracting kindex
            # If not, take kdict and kindex from the fourier_domain
            if kpack is None:
                power_indices =\
                    harmonic_domain.power_indices.get_index_dict(**lnb_dict)

                kindex = power_indices['kindex']
                kdict = power_indices['kdict']
                kpack = [power_indices['pindex'], power_indices['kindex']]
            else:
                kindex = kpack[1]
                kdict = harmonic_domain.power_indices.\
                    _compute_kdict_from_pindex_kindex(kpack[0], kpack[1])

            # draw the random samples
            # Case 1: self is a harmonic space
            if self.harmonic:
                # subcase 1: self is real
                # -> simply generate a random field in fourier space and
                # weight the entries accordingly to the powerspectrum
                if self.paradict['complexity'] == 0:
                    sample = self.get_random_values(random='gau',
                                                    mean=0,
                                                    std=1)
                # subcase 2: self is hermitian but probably complex
                # -> generate a real field (in position space) and transform
                # it to harmonic space -> field in harmonic space is
                # hermitian. Now weight the modes accordingly to the
                # powerspectrum.
                elif self.paradict['complexity'] == 1:
                    temp_codomain = self.get_codomain()
                    sample = temp_codomain.get_random_values(random='gau',
                                                             mean=0,
                                                             std=1)

                    # In order to get the normalisation right, the sqrt
                    # of self.dim must be divided out.
                    # Furthermore, the normalisation in the fft routine
                    # must be undone
                    # TODO: Insert explanation
                    sqrt_of_dim = np.sqrt(self.get_dim())
                    sample /= sqrt_of_dim
                    sample = temp_codomain.calc_weight(sample, power=-1)

                    # tronsform the random field to harmonic space
                    sample = temp_codomain.\
                        calc_transform(sample, codomain=self)

                    # ensure that the kdict and the harmonic_sample have the
                    # same distribution strategy
                    try:
                        assert(kdict.distribution_strategy ==
                               sample.distribution_strategy)
                    except AttributeError:
                        pass

                # subcase 3: self is fully complex
                # -> generate a complex random field in harmonic space and
                # weight the modes accordingly to the powerspectrum
                elif self.paradict['complexity'] == 2:
                    sample = self.get_random_values(random='gau',
                                                    mean=0,
                                                    std=1)

                # apply the powerspectrum renormalization
                # extract the local data from kdict
                local_kdict = kdict.get_local_data()
                rescaler = np.sqrt(
                    spec[np.searchsorted(kindex, local_kdict)])
                sample.apply_scalar_function(lambda x: x * rescaler,
                                             inplace=True)

            # Case 2: self is a position space
            else:
                # get a suitable codomain
                temp_codomain = self.get_codomain()

                # subcase 1: self is a real space.
                # -> generate a hermitian sample with the codomain in harmonic
                # space and make a fourier transformation.
                if self.paradict['complexity'] == 0:
                    # check that the codomain is hermitian
                    assert(temp_codomain.paradict['complexity'] == 1)

                # subcase 2: self is hermitian but probably complex
                # -> generate a real-valued random sample in fourier space
                # and transform it to real space
                elif self.paradict['complexity'] == 1:
                    # check that the codomain is real
                    assert(temp_codomain.paradict['complexity'] == 0)

                # subcase 3: self is fully complex
                # -> generate a complex-valued random sample in fourier space
                # and transform it to real space
                elif self.paradict['complexity'] == 2:
                    # check that the codomain is real
                    assert(temp_codomain.paradict['complexity'] == 2)

                # Get a hermitian/real/complex sample in harmonic space from
                # the codomain
                sample = temp_codomain.get_random_values(random='syn',
                                                         pindex=kpack[0],
                                                         kindex=kpack[1],
                                                         spec=spec,
                                                         codomain=self,
                                                         **lnb_dict)

                # Perform a fourier transform
                sample = temp_codomain.calc_transform(sample, codomain=self)

            if self.paradict['complexity'] == 1:
                try:
                    sample.hermitian = True
                except AttributeError:
                    pass

        else:
            raise KeyError(about._errors.cstring(
                "ERROR: unsupported random key '" + str(arg['random']) + "'."))

        return sample

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
        # weight
        x = x * self.get_weight(power=power)
        return x

    def get_weight(self, power=1):
        return np.prod(self.distances)**power

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
            if (np.absolute(result.imag) >
                    self.epsilon**2 * np.absolute(result.real)):
                about.warnings.cprint(
                    "WARNING: Discarding considerable imaginary part.")
            result = np.asscalar(np.real(result))
        return result

    def calc_transform(self, x, codomain=None, **kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.rg_space, *optional*
                codomain space to which the transformation shall map
                (default: None).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array
        """
        x = self.cast(x)

        if codomain is None:
            codomain = self.get_codomain()

        # Check if the given codomain is suitable for the transformation
        if not self.check_codomain(codomain):
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported codomain."))

        if codomain.harmonic:
            # correct for forward fft
            x = self.calc_weight(x, power=1)

        # Perform the transformation
        Tx = self.fft_machine.transform(val=x, domain=self, codomain=codomain,
                                        **kwargs)

        if not codomain.harmonic:
            # correct for inverse fft
            Tx = codomain.calc_weight(Tx, power=-1)

        # when the codomain space is purely real, the result of the
        # transformation must be corrected accordingly. Using the casting
        # method of codomain is sufficient
        # TODO: Let .transform  yield the correct dtype
        Tx = codomain.cast(Tx)

        return Tx

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

        # Check sigma
        if sigma == 0:
            return self.unary_operation(x, op='copy')
        elif sigma == -1:
            about.infos.cprint(
                "INFO: Resetting sigma to sqrt(2)*max(dist).")
            sigma = np.sqrt(2) * np.max(self.distances)
        elif(sigma < 0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))

        # if a codomain was given...
        if codomain is not None:
            # ...check if it was suitable
            if not self.check_codomain(codomain):
                raise ValueError(about._errors.cstring(
                    "ERROR: the given codomain is not a compatible!"))
        else:
            codomain = self.get_codomain()

        x = self.calc_transform(x, codomain=codomain)
        x = codomain._calc_smooth_helper(x, sigma)
        x = codomain.calc_transform(x, codomain=self)
        return x

    def _calc_smooth_helper(self, x, sigma):
        # multiply the gaussian kernel, etc...

        # Cast the input
        x = self.cast(x)

        # if x is hermitian it remains hermitian during smoothing
        # TODO look at this later
        # if self.datamodel in RG_DISTRIBUTION_STRATEGIES:
        remeber_hermitianQ = x.hermitian

        # Define the Gaussian kernel function
        gaussian = lambda x: np.exp(-2. * np.pi**2 * x**2 * sigma**2)

        # Define the variables in the dialect of the legacy smoothing.py
        nx = np.array(self.get_shape())
        dx = 1 / nx / self.distances
        # Multiply the data along each axis with suitable the gaussian kernel
        for i in range(len(nx)):
            # Prepare the exponent
            dk = 1. / nx[i] / dx[i]
            nk = nx[i]
            k = -0.5 * nk * dk + np.arange(nk) * dk
            if self.paradict['zerocenter'][i] == False:
                k = np.fft.fftshift(k)
            # compute the actual kernel vector
            gaussian_kernel_vector = gaussian(k)
            # blow up the vector to an array of shape (1,.,1,len(nk),1,.,1)
            blown_up_shape = [1, ] * len(nx)
            blown_up_shape[i] = len(gaussian_kernel_vector)
            gaussian_kernel_vector =\
                gaussian_kernel_vector.reshape(blown_up_shape)
            # apply the blown-up gaussian_kernel_vector
            x = x*gaussian_kernel_vector

        try:
            x.hermitian = remeber_hermitianQ
        except AttributeError:
            pass

        return x

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

        """
        x = self.cast(x)

        # If self is a position space, delegate calc_power to its codomain.
        if not self.harmonic:
            try:
                codomain = kwargs['codomain']
            except(KeyError):
                codomain = self.get_codomain()

            y = self.calc_transform(x, codomain)
            kwargs.update({'codomain': self})
            return codomain.calc_power(y, **kwargs)

        # If some of the pindex, kindex or rho arrays are given explicitly,
        # favor them over those from the self.power_indices dictionary.
        # As the default value in kwargs.get(key, default) does NOT evaluate
        # lazy, a distinction of cases is necessary. Otherwise the
        # powerindices might be computed, although not needed
        if 'pindex' in kwargs and 'rho' in kwargs:
            pindex = kwargs.get('pindex')
            rho = kwargs.get('rho')
        else:
            power_indices = self.power_indices.get_index_dict(**kwargs)
            pindex = kwargs.get('pindex', power_indices['pindex'])
            rho = kwargs.get('rho', power_indices['rho'])

        fieldabs = abs(x)**2
        power_spectrum = np.zeros(rho.shape)

        power_spectrum = pindex.bincount(weights=fieldabs)

        # Divide out the degeneracy factor
        power_spectrum /= rho
        return power_spectrum

    def get_plot(self,x,title="",vmin=None,vmax=None,power=None,unit="",
                 norm=None,cmap=None,cbar=True,other=None,legend=False,mono=True,**kwargs):
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
            try:
                x = x.get_full_data()
            except AttributeError:
                pass

            fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,
                            facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

            ## explicit kindex
            xaxes = kwargs.get("kindex",None)
            ## implicit kindex
            if(xaxes is None):
                try:
                    self.power_indices
                    kindex_supply_space = self
                except:
                    kindex_supply_space = self.get_codomain()

                xaxes = kindex_supply_space.power_indices.get_index_dict(
                                                **kwargs)['kindex']


#                try:
#                    self.set_power_indices(**kwargs)
#                except:
#                    codomain = kwargs.get("codomain",self.get_codomain())
#                    codomain.set_power_indices(**kwargs)
#                    xaxes = codomain.power_indices.get("kindex")
#                else:
#                    xaxes = self.power_indices.get("kindex")

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
            try:
                x = x.get_full_data()
            except AttributeError:
                pass
            if(naxes==1):
                fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
                ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

                xaxes = (np.arange(self.para[0],dtype=np.int)+self.para[2]*(self.para[0]//2))*self.distances
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
                        other = [self._enforce_values(xx,extend=True) for xx in other]
                    else:
                        other = [self._enforce_values(other,extend=True)]
                    imax = max(1,len(other)-1)
                    for ii in xrange(len(other)):
                        ax0graph(xaxes,other[ii],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if("error" in kwargs):
                        error = self._enforce_values(np.absolute(kwargs.get("error")),extend=True)
                        ax0.fill_between(xaxes,x-error,x+error,color=[0.8,0.8,0.8],label="error 0",zorder=-len(other))
                    if(legend):
                        ax0.legend()
                else:
                    ax0graph(xaxes,x,color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
                    if("error" in kwargs):
                        error = self._enforce_values(np.absolute(kwargs.get("error")),extend=True)
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

                    s_ = np.array([self.para[1]*self.distances[1]/np.max(self.para[:naxes]*self.distances,axis=None,out=None),self.para[0]*self.distances[0]/np.max(self.para[:naxes]*self.distances,axis=None,out=None)*(1.0+0.159*bool(cbar))])
                    fig = pl.figure(num=None,figsize=(6.4*s_[0],6.4*s_[1]),dpi=None,facecolor="none",edgecolor="none",frameon=False,FigureClass=pl.Figure)
                    ax0 = fig.add_axes([0.06/s_[0],0.06/s_[1],1.0-0.12/s_[0],1.0-0.12/s_[1]])

                    xaxes = (np.arange(self.para[1]+1,dtype=np.int)-0.5+self.para[4]*(self.para[1]//2))*self.distances[1]
                    yaxes = (np.arange(self.para[0]+1,dtype=np.int)-0.5+self.para[3]*(self.para[0]//2))*self.distances[0]
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


    def _enforce_values(self, x, extend=True):
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
        about.warnings.cflush(
            "WARNING: _enforce_values is deprecated function. Please use self.cast")
        if(isinstance(x, field)):
            if(self == x.domain):
                if(self.dtype is not x.domain.dtype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '" + str(
                        np.result_type(self.dtype)) + "' <> '" + str(np.result_type(x.domain.dtype)) + "' )."))
                else:
                    x = np.copy(x.val)
            else:
                raise ValueError(about._errors.cstring(
                    "ERROR: inequal domains."))
        else:
            if(np.size(x) == 1):
                if(extend):
                    x = self.dtype(
                        x) * np.ones(self.get_dim(split=True), dtype=self.dtype, order='C')
                else:
                    if(np.isscalar(x)):
                        x = np.array([x], dtype=self.dtype)
                    else:
                        x = np.array(x, dtype=self.dtype)
            else:
                x = self.enforce_shape(np.array(x, dtype=self.dtype))

        # hermitianize if ...
        if(about.hermitianize.status)and(np.size(x) != 1)and(self.para[(np.size(self.para) - 1) // 2] == 1):
            #x = gp.nhermitianize_fast(x,self.para[-((np.size(self.para)-1)//2):].astype(np.bool),special=False)
            x = utilities.hermitianize(x)
        # check finiteness
        if(not np.all(np.isfinite(x))):
            about.warnings.cprint("WARNING: infinite value(s).")

        return x



    def get_plot_new(self, x, title="", vmin=None, vmax=None, power=None, unit="",
                 norm=None, cmap=None, cbar=True, other=None, legend=False,
                 mono=True, save=None, **kwargs):
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
        try:
            x = x.get_full_data()
        except AttributeError:
            pass

        if(not pl.isinteractive())and(not bool(kwargs.get("save", False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        naxes = (np.size(self.para) - 1) // 2
        if(power is None):
            power = bool(self.para[naxes])

        if(power):
            x = self.calc_power(x, **kwargs)

            fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None, facecolor="none",
                            edgecolor="none", frameon=False, FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])

            # explicit kindex
            xaxes = kwargs.get("kindex", None)
            # implicit kindex
            if(xaxes is None):
                try:
                    self.set_power_indices(**kwargs)
                except:
                    codomain = kwargs.get("codomain", self.get_codomain())
                    codomain.set_power_indices(**kwargs)
                    xaxes = codomain.power_indices.get("kindex")
                else:
                    xaxes = self.power_indices.get("kindex")

            if(norm is None)or(not isinstance(norm, int)):
                norm = naxes
            if(vmin is None):
                vmin = np.min(x[:mono].tolist() + (xaxes**norm * x)
                              [1:].tolist(), axis=None, out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist() + (xaxes**norm * x)
                              [1:].tolist(), axis=None, out=None)
            ax0.loglog(xaxes[1:], (xaxes**norm * x)[1:], color=[0.0, 0.5, 0.0],
                       label="graph 0", linestyle='-', linewidth=2.0, zorder=1)
            if(mono):
                ax0.scatter(0.5 * (xaxes[1] + xaxes[2]), x[0], s=20, color=[0.0, 0.5, 0.0], marker='o',
                            cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, zorder=1)

            if(other is not None):
                if(isinstance(other, tuple)):
                    other = list(other)
                    for ii in xrange(len(other)):
                        if(isinstance(other[ii], field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(
                                other[ii], size=np.size(xaxes), kindex=xaxes)
                elif(isinstance(other, field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(
                        other, size=np.size(xaxes), kindex=xaxes)]
                imax = max(1, len(other) - 1)
                for ii in xrange(len(other)):
                    ax0.loglog(xaxes[1:], (xaxes**norm * other[ii])[1:], color=[max(0.0, 1.0 - (2 * ii / imax)**2), 0.5 * ((2 * ii - imax) / imax)**2, max(
                        0.0, 1.0 - (2 * (ii - imax) / imax)**2)], label="graph " + str(ii + 1), linestyle='-', linewidth=1.0, zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5 * (xaxes[1] + xaxes[2]), other[ii][0], s=20, color=[max(0.0, 1.0 - (2 * ii / imax)**2), 0.5 * ((2 * ii - imax) / imax)**2, max(
                            0.0, 1.0 - (2 * (ii - imax) / imax)**2)], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1], xaxes[-1])
            ax0.set_xlabel(r"$|k|$")
            ax0.set_ylim(vmin, vmax)
            ax0.set_ylabel(r"$|k|^{%i} P_k$" % norm)
            ax0.set_title(title)

        else:
            x = self.cast(x)

            if(naxes == 1):
                fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None, facecolor="none",
                                edgecolor="none", frameon=False, FigureClass=pl.Figure)
                ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])

                xaxes = (np.arange(
                    self.para[0], dtype=np.int) + self.para[2] * (self.para[0] // 2)) * self.distances
                if(vmin is None):
                    if(np.iscomplexobj(x)):
                        vmin = min(np.min(np.absolute(x), axis=None, out=None), np.min(
                            np.real(x), axis=None, out=None), np.min(np.imag(x), axis=None, out=None))
                    else:
                        vmin = np.min(x, axis=None, out=None)
                if(vmax is None):
                    if(np.iscomplexobj(x)):
                        vmax = max(np.max(np.absolute(x), axis=None, out=None), np.max(
                            np.real(x), axis=None, out=None), np.max(np.imag(x), axis=None, out=None))
                    else:
                        vmax = np.max(x, axis=None, out=None)
                if(norm == "log"):
                    ax0graph = ax0.semilogy
                    if(vmin <= 0):
                        raise ValueError(about._errors.cstring(
                            "ERROR: nonpositive value(s)."))
                else:
                    ax0graph = ax0.plot

                if(np.iscomplexobj(x)):
                    ax0graph(xaxes, np.absolute(x), color=[
                             0.0, 0.5, 0.0], label="graph (absolute)", linestyle='-', linewidth=2.0, zorder=1)
                    ax0graph(xaxes, np.real(x), color=[
                             0.0, 0.5, 0.0], label="graph (real part)", linestyle="--", linewidth=1.0, zorder=0)
                    ax0graph(xaxes, np.imag(x), color=[
                             0.0, 0.5, 0.0], label="graph (imaginary part)", linestyle=':', linewidth=1.0, zorder=0)
                    if(legend):
                        ax0.legend()
                elif(other is not None):
                    ax0graph(xaxes, x, color=[
                             0.0, 0.5, 0.0], label="graph 0", linestyle='-', linewidth=2.0, zorder=1)
                    if(isinstance(other, tuple)):
                        other = [self._enforce_values(
                            xx, extend=True) for xx in other]
                    else:
                        other = [self._enforce_values(other, extend=True)]
                    imax = max(1, len(other) - 1)
                    for ii in xrange(len(other)):
                        ax0graph(xaxes, other[ii], color=[max(0.0, 1.0 - (2 * ii / imax)**2), 0.5 * ((2 * ii - imax) / imax)**2, max(
                            0.0, 1.0 - (2 * (ii - imax) / imax)**2)], label="graph " + str(ii + 1), linestyle='-', linewidth=1.0, zorder=-ii)
                    if("error" in kwargs):
                        error = self._enforce_values(
                            np.absolute(kwargs.get("error")), extend=True)
                        ax0.fill_between(
                            xaxes, x - error, x + error, color=[0.8, 0.8, 0.8], label="error 0", zorder=-len(other))
                    if(legend):
                        ax0.legend()
                else:
                    ax0graph(xaxes, x, color=[
                             0.0, 0.5, 0.0], label="graph 0", linestyle='-', linewidth=2.0, zorder=1)
                    if("error" in kwargs):
                        error = self._enforce_values(
                            np.absolute(kwargs.get("error")), extend=True)
                        ax0.fill_between(
                            xaxes, x - error, x + error, color=[0.8, 0.8, 0.8], label="error 0", zorder=0)

                ax0.set_xlim(xaxes[0], xaxes[-1])
                ax0.set_xlabel("coordinate")
                ax0.set_ylim(vmin, vmax)
                if(unit):
                    unit = " [" + unit + "]"
                ax0.set_ylabel("values" + unit)
                ax0.set_title(title)

            elif(naxes == 2):
                if issubclass(x.dtype.type, np.complexfloating):
                    about.infos.cprint(
                        "INFO: absolute values and phases are plotted.")
                    if title:
                        title = str(title) + " "
                    if save is not None:
                        temp_save = os.path.splitext(os.path.basename(
                                                                  str("save")))
                        save = temp_save[0] + "_absolute" + temp_save[1]
                    self.get_plot(self.unary_operation(x, op='abs'),
                                  title=title + "(absolute)", vmin=vmin,
                                  vmax=vmax, power=False, unit=unit, norm=norm,
                                  cmap=cmap, cbar=cbar, other=None,
                                  legend=legend, **kwargs)
                    if unit:
                        unit = "rad"
                    if cmap is None:
                        cmap = pl.cm.hsv_r
                    if save is not None:
                        save = temp_save[0] + "_phase" + temp_save[1]
                    self.get_plot(np.angle(x, deg=False), title=title + "(phase)", vmin=-3.1416, vmax=3.1416, power=False,
                                  unit=unit, norm=None, cmap=cmap, cbar=cbar, other=None, legend=False, **kwargs)  # values in [-pi,pi]
                    return None  # leave method
                else:
                    if(vmin is None):
                        vmin = np.min(x, axis=None, out=None)
                    if(vmax is None):
                        vmax = np.max(x, axis=None, out=None)
                    if(norm == "log")and(vmin <= 0):
                        raise ValueError(about._errors.cstring(
                            "ERROR: nonpositive value(s)."))

                    s_ = np.array([self.para[1] * self.distances[1] / np.max(self.para[:naxes] * self.distances, axis=None, out=None), self.para[
                                  0] * self.distances[0] / np.max(self.para[:naxes] * self.distances, axis=None, out=None) * (1.0 + 0.159 * bool(cbar))])
                    fig = pl.figure(num=None, figsize=(
                        6.4 * s_[0], 6.4 * s_[1]), dpi=None, facecolor="none", edgecolor="none", frameon=False, FigureClass=pl.Figure)
                    ax0 = fig.add_axes(
                        [0.06 / s_[0], 0.06 / s_[1], 1.0 - 0.12 / s_[0], 1.0 - 0.12 / s_[1]])

                    xaxes = (np.arange(self.para[
                             1] + 1, dtype=np.int) - 0.5 + self.para[4] * (self.para[1] // 2)) * self.distances[1]
                    yaxes = (np.arange(self.para[
                             0] + 1, dtype=np.int) - 0.5 + self.para[3] * (self.para[0] // 2)) * self.distances[0]
                    if(norm == "log"):
                        n_ = ln(vmin=vmin, vmax=vmax)
                    else:
                        n_ = None
                    sub = ax0.pcolormesh(
                        xaxes, yaxes, x, cmap=cmap, norm=n_, vmin=vmin, vmax=vmax)
                    ax0.set_xlim(xaxes[0], xaxes[-1])
                    ax0.set_xticks([0], minor=False)
                    ax0.set_ylim(yaxes[0], yaxes[-1])
                    ax0.set_yticks([0], minor=False)
                    ax0.set_aspect("equal")
                    if(cbar):
                        if(norm == "log"):
                            f_ = lf(10, labelOnlyBase=False)
                            b_ = sub.norm.inverse(
                                np.linspace(0, 1, sub.cmap.N + 1))
                            v_ = np.linspace(
                                sub.norm.vmin, sub.norm.vmax, sub.cmap.N)
                        else:
                            f_ = None
                            b_ = None
                            v_ = None
                        cb0 = fig.colorbar(sub, ax=ax0, orientation="horizontal", fraction=0.1, pad=0.05, shrink=0.75, aspect=20, ticks=[
                                           vmin, vmax], format=f_, drawedges=False, boundaries=b_, values=v_)
                        cb0.ax.text(0.5, -1.0, unit, fontdict=None, withdash=False, transform=cb0.ax.transAxes,
                                    horizontalalignment="center", verticalalignment="center")
                    ax0.set_title(title)

            else:
                raise ValueError(about._errors.cstring(
                    "ERROR: unsupported number of axes ( " + str(naxes) + " > 2 )."))

        if(bool(kwargs.get("save", False))):
            fig.savefig(str(kwargs.get("save")), dpi=None, facecolor="none", edgecolor="none", orientation="portrait",
                        papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()

    def __repr__(self):
        string = super(rg_space, self).__repr__()
        string += repr(self.fft_machine) + "\n "
        return string