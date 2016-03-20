# NIFTY (Numerical Information Field Theory) has been developed at the
# Max-Planck-Institute for Astrophysics.
#
# Copyright (C) 2015 Max-Planck-Society
#
# Author: Marco Selig
# Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

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

import os
import numpy as np
import pylab as pl
from matplotlib.colors import LogNorm as ln
from matplotlib.ticker import LogFormatter as lf

from nifty.nifty_core import space,\
                             point_space,\
                             field
from nifty.keepers import about,\
                    global_configuration as gc,\
                    global_dependency_injector as gdi
from nifty.nifty_paradict import lm_space_paradict,\
                                 gl_space_paradict,\
                                 hp_space_paradict
from nifty.nifty_power_indices import lm_power_indices

from nifty.nifty_mpi_data import distributed_data_object
from nifty.nifty_mpi_data import STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.nifty_random import random

gl = gdi.get('libsharp_wrapper_gl')
hp = gdi.get('healpy')

LM_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']
GL_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']
HP_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


class lm_space(point_space):
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
        dtype : numpy.dtype, *optional*
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
        dtype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Parameter captioning the fact that an :py:class:`lm_space` is
            always discrete.
        vol : numpy.ndarray
            Pixel volume of the :py:class:`lm_space`, which is always 1.
    """

    def __init__(self, lmax, mmax=None, dtype=np.dtype('complex128'),
                 datamodel='not', comm=gc['default_comm']):
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
            dtype : numpy.dtype, *optional*
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

        # check imports
        if not gc['use_libsharp'] and not gc['use_healpy']:
            raise ImportError(about._errors.cstring(
                "ERROR: neither libsharp_wrapper_gl nor healpy activated."))

        self._cache_dict = {'check_codomain': {}}

        self.paradict = lm_space_paradict(lmax=lmax, mmax=mmax)

        # check data type
        dtype = np.dtype(dtype)
        if dtype not in [np.dtype('complex64'), np.dtype('complex128')]:
            about.warnings.cprint("WARNING: data type set to complex128.")
            dtype = np.dtype('complex128')
        self.dtype = dtype

        # set datamodel
        if datamodel not in ['not']:
            about.warnings.cprint(
                "WARNING: %s is not a recommended datamodel for lm_space."
                % datamodel)
        if datamodel not in LM_DISTRIBUTION_STRATEGIES:
            raise ValueError(about._errors.cstring(
                "ERROR: %s is not a valid datamodel" % datamodel))

        self.datamodel = datamodel

        self.discrete = True
        self.harmonic = True
        self.distances = (np.float(1),)
        self.comm = self._parse_comm(comm)

        self.power_indices = lm_power_indices(
                    lmax=self.paradict['lmax'],
                    dim=self.get_dim(),
                    comm=self.comm,
                    datamodel=self.datamodel,
                    allowed_distribution_strategies=LM_DISTRIBUTION_STRATEGIES)

    @property
    def para(self):
        temp = np.array([self.paradict['lmax'],
                         self.paradict['mmax']], dtype=int)
        return temp

    @para.setter
    def para(self, x):
        self.paradict['lmax'] = x[0]
        self.paradict['mmax'] = x[1]

    def __hash__(self):
        result_hash = 0
        for (key, item) in vars(self).items():
            if key in ['_cache_dict', 'power_indices']:
                continue
            result_hash ^= item.__hash__() * hash(key)
        return result_hash

    def _identifier(self):
        # Extract the identifying parts from the vars(self) dict.
        temp = [(ii[0],
                 ((lambda x: tuple(x) if
                  isinstance(x, np.ndarray) else x)(ii[1])))
                for ii in vars(self).iteritems()
                if ii[0] not in ['_cache_dict', 'power_indices', 'comm']]
        temp.append(('comm', self.comm.__hash__()))
        # Return the sorted identifiers as a tuple.
        return tuple(sorted(temp))

    def copy(self):
        return lm_space(lmax=self.paradict['lmax'],
                        mmax=self.paradict['mmax'],
                        dtype=self.dtype)

    def get_shape(self):
        lmax = self.paradict['lmax']
        mmax = self.paradict['mmax']
        return (np.int((mmax + 1) * (lmax + 1) - ((mmax + 1) * mmax) // 2),)

    def get_dof(self, split=False):
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
        # dof = 2*dim-(lmax+1) = (lmax+1)*(2*mmax+1)*(mmax+1)*mmax
        lmax = self.paradict['lmax']
        mmax = self.paradict['mmax']
        dof = np.int((lmax + 1) * (2 * mmax + 1) - (mmax + 1) * mmax)
        if split:
            return (dof, )
        else:
            return dof

    def get_meta_volume(self, split=False):
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
        if not split:
            return np.float(self.get_dof())
        else:
            mol = self.cast(1, dtype=np.float)
            mol[self.paradict['lmax'] + 1:] = 2  # redundant: (l,m) and (l,-m)
            return mol

    def _cast_to_d2o(self, x, dtype=None, **kwargs):
        casted_x = super(lm_space, self)._cast_to_d2o(x=x,
                                                      dtype=dtype,
                                                      **kwargs)
        lmax = self.paradict['lmax']
        complexity_mask = casted_x[:lmax+1].iscomplex()
        if complexity_mask.any():
            about.warnings.cprint("WARNING: Taking the absolute values for " +
                                  "all complex entries where lmax==0")
            casted_x[:lmax+1] = abs(casted_x[:lmax+1])
        return casted_x

    # TODO: Extend to binning/log
    def enforce_power(self, spec, size=None, kindex=None):
        if kindex is None:
            kindex_size = self.paradict['lmax'] + 1
            kindex = np.arange(kindex_size,
                               dtype=np.array(self.distances).dtype)
        return self._enforce_power_helper(spec=spec,
                                          size=size,
                                          kindex=kindex)

    def _check_codomain(self, codomain):
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
        if codomain is None:
            return False

        if not isinstance(codomain, space):
            raise TypeError(about._errors.cstring(
                "ERROR: The given codomain must be a nifty lm_space."))

        if self.comm is not codomain.comm:
            return False

        if self.datamodel is not codomain.datamodel:
            return False

        elif isinstance(codomain, gl_space):
            # lmax==mmax
            # nlat==lmax+1
            # nlon==2*lmax+1
            if ((self.paradict['lmax'] == self.paradict['mmax']) and
                    (codomain.paradict['nlat'] == self.paradict['lmax']+1) and
                    (codomain.paradict['nlon'] == 2*self.paradict['lmax']+1)):
                return True

        elif isinstance(codomain, hp_space):
            # lmax==mmax
            # 3*nside-1==lmax
            if ((self.paradict['lmax'] == self.paradict['mmax']) and
                    (3*codomain.paradict['nside']-1 == self.paradict['lmax'])):
                return True

        return False

    def get_codomain(self, coname=None, **kwargs):
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
            .. [#] M. Reinecke and D. Sverre Seljebotn, 2013,
                   "Libsharp - spherical
                   harmonic transforms revisited";
                   `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

        """
        if coname == 'gl' or (coname is None and gc['lm2gl']):
            if self.dtype == np.dtype('complex64'):
                new_dtype = np.float32
            elif self.dtype == np.dtype('complex128'):
                new_dtype = np.float64
            else:
                raise NotImplementedError
            nlat = self.paradict['lmax'] + 1
            nlon = self.paradict['lmax'] * 2 + 1
            return gl_space(nlat=nlat, nlon=nlon, dtype=new_dtype,
                            datamodel=self.datamodel,
                            comm=self.comm)

        elif coname == 'hp' or (coname is None and not gc['lm2gl']):
            nside = (self.paradict['lmax']+1) // 3
            return hp_space(nside=nside,
                            datamodel=self.datamodel,
                            comm=self.comm)

        else:
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported or incompatible codomain '"+coname+"'."))

    def get_random_values(self, **kwargs):
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
            spec : {scalar, list, numpy.array, nifty.field, function},
                *optional*
                Power spectrum (default: 1).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.parse_arguments(self, **kwargs)

        if arg is None:
            x = 0

        elif arg['random'] == "pm1":
            x = random.pm1(dtype=self.dtype, shape=self.get_shape())

        elif arg['random'] == "gau":
            x = random.gau(dtype=self.dtype,
                           shape=self.get_shape(),
                           mean=arg['mean'],
                           std=arg['std'])

        elif arg['random'] == "syn":
            lmax = self.paradict['lmax']
            mmax = self.paradict['mmax']
            if self.dtype == np.dtype('complex64'):
                if gc['use_libsharp']:
                    x = gl.synalm_f(arg['spec'], lmax=lmax, mmax=mmax)
                else:
                    x = hp.synalm(arg['spec'].astype(np.complex128),
                                  lmax=lmax, mmax=mmax).astype(np.complex64)
            else:
                if gc['use_healpy']:
                    x = hp.synalm(arg['spec'], lmax=lmax, mmax=mmax)
                else:
                    x = gl.synalm(arg['spec'], lmax=lmax, mmax=mmax)

        elif arg['random'] == "uni":
            x = random.uni(dtype=self.dtype,
                           shape=self.get_shape(),
                           vmin=arg['vmin'],
                           vmax=arg['vmax'])

        else:
            raise KeyError(about._errors.cstring(
                "ERROR: unsupported random key '" + str(arg['random']) + "'."))

        return self.cast(x)

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

        lmax = self.paradict['lmax']

        x_low = x[:lmax + 1]
        x_high = x[lmax + 1:]
        y_low = y[:lmax + 1]
        y_high = y[lmax + 1:]

        dot = (x_low.real * y_low.real).sum()
        dot += 2 * (x_high.real * y_high.real).sum()
        dot += 2 * (x_high.imag * y_high.imag).sum()
        return dot

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
        """
        x = self.cast(x)

        if codomain is None:
            codomain = self.get_codomain()

        # Check if the given codomain is suitable for the transformation
        if not self.check_codomain(codomain):
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported codomain."))

        if self.datamodel != 'not':
            about.warnings.cprint(
                "WARNING: Field data is consolidated to all nodes for "
                "external alm2map method!")

        np_x = x.get_full_data()

        if isinstance(codomain, gl_space):
            nlat = codomain.paradict['nlat']
            nlon = codomain.paradict['nlon']
            lmax = self.paradict['lmax']
            mmax = self.paradict['mmax']

            # transform
            if self.dtype == np.dtype('complex64'):
                np_Tx = gl.alm2map_f(np_x, nlat=nlat, nlon=nlon,
                                     lmax=lmax, mmax=mmax, cl=False)
            else:
                np_Tx = gl.alm2map(np_x, nlat=nlat, nlon=nlon,
                                   lmax=lmax, mmax=mmax, cl=False)
            Tx = codomain.cast(np_Tx)

        elif isinstance(codomain, hp_space):
            nside = codomain.paradict['nside']
            lmax = self.paradict['lmax']
            mmax = self.paradict['mmax']

            # transform
            np_x = np_x.astype(np.complex128, copy=False)
            np_Tx = hp.alm2map(np_x, nside, lmax=lmax,
                               mmax=mmax, pixwin=False, fwhm=0.0, sigma=None,
                               pol=True, inplace=False)
            Tx = codomain.cast(np_Tx)

        else:
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported transformation."))

        # re-weight if discrete
        if codomain.discrete:
            Tx = codomain.calc_weight(Tx, power=0.5)

        return codomain.cast(Tx)

    def calc_smooth(self, x, sigma=0, **kwargs):
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
        x = self.cast(x)
        # check sigma
        if sigma == 0:
            return self.unary_operation(x, op='copy')
        elif sigma == -1:
            about.infos.cprint("INFO: invalid sigma reset.")
            sigma = np.sqrt(2) * np.pi / (self.paradict['lmax'] + 1)
        elif sigma < 0:
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))

        if gc['use_healpy']:
            return self.cast(hp.smoothalm(x, fwhm=0.0, sigma=sigma,
                                pol=True, mmax=self.paradict['mmax'],
                                verbose=False, inplace=False))
        else:
            return self.cast(gl.smoothalm(x, lmax=self.paradict['lmax'],
                                mmax=self.paradict['mmax'],
                                fwhm=0.0, sigma=sigma, overwrite=False))

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
        """
        x = self.cast(x)
        lmax = self.paradict['lmax']
        mmax = self.paradict['mmax']

        # power spectrum
        if self.dtype == np.dtype('complex64'):
            if gc['use_libsharp']:
                return gl.anaalm_f(np.array(x), lmax=lmax, mmax=mmax)
            else:
                return hp.alm2cl(np.array(x).astype(np.complex128), alms2=None,
                                 lmax=lmax, mmax=mmax, lmax_out=lmax,
                                 nspec=None).astype(np.float32)
        else:
            if gc['use_healpy']:
                return hp.alm2cl(np.array(x), alms2=None, lmax=lmax, mmax=mmax,
                                 lmax_out=lmax, nspec=None)
            else:
                return gl.anaalm(np.array(x), lmax=lmax, mmax=mmax)

    def get_plot(self, x, title="", vmin=None, vmax=None, power=True,
                 norm=None, cmap=None, cbar=True, other=None, legend=False,
                 mono=True, **kwargs):
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
        if(not pl.isinteractive())and(not bool(kwargs.get("save", False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(power):
            x = self.calc_power(x)

            fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None, facecolor="none",
                            edgecolor="none", frameon=False, FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])

            xaxes = np.arange(self.para[0] + 1, dtype=np.int)
            if(vmin is None):
                vmin = np.min(x[:mono].tolist(
                ) + (xaxes * (2 * xaxes + 1) * x)[1:].tolist(), axis=None, out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist(
                ) + (xaxes * (2 * xaxes + 1) * x)[1:].tolist(), axis=None, out=None)
            ax0.loglog(xaxes[1:], (xaxes * (2 * xaxes + 1) * x)[1:], color=[0.0,
                                                                            0.5, 0.0], label="graph 0", linestyle='-', linewidth=2.0, zorder=1)
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
                            other[ii] = self.enforce_power(other[ii])
                elif(isinstance(other, field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other)]
                imax = max(1, len(other) - 1)
                for ii in xrange(len(other)):
                    ax0.loglog(xaxes[1:], (xaxes * (2 * xaxes + 1) * other[ii])[1:], color=[max(0.0, 1.0 - (2 * ii / imax)**2), 0.5 * ((2 * ii - imax) / imax)
                                                                                            ** 2, max(0.0, 1.0 - (2 * (ii - imax) / imax)**2)], label="graph " + str(ii + 1), linestyle='-', linewidth=1.0, zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5 * (xaxes[1] + xaxes[2]), other[ii][0], s=20, color=[max(0.0, 1.0 - (2 * ii / imax)**2), 0.5 * ((2 * ii - imax) / imax)**2, max(
                            0.0, 1.0 - (2 * (ii - imax) / imax)**2)], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1], xaxes[-1])
            ax0.set_xlabel(r"$\ell$")
            ax0.set_ylim(vmin, vmax)
            ax0.set_ylabel(r"$\ell(2\ell+1) C_\ell$")
            ax0.set_title(title)

        else:
            x = self.cast(x)
            if(np.iscomplexobj(x)):
                if(title):
                    title += " "
                if(bool(kwargs.get("save", False))):
                    save_ = os.path.splitext(
                        os.path.basename(str(kwargs.get("save"))))
                    kwargs.update(save=save_[0] + "_absolute" + save_[1])
                self.get_plot(np.absolute(x), title=title + "(absolute)", vmin=vmin, vmax=vmax,
                              power=False, norm=norm, cmap=cmap, cbar=cbar, other=None, legend=False, **kwargs)
#                self.get_plot(np.real(x),title=title+"(real part)",vmin=vmin,vmax=vmax,power=False,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
#                self.get_plot(np.imag(x),title=title+"(imaginary part)",vmin=vmin,vmax=vmax,power=False,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
                if(cmap is None):
                    cmap = pl.cm.hsv_r
                if(bool(kwargs.get("save", False))):
                    kwargs.update(save=save_[0] + "_phase" + save_[1])
                self.get_plot(np.angle(x, deg=False), title=title + "(phase)", vmin=-3.1416, vmax=3.1416, power=False,
                              norm=None, cmap=cmap, cbar=cbar, other=None, legend=False, **kwargs)  # values in [-pi,pi]
                return None  # leave method
            else:
                if(vmin is None):
                    vmin = np.min(x, axis=None, out=None)
                if(vmax is None):
                    vmax = np.max(x, axis=None, out=None)
                if(norm == "log")and(vmin <= 0):
                    raise ValueError(about._errors.cstring(
                        "ERROR: nonpositive value(s)."))

                # not a number
                xmesh = np.nan * \
                    np.empty(self.para[::-1] + 1, dtype=np.float16, order='C')
                xmesh[4, 1] = None
                xmesh[1, 4] = None
                lm = 0
                for mm in xrange(self.para[1] + 1):
                    xmesh[mm][mm:] = x[lm:lm + self.para[0] + 1 - mm]
                    lm += self.para[0] + 1 - mm

                s_ = np.array([1, self.para[1] / self.para[0]
                               * (1.0 + 0.159 * bool(cbar))])
                fig = pl.figure(num=None, figsize=(
                    6.4 * s_[0], 6.4 * s_[1]), dpi=None, facecolor="none", edgecolor="none", frameon=False, FigureClass=pl.Figure)
                ax0 = fig.add_axes(
                    [0.06 / s_[0], 0.06 / s_[1], 1.0 - 0.12 / s_[0], 1.0 - 0.12 / s_[1]])
                ax0.set_axis_bgcolor([0.0, 0.0, 0.0, 0.0])

                xaxes = np.arange(self.para[0] + 2, dtype=np.int) - 0.5
                yaxes = np.arange(self.para[1] + 2, dtype=np.int) - 0.5
                if(norm == "log"):
                    n_ = ln(vmin=vmin, vmax=vmax)
                else:
                    n_ = None
                sub = ax0.pcolormesh(xaxes, yaxes, np.ma.masked_where(np.isnan(
                    xmesh), xmesh), cmap=cmap, norm=n_, vmin=vmin, vmax=vmax, clim=(vmin, vmax))
                ax0.set_xlim(xaxes[0], xaxes[-1])
                ax0.set_xticks([0], minor=False)
                ax0.set_xlabel(r"$\ell$")
                ax0.set_ylim(yaxes[0], yaxes[-1])
                ax0.set_yticks([0], minor=False)
                ax0.set_ylabel(r"$m$")
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
                    fig.colorbar(sub, ax=ax0, orientation="horizontal", fraction=0.1, pad=0.05, shrink=0.75, aspect=20, ticks=[
                                 vmin, vmax], format=f_, drawedges=False, boundaries=b_, values=v_)
                ax0.set_title(title)

        if(bool(kwargs.get("save", False))):
            fig.savefig(str(kwargs.get("save")), dpi=None, facecolor="none", edgecolor="none", orientation="portrait",
                        papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()

    def getlm(self):  # > compute all (l,m)
        index = np.arange(self.get_dim())
        n = 2 * self.paradict['lmax'] + 1
        m = np.ceil(
            (n - np.sqrt(n**2 - 8 * (index - self.paradict['lmax']))) / 2
                    ).astype(np.int)
        l = index - self.paradict['lmax'] * m + m * (m - 1) // 2
        return l, m


class gl_space(point_space):
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
        dtype : numpy.dtype, *optional*
            Data type of the field values (default: numpy.float64).

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        lm_space : A class for spherical harmonic components.

        Notes
        -----
        Only real-valued fields on the two-sphere are supported, i.e.
        `dtype` has to be either numpy.float64 or numpy.float32.

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
        dtype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for spherical spaces.
        vol : numpy.ndarray
            An array containing the pixel sizes.
    """

    def __init__(self, nlat, nlon=None, dtype=np.dtype('float64'),
                 datamodel='not', comm=gc['default_comm']):
        """
            Sets the attributes for a gl_space class instance.

            Parameters
            ----------
            nlat : int
                Number of latitudinal bins, or rings.
            nlon : int, *optional*
                Number of longitudinal bins (default: ``2*nlat - 1``).
            dtype : numpy.dtype, *optional*
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
        # check imports
        if not gc['use_libsharp']:
            raise ImportError(about._errors.cstring(
                "ERROR: libsharp_wrapper_gl not loaded."))

        self._cache_dict = {'check_codomain': {}}
        self.paradict = gl_space_paradict(nlat=nlat, nlon=nlon)

        # check data type
        dtype = np.dtype(dtype)
        if dtype not in [np.dtype('float32'), np.dtype('float64')]:
            about.warnings.cprint("WARNING: data type set to default.")
            dtype = np.dtype('float')
        self.dtype = dtype

        # set datamodel
        if datamodel not in ['not']:
            about.warnings.cprint("WARNING: datamodel set to default.")
        self.datamodel = datamodel

        self.discrete = False
        self.harmonic = False
        self.distances = tuple(gl.vol(self.paradict['nlat'],
                                      nlon=self.paradict['nlon']
                                      ).astype(np.float))
        self.comm = self._parse_comm(comm)

    @property
    def para(self):
        temp = np.array([self.paradict['nlat'],
                         self.paradict['nlon']], dtype=int)
        return temp

    @para.setter
    def para(self, x):
        self.paradict['nlat'] = x[0]
        self.paradict['nlon'] = x[1]

    def copy(self):
        return gl_space(nlat=self.paradict['nlat'],
                        nlon=self.paradict['nlon'],
                        dtype=self.dtype)

    def get_shape(self):
        return (np.int((self.paradict['nlat'] * self.paradict['nlon'])),)

    def get_dof(self, split=False):
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
        if split:
            return self.get_shape()
        else:
            return self.get_dim()

    def get_meta_volume(self, split=False):
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
        if not split:
            return np.float(4 * np.pi)
        else:
            mol = self.cast(1, dtype=np.float)
            return self.calc_weight(mol, power=1)

    # TODO: Extend to binning/log
    def enforce_power(self, spec, size=None, kindex=None):
        if kindex is None:
            kindex_size = self.paradict['nlat']
            kindex = np.arange(kindex_size,
                               dtype=np.array(self.distances).dtype)
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

            Notes
            -----
            Compatible codomains are instances of :py:class:`gl_space` and
            :py:class:`lm_space`.
        """
        if codomain is None:
            return False

        if not isinstance(codomain, space):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if self.datamodel is not codomain.datamodel:
            return False

        if self.comm is not codomain.comm:
            return False

        if isinstance(codomain, lm_space):
            nlat = self.paradict['nlat']
            nlon = self.paradict['nlon']
            lmax = codomain.paradict['lmax']
            mmax = codomain.paradict['mmax']
            # nlon==2*lat-1
            # lmax==nlat-1
            # lmax==mmax
            if (nlon == 2*nlat-1) and (lmax == nlat-1) and (lmax == mmax):
                return True

        return False

    def get_codomain(self, **kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Returns
            -------
            codomain : nifty.lm_space
                A compatible codomain.
        """
        nlat = self.paradict['nlat']
        lmax = nlat-1
        mmax = nlat-1
        # lmax,mmax = nlat-1,nlat-1
        if self.dtype == np.dtype('float32'):
            return lm_space(lmax=lmax, mmax=mmax, dtype=np.complex64,
                            datamodel=self.datamodel,
                            comm=self.comm)
        else:
            return lm_space(lmax=lmax, mmax=mmax, dtype=np.complex128,
                            datamodel=self.datamodel,
                            comm=self.comm)

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
            spec : {scalar, list, numpy.array, nifty.field, function},
            *optional*
                Power spectrum (default: 1).
            codomain : nifty.lm_space, *optional*
                A compatible codomain for power indexing (default: None).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.parse_arguments(self, **kwargs)

        if(arg is None):
            x = np.zeros(self.get_shape(), dtype=self.dtype)

        elif(arg['random'] == "pm1"):
            x = random.pm1(dtype=self.dtype, shape=self.get_shape())

        elif(arg['random'] == "gau"):
            x = random.gau(dtype=self.dtype,
                           shape=self.get_shape(),
                           mean=arg['mean'],
                           std=arg['std'])

        elif(arg['random'] == "syn"):
            nlat = self.paradict['nlat']
            nlon = self.paradict['nlon']
            lmax = nlat - 1
            if self.dtype == np.dtype('float32'):
                x = self.cast(gl.synfast_f(arg['spec'],
                                 nlat=nlat, nlon=nlon,
                                 lmax=lmax, mmax=lmax, alm=False))
            else:
                x = self.cast(gl.synfast(arg['spec'],
                               nlat=nlat, nlon=nlon,
                               lmax=lmax, mmax=lmax, alm=False))
            # weight if discrete
            if self.discrete:
                x = self.calc_weight(x, power=0.5)

        elif(arg['random'] == "uni"):
            x = random.uni(dtype=self.dtype,
                           shape=self.get_shape(),
                           vmin=arg['vmin'],
                           vmax=arg['vmax'])

        else:
            raise KeyError(about._errors.cstring(
                "ERROR: unsupported random key '" + str(arg['random']) + "'."))

        return x

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
        x = self._cast_to_np(x)
        # weight
        nlat = self.paradict['nlat']
        nlon = self.paradict['nlon']
        if self.dtype == np.dtype('float32'):
            return self.cast(gl.weight_f(x,
                               np.array(self.distances),
                               p=np.float32(power),
                               nlat=nlat, nlon=nlon,
                               overwrite=False))
        else:
            return self.cast(gl.weight(x,
                             np.array(self.distances),
                             p=np.float32(power),
                             nlat=nlat, nlon=nlon,
                             overwrite=False))

    def get_weight(self, power=1):
        # TODO: Check if this function is compatible to the rest of nifty
        # TODO: Can this be done more efficiently?
        dummy = self.dtype(1)
        weighted_dummy = self.calc_weight(dummy, power=power)
        return weighted_dummy / dummy

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

            Notes
            -----
            Only instances of the :py:class:`lm_space` or :py:class:`gl_space`
            classes are allowed as `codomain`.
        """
        x = self.cast(x)

        # Check if the given codomain is suitable for the transformation
        if not self.check_codomain(codomain):
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported codomain."))

        if isinstance(codomain, lm_space):
            # weight if discrete
            if self.discrete:
                x = self.calc_weight(x, power=-0.5)
            # transform
            nlat = self.paradict['nlat']
            nlon = self.paradict['nlon']
            lmax = codomain.paradict['lmax']
            mmax = codomain.paradict['mmax']

            if self.dtype == np.dtype('float32'):
                Tx = gl.map2alm_f(np.array(x),
                                  nlat=nlat, nlon=nlon,
                                  lmax=lmax, mmax=mmax)
            else:
                Tx = gl.map2alm(np.array(x),
                                nlat=nlat, nlon=nlon,
                                lmax=lmax, mmax=mmax)
        else:
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported transformation."))

        return codomain.cast(Tx.astype(codomain.dtype))

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
                of length in position space; for testing: a sigma of -1 will be
                reset to a reasonable value (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.
        """
        x = self.cast(x)
        # check sigma
        if sigma == 0:
            return self.unary_operation(x, op='copy')
        elif sigma == -1:
            about.infos.cprint("INFO: invalid sigma reset.")
            sigma = np.sqrt(2) * np.pi / self.paradict['nlat']
        elif sigma < 0:
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        # smooth
        nlat = self.paradict['nlat']
        return self.cast(gl.smoothmap(np.array(x),
                            nlat=nlat, nlon=self.paradict['nlon'],
                            lmax=nlat - 1, mmax=nlat - 1,
                            fwhm=0.0, sigma=sigma))

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
        """
        x = self.cast(x)
        # weight if discrete
        if self.discrete:
            x = self.calc_weight(x, power=-0.5)
        # calculate the power spectrum
        nlat = self.paradict['nlat']
        nlon = self.paradict['nlon']
        lmax = nlat - 1
        mmax = nlat - 1
        if self.dtype == np.dtype('float32'):
            return gl.anafast_f(np.array(x),
                                nlat=nlat, nlon=nlon,
                                lmax=lmax, mmax=mmax,
                                alm=False)
        else:
            return gl.anafast(np.array(x),
                              nlat=nlat, nlon=nlon,
                              lmax=lmax, mmax=mmax,
                              alm=False)

    def get_plot(self, x, title="", vmin=None, vmax=None, power=False,
                 unit="", norm=None, cmap=None, cbar=True, other=None,
                 legend=False, mono=True, **kwargs):
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
        if(not pl.isinteractive())and(not bool(kwargs.get("save", False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(power):
            x = self.calc_power(x)

            fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None, facecolor="none",
                            edgecolor="none", frameon=False, FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])

            xaxes = np.arange(self.para[0], dtype=np.int)
            if(vmin is None):
                vmin = np.min(x[:mono].tolist(
                ) + (xaxes * (2 * xaxes + 1) * x)[1:].tolist(), axis=None, out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist(
                ) + (xaxes * (2 * xaxes + 1) * x)[1:].tolist(), axis=None, out=None)
            ax0.loglog(xaxes[1:], (xaxes * (2 * xaxes + 1) * x)[1:], color=[0.0,
                                                                            0.5, 0.0], label="graph 0", linestyle='-', linewidth=2.0, zorder=1)
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
                            other[ii] = self.enforce_power(other[ii])
                elif(isinstance(other, field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other)]
                imax = max(1, len(other) - 1)
                for ii in xrange(len(other)):
                    ax0.loglog(xaxes[1:], (xaxes * (2 * xaxes + 1) * other[ii])[1:], color=[max(0.0, 1.0 - (2 * ii / imax)**2), 0.5 * ((2 * ii - imax) / imax)
                                                                                            ** 2, max(0.0, 1.0 - (2 * (ii - imax) / imax)**2)], label="graph " + str(ii + 1), linestyle='-', linewidth=1.0, zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5 * (xaxes[1] + xaxes[2]), other[ii][0], s=20, color=[max(0.0, 1.0 - (2 * ii / imax)**2), 0.5 * ((2 * ii - imax) / imax)**2, max(
                            0.0, 1.0 - (2 * (ii - imax) / imax)**2)], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1], xaxes[-1])
            ax0.set_xlabel(r"$l$")
            ax0.set_ylim(vmin, vmax)
            ax0.set_ylabel(r"$l(2l+1) C_l$")
            ax0.set_title(title)

        else:
            x = self.cast(x)
            if(vmin is None):
                vmin = np.min(x, axis=None, out=None)
            if(vmax is None):
                vmax = np.max(x, axis=None, out=None)
            if(norm == "log")and(vmin <= 0):
                raise ValueError(about._errors.cstring(
                    "ERROR: nonpositive value(s)."))

            fig = pl.figure(num=None, figsize=(8.5, 5.4), dpi=None, facecolor="none",
                            edgecolor="none", frameon=False, FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.02, 0.05, 0.96, 0.9])

            lon, lat = gl.bounds(self.para[0], nlon=self.para[1])
            lon = (lon - np.pi) * 180 / np.pi
            lat = (lat - np.pi / 2) * 180 / np.pi
            if(norm == "log"):
                n_ = ln(vmin=vmin, vmax=vmax)
            else:
                n_ = None
            sub = ax0.pcolormesh(lon, lat, np.roll(np.array(x).reshape((self.para[0], self.para[1]), order='C'), self.para[
                                 1] // 2, axis=1)[::-1, ::-1], cmap=cmap, norm=n_, vmin=vmin, vmax=vmax)
            ax0.set_xlim(-180, 180)
            ax0.set_ylim(-90, 90)
            ax0.set_aspect("equal")
            ax0.axis("off")
            if(cbar):
                if(norm == "log"):
                    f_ = lf(10, labelOnlyBase=False)
                    b_ = sub.norm.inverse(np.linspace(0, 1, sub.cmap.N + 1))
                    v_ = np.linspace(sub.norm.vmin, sub.norm.vmax, sub.cmap.N)
                else:
                    f_ = None
                    b_ = None
                    v_ = None
                cb0 = fig.colorbar(sub, ax=ax0, orientation="horizontal", fraction=0.1, pad=0.05, shrink=0.5, aspect=25, ticks=[
                                   vmin, vmax], format=f_, drawedges=False, boundaries=b_, values=v_)
                cb0.ax.text(0.5, -1.0, unit, fontdict=None, withdash=False, transform=cb0.ax.transAxes,
                            horizontalalignment="center", verticalalignment="center")
            ax0.set_title(title)

        if(bool(kwargs.get("save", False))):
            fig.savefig(str(kwargs.get("save")), dpi=None, facecolor="none", edgecolor="none", orientation="portrait",
                        papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()


class hp_space(point_space):
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
        dtype : numpy.dtype
            Data type of the field values, which is always numpy.float64.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for spherical spaces.
        vol : numpy.ndarray
            An array with one element containing the pixel size.
    """

    def __init__(self, nside, datamodel='not', comm=gc['default_comm']):
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
        # check imports
        if not gc['use_healpy']:
            raise ImportError(about._errors.cstring(
                "ERROR: healpy not available."))

        self._cache_dict = {'check_codomain': {}}
        # check parameters
        self.paradict = hp_space_paradict(nside=nside)

        self.dtype = np.dtype('float64')

        # set datamodel
        if datamodel not in ['not']:
            about.warnings.cprint("WARNING: datamodel set to default.")
        self.datamodel = datamodel

        self.discrete = False
        self.harmonic = False
        self.distances = (np.float(4*np.pi / (12*self.paradict['nside']**2)),)
        self.comm = self._parse_comm(comm)

    @property
    def para(self):
        temp = np.array([self.paradict['nside']], dtype=int)
        return temp

    @para.setter
    def para(self, x):
        self.paradict['nside'] = x[0]

    def copy(self):
        return hp_space(nside=self.paradict['nside'])

    def get_shape(self):
        return (np.int(12 * self.paradict['nside']**2),)

    def get_dof(self, split=False):
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
        if split:
            return self.get_shape()
        else:
            return self.get_dim()

    def get_meta_volume(self, split=False):
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
        if not split:
            return np.float(4 * np.pi)
        else:
            mol = self.cast(1, dtype=np.float)
            return self.calc_weight(mol, power=1)

    # TODO: Extend to binning/log
    def enforce_power(self, spec, size=None, kindex=None):
        if kindex is None:
            kindex_size = self.paradict['nside'] * 3
            kindex = np.arange(kindex_size,
                               dtype=np.array(self.distances).dtype)
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

            Notes
            -----
            Compatible codomains are instances of :py:class:`hp_space` and
            :py:class:`lm_space`.
        """
        if codomain is None:
            return False

        if not isinstance(codomain, space):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if self.datamodel is not codomain.datamodel:
            return False

        if self.comm is not codomain.comm:
            return False

        if isinstance(codomain, lm_space):
            nside = self.paradict['nside']
            lmax = codomain.paradict['lmax']
            mmax = codomain.paradict['mmax']
            # 3*nside-1==lmax
            # lmax==mmax
            if (3*nside-1 == lmax) and (lmax == mmax):
                return True

        return False

    def get_codomain(self, **kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Returns
            -------
            codomain : nifty.lm_space
                A compatible codomain.
        """
        lmax = 3*self.paradict['nside'] - 1
        mmax = lmax
        return lm_space(lmax=lmax, mmax=mmax, dtype=np.dtype('complex128'),
                        datamodel=self.datamodel,
                        comm=self.comm)

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

                - "pm1" (uniform distribution over {+1,-1}
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
            spec : {scalar, list, numpy.array, nifty.field, function},
                *optional*
                Power spectrum (default: 1).
            codomain : nifty.lm_space, *optional*
                A compatible codomain for power indexing (default: None).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.parse_arguments(self, **kwargs)

        if arg is None:
            x = np.zeros(self.get_shape(), dtype=self.dtype)

        elif arg['random'] == "pm1":
            x = random.pm1(dtype=self.dtype, shape=self.get_shape())

        elif arg['random'] == "gau":
            x = random.gau(dtype=self.dtype, shape=self.get_shape(),
                           mean=arg['mean'],
                           std=arg['std'])

        elif arg['random'] == "syn":
            nside = self.paradict['nside']
            lmax = 3*nside-1
            x = self.cast(hp.synfast(arg['spec'], nside, lmax=lmax, mmax=lmax, alm=False,
                           pol=True, pixwin=False, fwhm=0.0, sigma=None))
            # weight if discrete
            if self.discrete:
                x = self.calc_weight(x, power=0.5)

        elif arg['random'] == "uni":
            x = random.uni(dtype=self.dtype, shape=self.get_shape(),
                           vmin=arg['vmin'],
                           vmax=arg['vmax'])

        else:
            raise KeyError(about._errors.cstring(
                "ERROR: unsupported random key '" + str(arg['random']) + "'."))

        return x

    def calc_transform(self, x, codomain=None, niter=0, **kwargs):
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
                Number of iterations performed in the HEALPix basis
                transformation.

            Notes
            -----
            Only instances of the :py:class:`lm_space` or :py:class:`hp_space`
            classes are allowed as `codomain`.
        """
        x = self.cast(x)

        # Check if the given codomain is suitable for the transformation
        if not self.check_codomain(codomain):
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported codomain."))

        if isinstance(codomain, lm_space):
            # weight if discrete
            if self.discrete:
                x = self.calc_weight(x, power=-0.5)
            # transform
            Tx = hp.map2alm(x.copy(dtype=np.float64),
                            lmax=codomain.paradict['lmax'],
                            mmax=codomain.paradict['mmax'],
                            iter=niter, pol=True, use_weights=False,
                            datapath=None)

        else:
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported transformation."))

        return codomain.cast(Tx.astype(codomain.dtype))

    def calc_smooth(self, x, sigma=0, niter=0, **kwargs):
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
        nside = self.paradict['nside']

        x = self.cast(x)
        # check sigma
        if sigma == 0:
            return self.unary_operation(x, op='copy')
        elif sigma == -1:
            about.infos.cprint("INFO: invalid sigma reset.")
            # sqrt(2)*pi/(lmax+1)
            sigma = np.sqrt(2) * np.pi / (3. * nside)
        elif sigma < 0:
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        # smooth
        lmax = 3*nside-1
        mmax = lmax
        return self.cast(hp.smoothing(x, fwhm=0.0, sigma=sigma, pol=True,
                            iter=niter, lmax=lmax, mmax=mmax,
                            use_weights=False, datapath=None))

    def calc_power(self, x, niter=0, **kwargs):
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
        x = self.cast(x)
        # weight if discrete
        if self.discrete:
            x = self.calc_weight(x, power=-0.5)

        nside = self.paradict['nside']
        lmax = 3*nside-1
        mmax = lmax
        # power spectrum
        return hp.anafast(np.array(x), map2=None, nspec=None, lmax=lmax, mmax=mmax,
                          iter=niter, alm=False, pol=True, use_weights=False,
                          datapath=None)

    def get_plot(self, x, title="", vmin=None, vmax=None, power=False, unit="",
                 norm=None, cmap=None, cbar=True, other=None, legend=False,
                 mono=True, **kwargs):
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
        if(not pl.isinteractive())and(not bool(kwargs.get("save", False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(power):
            x = self.calc_power(x, **kwargs)

            fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None, facecolor="none",
                            edgecolor="none", frameon=False, FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])

            xaxes = np.arange(3 * self.para[0], dtype=np.int)
            if(vmin is None):
                vmin = np.min(x[:mono].tolist(
                ) + (xaxes * (2 * xaxes + 1) * x)[1:].tolist(), axis=None, out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist(
                ) + (xaxes * (2 * xaxes + 1) * x)[1:].tolist(), axis=None, out=None)
            ax0.loglog(xaxes[1:], (xaxes * (2 * xaxes + 1) * x)[1:], color=[0.0,
                                                                            0.5, 0.0], label="graph 0", linestyle='-', linewidth=2.0, zorder=1)
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
                            other[ii] = self.enforce_power(other[ii])
                elif(isinstance(other, field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other)]
                imax = max(1, len(other) - 1)
                for ii in xrange(len(other)):
                    ax0.loglog(xaxes[1:], (xaxes * (2 * xaxes + 1) * other[ii])[1:], color=[max(0.0, 1.0 - (2 * ii / imax)**2), 0.5 * ((2 * ii - imax) / imax)
                                                                                            ** 2, max(0.0, 1.0 - (2 * (ii - imax) / imax)**2)], label="graph " + str(ii + 1), linestyle='-', linewidth=1.0, zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5 * (xaxes[1] + xaxes[2]), other[ii][0], s=20, color=[max(0.0, 1.0 - (2 * ii / imax)**2), 0.5 * ((2 * ii - imax) / imax)**2, max(
                            0.0, 1.0 - (2 * (ii - imax) / imax)**2)], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1], xaxes[-1])
            ax0.set_xlabel(r"$\ell$")
            ax0.set_ylim(vmin, vmax)
            ax0.set_ylabel(r"$\ell(2\ell+1) C_\ell$")
            ax0.set_title(title)

        else:
            x = self.cast(x)
            if(norm == "log"):
                if(vmin is not None):
                    if(vmin <= 0):
                        raise ValueError(about._errors.cstring(
                            "ERROR: nonpositive value(s)."))
                elif(np.min(x, axis=None, out=None) <= 0):
                    raise ValueError(about._errors.cstring(
                        "ERROR: nonpositive value(s)."))
            if(cmap is None):
                cmap = pl.cm.jet  # default
            cmap.set_under(color='k', alpha=0.0)  # transparent box
            hp.mollview(x, fig=None, rot=None, coord=None, unit=unit, xsize=800, title=title, nest=False, min=vmin, max=vmax, flip="astro", remove_dip=False,
                        remove_mono=False, gal_cut=0, format="%g", format2="%g", cbar=cbar, cmap=cmap, notext=False, norm=norm, hold=False, margins=None, sub=None)
            fig = pl.gcf()

        if(bool(kwargs.get("save", False))):
            fig.savefig(str(kwargs.get("save")), dpi=None, facecolor="none", edgecolor="none", orientation="portrait",
                        papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()
