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

import numpy as np
import pylab as pl

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.lm import LMSpace

from nifty.space import Space
from nifty.field import Field

from nifty.config import about,\
                         nifty_configuration as gc,\
                         dependency_injector as gdi
from nifty.nifty_paradict import hp_space_paradict
from nifty.nifty_random import random

hp = gdi.get('healpy')

HP_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


class HPSpace(Space):
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

    def __init__(self, nside):
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

        self.discrete = False
        self.harmonic = False
        self.distances = (np.float(4*np.pi / (12*self.paradict['nside']**2)),)

    @property
    def para(self):
        temp = np.array([self.paradict['nside']], dtype=int)
        return temp

    @para.setter
    def para(self, x):
        self.paradict['nside'] = x[0]

    def copy(self):
        return HPSpace(nside=self.paradict['nside'])

    @property
    def shape(self):
        return (np.int(12 * self.paradict['nside']**2),)

    @property
    def meta_volume(self):
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
        return np.float(4 * np.pi)

    @property
    def meta_volume_split(self):
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

        if not isinstance(codomain, Space):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if isinstance(codomain, LMSpace):
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
        return LMSpace(lmax=lmax, mmax=mmax, dtype=np.dtype('complex128'))

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

#        if arg is None:
#            x = np.zeros(self.shape, dtype=self.dtype)
#
#        elif arg['random'] == "pm1":
#            x = random.pm1(dtype=self.dtype, shape=self.shape)
#
#        elif arg['random'] == "gau":
#            x = random.gau(dtype=self.dtype, shape=self.shape,
#                           mean=arg['mean'],
#                           std=arg['std'])

        if arg['random'] == "syn":
            nside = self.paradict['nside']
            lmax = 3*nside-1
            sample = hp.synfast(arg['spec'],
                                nside,
                                lmax=lmax, mmax=lmax,
                                alm=False, pol=True, pixwin=False,
                                fwhm=0.0, sigma=None)
            # weight if discrete
            if self.discrete:
                sample = self.calc_weight(sample, power=0.5)

        else:
            sample = super(HPSpace, self).get_random_values(**arg)


#        elif arg['random'] == "uni":
#            x = random.uni(dtype=self.dtype, shape=self.shape,
#                           vmin=arg['vmin'],
#                           vmax=arg['vmax'])
#
#        else:
#            raise KeyError(about._errors.cstring(
#                "ERROR: unsupported random key '" + str(arg['random']) + "'."))
        sample = self.cast(sample)
        return sample

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

        # TODO look at these kinds of checks maybe need replacement
        # if self.datamodel != 'not':
        #     about.warnings.cprint(
        #         "WARNING: Field data is consolidated to all nodes for "
        #         "external map2alm method!")

        np_x = x.get_full_data()

        if isinstance(codomain, LMSpace):
            # weight if discrete
            if self.discrete:
                x = self.calc_weight(x, power=-0.5)
            # transform
            np_Tx = hp.map2alm(np_x.astype(np.float64, copy=False),
                               lmax=codomain.paradict['lmax'],
                               mmax=codomain.paradict['mmax'],
                               iter=niter, pol=True, use_weights=False,
                               datapath=None)

        else:
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported transformation."))

        return codomain.cast(np_Tx)

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

        # if self.datamodel != 'not':
        #     about.warnings.cprint(
        #         "WARNING: Field data is consolidated to all nodes for "
        #         "external smoothalm method!")

        np_x = x.get_full_data()

        lmax = 3*nside-1
        mmax = lmax
        result = hp.smoothing(np_x, fwhm=0.0, sigma=sigma, pol=True,
                              iter=niter, lmax=lmax, mmax=mmax,
                              use_weights=False, datapath=None)
        return self.cast(result)

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

        # if self.datamodel != 'not':
        #     about.warnings.cprint(
        #         "WARNING: Field data is consolidated to all nodes for "
        #         "external smoothalm method!")

        np_x = x.get_full_data()

        # power spectrum
        return hp.anafast(np_x, map2=None, nspec=None, lmax=lmax, mmax=mmax,
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
        try:
            x = x.get_full_data()
        except AttributeError:
            pass

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
                        if(isinstance(other[ii], Field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii])
                elif(isinstance(other, Field)):
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
