from __future__ import division

import numpy as np
import pylab as pl
from matplotlib.colors import LogNorm as ln
from matplotlib.ticker import LogFormatter as lf

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.lm.lm_space import LMSpace

from nifty.space import Space
from nifty.config import about,\
                         nifty_configuration as gc,\
                         dependency_injector as gdi
from nifty.nifty_paradict import gl_space_paradict
from nifty.nifty_random import random

gl = gdi.get('libsharp_wrapper_gl')

GL_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


class GLSpace(Space):
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

    def __init__(self, nlat, nlon=None, dtype=np.dtype('float64')):
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

        self.discrete = False
        self.harmonic = False
        self.distances = (tuple(gl.vol(self.paradict['nlat'],
                                       nlon=self.paradict['nlon']
                                       ).astype(np.float)),)

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
        return GLSpace(nlat=self.paradict['nlat'],
                       nlon=self.paradict['nlon'],
                       dtype=self.dtype)

    @property
    def shape(self):
        return (np.int((self.paradict['nlat'] * self.paradict['nlon'])),)

    @property
    def vol(self):
        return np.sum(self.paradict['nlon'] * np.array(self.distances[0]))

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
            For Gauss-Legendre pixelizations, the meta volumes are the pixel
            sizes.
        """
        return np.float(4 * np.pi)

    @property
    def meta_volume_split(self):
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

        if not isinstance(codomain, Space):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if isinstance(codomain, LMSpace):
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
            return LMSpace(lmax=lmax, mmax=mmax, dtype=np.complex64)
        else:
            return LMSpace(lmax=lmax, mmax=mmax, dtype=np.complex128)

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

#        if(arg is None):
#            x = np.zeros(self.shape, dtype=self.dtype)
#
#        elif(arg['random'] == "pm1"):
#            x = random.pm1(dtype=self.dtype, shape=self.shape)
#
#        elif(arg['random'] == "gau"):
#            x = random.gau(dtype=self.dtype,
#                           shape=self.shape,
#                           mean=arg['mean'],
#                           std=arg['std'])
#
        if(arg['random'] == "syn"):
            nlat = self.paradict['nlat']
            nlon = self.paradict['nlon']
            lmax = nlat - 1
            if self.dtype == np.dtype('float32'):
                sample = gl.synfast_f(arg['spec'],
                                      nlat=nlat, nlon=nlon,
                                      lmax=lmax, mmax=lmax, alm=False)
            else:
                sample = gl.synfast(arg['spec'],
                                    nlat=nlat, nlon=nlon,
                                    lmax=lmax, mmax=lmax, alm=False)
            # weight if discrete
            if self.discrete:
                sample = self.calc_weight(sample, power=0.5)

        else:
            sample = super(GLSpace, self).get_random_values(**arg)


#        elif(arg['random'] == "uni"):
#            x = random.uni(dtype=self.dtype,
#                           shape=self.shape,
#                           vmin=arg['vmin'],
#                           vmax=arg['vmax'])
#
#        else:
#            raise KeyError(about._errors.cstring(
#                "ERROR: unsupported random key '" + str(arg['random']) + "'."))
        sample = self.cast(sample)
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
        x = self.cast(x)

        # if self.datamodel != 'not':
        #     about.warnings.cprint(
        #         "WARNING: Field data is consolidated to all nodes for "
        #         "external alm2map method!")
        np_x = x.get_full_data()

        # weight
        nlat = self.paradict['nlat']
        nlon = self.paradict['nlon']
        if self.dtype == np.dtype('float32'):
            np_result = gl.weight_f(np_x,
                                    np.array(self.distances),
                                    p=np.float32(power),
                                    nlat=nlat, nlon=nlon,
                                    overwrite=False)
        else:
            np_result = gl.weight(np_x,
                                  np.array(self.distances),
                                  p=np.float32(power),
                                  nlat=nlat, nlon=nlon,
                                  overwrite=False)
        # return self.cast(np_result)
        return np_result

    def get_weight(self, power=1):
        # TODO: Check if this function is compatible to the rest of nifty
        # TODO: Can this be done more efficiently?
        dummy = self.dtype.type(1)
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

        if codomain is None:
            codomain = self.get_codomain()

        # Check if the given codomain is suitable for the transformation
        if not self.check_codomain(codomain):
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported codomain."))

        if isinstance(codomain, LMSpace):

            # weight if discrete
            if self.discrete:
                x = self.calc_weight(x, power=-0.5)
            # transform
            nlat = self.paradict['nlat']
            nlon = self.paradict['nlon']
            lmax = codomain.paradict['lmax']
            mmax = codomain.paradict['mmax']

            # if self.datamodel != 'not':
            #     about.warnings.cprint(
            #         "WARNING: Field data is consolidated to all nodes for "
            #         "external map2alm method!")

            np_x = x.get_full_data()

            if self.dtype == np.dtype('float32'):
                Tx = gl.map2alm_f(np_x,
                                  nlat=nlat, nlon=nlon,
                                  lmax=lmax, mmax=mmax)
            else:
                Tx = gl.map2alm(np_x,
                                nlat=nlat, nlon=nlon,
                                lmax=lmax, mmax=mmax)
        else:
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported transformation."))

        return codomain.cast(Tx)

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

        # if self.datamodel != 'not':
        #     about.warnings.cprint(
        #         "WARNING: Field data is consolidated to all nodes for "
        #         "external smoothmap method!")

        np_x = x.get_full_data()

        result = self.cast(gl.smoothmap(np_x,
                           nlat=nlat, nlon=self.paradict['nlon'],
                           lmax=nlat - 1, mmax=nlat - 1,
                           fwhm=0.0, sigma=sigma))
        return result

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

        # if self.datamodel != 'not':
        #     about.warnings.cprint(
        #         "WARNING: Field data is consolidated to all nodes for "
        #         "external anafast method!")

        np_x = x.get_full_data()

        if self.dtype == np.dtype('float32'):
            result = gl.anafast_f(np_x,
                                  nlat=nlat, nlon=nlon,
                                  lmax=lmax, mmax=mmax,
                                  alm=False)
        else:
            result = gl.anafast(np_x,
                                nlat=nlat, nlon=nlon,
                                lmax=lmax, mmax=mmax,
                                alm=False)

        return result

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
        try:
            x = x.get_full_data()
        except AttributeError:
            pass

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

