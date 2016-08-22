
from __future__ import division

import os
import numpy as np
import pylab as pl
from matplotlib.colors import LogNorm as ln
from matplotlib.ticker import LogFormatter as lf

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.spaces.space import Space

from nifty.config import about,\
                         nifty_configuration as gc,\
                         dependency_injector as gdi

from lm_space_paradict import LMSpaceParadict
# from nifty.nifty_power_indices import lm_power_indices
from nifty.nifty_random import random

gl = gdi.get('libsharp_wrapper_gl')
hp = gdi.get('healpy')

LM_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


class LMSpace(Space):
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

    def __init__(self, lmax, mmax=None, dtype=np.dtype('complex128')):
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

        self.paradict = LMSpaceParadict(lmax=lmax, mmax=mmax)

        # check data type
        dtype = np.dtype(dtype)
        if dtype not in [np.dtype('complex64'), np.dtype('complex128')]:
            about.warnings.cprint("WARNING: data type set to complex128.")
            dtype = np.dtype('complex128')
        self.dtype = dtype

        self.harmonic = True
        self.distances = (np.float(1),)

        self.power_indices = lm_power_indices(
                    lmax=self.paradict['lmax'],
                    dim=self.dim,
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
                if ii[0] not in ['_cache_dict', 'power_indices']]
        # Return the sorted identifiers as a tuple.
        return tuple(sorted(temp))

    def copy(self):
        return LMSpace(lmax=self.paradict['lmax'],
                       mmax=self.paradict['mmax'],
                       dtype=self.dtype)

    @property
    def shape(self):
        lmax = self.paradict['lmax']
        mmax = self.paradict['mmax']
        return (np.int((mmax + 1) * (lmax + 1) - ((mmax + 1) * mmax) // 2),)


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
            The spherical harmonics components with :math:`m=0` have meta
            volume 1, the ones with :math:`m>0` have meta volume 2, sinnce they
            each determine another component with negative :math:`m`.
        """
        return np.float(self.dof())

    @property
    def meta_volume_split(self):
        mol = self.cast(1, dtype=np.float)
        mol[self.paradict['lmax'] + 1:] = 2  # redundant: (l,m) and (l,-m)
        return mol

    def complement_cast(self, x, axis=None, **kwargs):
        if axis is None:
            lmax = self.paradict['lmax']
            complexity_mask = x[:lmax+1].iscomplex()
            if complexity_mask.any():
                about.warnings.cprint("WARNING: Taking the absolute values for " +
                                      "all complex entries where lmax==0")
                x[:lmax+1] = abs(x[:lmax+1])
        else:
            # TODO hermitianize only on specific axis
            lmax = self.paradict['lmax']
            complexity_mask = x[:lmax+1].iscomplex()
            if complexity_mask.any():
                about.warnings.cprint("WARNING: Taking the absolute values for " +
                                      "all complex entries where lmax==0")
                x[:lmax+1] = abs(x[:lmax+1])
        return x

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

        from nifty.spaces.hp_space import HPSpace
        from nifty.spaces.gl_space import GLSpace
        if not isinstance(codomain, Space):
            raise TypeError(about._errors.cstring(
                "ERROR: The given codomain must be a nifty lm_space."))

        elif isinstance(codomain, GLSpace):
            # lmax==mmax
            # nlat==lmax+1
            # nlon==2*lmax+1
            if ((self.paradict['lmax'] == self.paradict['mmax']) and
                    (codomain.paradict['nlat'] == self.paradict['lmax']+1) and
                    (codomain.paradict['nlon'] == 2*self.paradict['lmax']+1)):
                return True

        elif isinstance(codomain, HPSpace):
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
        from hp_space import HPSpace
        from gl_space import GLSpace
        if coname == 'gl' or (coname is None and gc['lm2gl']):
            if self.dtype == np.dtype('complex64'):
                new_dtype = np.float32
            elif self.dtype == np.dtype('complex128'):
                new_dtype = np.float64
            else:
                raise NotImplementedError
            nlat = self.paradict['lmax'] + 1
            nlon = self.paradict['lmax'] * 2 + 1
            return GLSpace(nlat=nlat, nlon=nlon, dtype=new_dtype)

        elif coname == 'hp' or (coname is None and not gc['lm2gl']):
            nside = (self.paradict['lmax']+1) // 3
            return HPSpace(nside=nside)

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

#        if arg is None:
#            x = 0
#
#        elif arg['random'] == "pm1":
#            x = random.pm1(dtype=self.dtype, shape=self.shape)
#
#        elif arg['random'] == "gau":
#            x = random.gau(dtype=self.dtype,
#                           shape=self.shape,
#                           mean=arg['mean'],
#                           std=arg['std'])

        if arg['random'] == "syn":
            lmax = self.paradict['lmax']
            mmax = self.paradict['mmax']
            if self.dtype == np.dtype('complex64'):
                if gc['use_libsharp']:
                    sample = gl.synalm_f(arg['spec'], lmax=lmax, mmax=mmax)
                else:
                    sample = hp.synalm(
                                arg['spec'].astype(np.complex128),
                                lmax=lmax, mmax=mmax).astype(np.complex64,
                                                             copy=False)
            else:
                if gc['use_healpy']:
                    sample = hp.synalm(arg['spec'], lmax=lmax, mmax=mmax)
                else:
                    sample = gl.synalm(arg['spec'], lmax=lmax, mmax=mmax)

        else:
            sample = super(LMSpace, self).get_random_values(**arg)

#        elif arg['random'] == "uni":
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

#    def calc_dot(self, x, y):
#        """
#            Computes the discrete inner product of two given arrays of field
#            values.
#
#            Parameters
#            ----------
#            x : numpy.ndarray
#                First array
#            y : numpy.ndarray
#                Second array
#
#            Returns
#            -------
#            dot : scalar
#                Inner product of the two arrays.
#        """
#        x = self.cast(x)
#        y = self.cast(y)
#
#        lmax = self.paradict['lmax']
#
#        x_low = x[:lmax + 1]
#        x_high = x[lmax + 1:]
#        y_low = y[:lmax + 1]
#        y_high = y[lmax + 1:]
#
#        dot = (x_low.real * y_low.real).sum()
#        dot += 2 * (x_high.real * y_high.real).sum()
#        dot += 2 * (x_high.imag * y_high.imag).sum()
#        return dot

    def dot_contraction(self, x, axes):
        assert len(axes) == 1
        axis = axes[0]
        lmax = self.paradict['lmax']

        # extract the low and high parts of x
        extractor = ()
        extractor += (slice(None),)*axis
        low_extractor = extractor + (slice(None, lmax+1), )
        high_extractor = extractor + (slice(lmax+1), )

        result = x[low_extractor].sum(axes) + 2 * x[high_extractor].sum(axes)
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
        """
        x = self.cast(x)

        if codomain is None:
            codomain = self.get_codomain()

        # Check if the given codomain is suitable for the transformation
        if not self.check_codomain(codomain):
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported codomain."))

        # if self.datamodel != 'not':
        #     about.warnings.cprint(
        #         "WARNING: Field data is consolidated to all nodes for "
        #         "external alm2map method!")

        np_x = x.get_full_data()
        from hp_space import HPSpace
        from gl_space import GLSpace
        if isinstance(codomain, GLSpace):
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

        elif isinstance(codomain, HPSpace):
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

        # if self.datamodel != 'not':
        #     about.warnings.cprint(
        #         "WARNING: Field data is consolidated to all nodes for "
        #         "external smoothalm method!")

        np_x = x.get_full_data()

        if gc['use_healpy']:
            np_smoothed_x = hp.smoothalm(np_x,
                                         fwhm=0.0,
                                         sigma=sigma,
                                         pol=True,
                                         mmax=self.paradict['mmax'],
                                         verbose=False,
                                         inplace=False)
        else:
            np_smoothed_x = gl.smoothalm(np_x,
                                         lmax=self.paradict['lmax'],
                                         mmax=self.paradict['mmax'],
                                         fwhm=0.0,
                                         sigma=sigma,
                                         overwrite=False)
        return self.cast(np_smoothed_x)

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

        # if self.datamodel != 'not':
        #     about.warnings.cprint(
        #         "WARNING: Field data is consolidated to all nodes for "
        #         "external anaalm/alm2cl method!")

        np_x = x.get_full_data()

        # power spectrum
        if self.dtype == np.dtype('complex64'):
            if gc['use_libsharp']:
                result = gl.anaalm_f(np_x, lmax=lmax, mmax=mmax)
            else:
                np_x = np_x.astype(np.complex128, copy=False)
                result = hp.alm2cl(np_x,
                                   alms2=None,
                                   lmax=lmax,
                                   mmax=mmax,
                                   lmax_out=lmax,
                                   nspec=None)
        else:
            if gc['use_healpy']:
                result = hp.alm2cl(np_x,
                                   alms2=None,
                                   lmax=lmax,
                                   mmax=mmax,
                                   lmax_out=lmax,
                                   nspec=None)
            else:
                result = gl.anaalm(np_x,
                                   lmax=lmax,
                                   mmax=mmax)

        if self.dtype == np.dtype('complex64'):
            result = result.astype(np.float32, copy=False)
        elif self.dtype == np.dtype('complex128'):
            result = result.astype(np.float64, copy=False)
        else:
            raise NotImplementedError(about._errors.cstring(
                "ERROR: dtype %s not known to calc_power method." %
                str(self.dtype)))

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
        from nifty.field import Field

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
        index = np.arange(self.dim)
        n = 2 * self.paradict['lmax'] + 1
        m = np.ceil(
            (n - np.sqrt(n**2 - 8 * (index - self.paradict['lmax']))) / 2
                    ).astype(np.int)
        l = index - self.paradict['lmax'] * m + m * (m - 1) // 2
        return l, m
