from __future__ import division

import itertools
import numpy as np
import pylab as pl
from matplotlib.colors import LogNorm as ln
from matplotlib.ticker import LogFormatter as lf

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.spaces.lm_space import LMSpace

from nifty.spaces.space import Space
from nifty.config import about, nifty_configuration as gc,\
                         dependency_injector as gdi
from gl_space_paradict import GLSpaceParadict
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

        # setup paradict
        self.paradict = GLSpaceParadict(nlat=nlat, nlon=nlon)

        # check and set data type
        dtype = np.dtype(dtype)
        if dtype not in [np.dtype('float32'), np.dtype('float64')]:
            about.warnings.cprint("WARNING: data type set to default.")
            dtype = np.dtype('float')
        self.dtype = dtype

        # GLSpace is not harmonic
        self._harmonic = False

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

    def weight(self, x, power=1, axes=None, inplace=False):
        # check if the axes provided are valid given the input shape
        if axes is not None and \
                not all(axis in range(len(x.shape)) for axis in axes):
            raise ValueError("ERROR: Provided axes does not match array shape")

        weight = np.array(list(
            itertools.chain.from_iterable(
                itertools.repeat(x ** power, self.paradict['nlon'])
                for x in gl.vol(self.paradict['nlat'])
            )
        ))

        if axes is not None:
            # reshape the weight array to match the input shape
            new_shape = np.ones(x.shape)
            for index in range(len(axes)):
                new_shape[index] = len(weight)
            weight = weight.reshape(new_shape)

        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x * weight

        return result_x

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
        from nifty.field import Field

        try:
            x = x.get_full_data()
        except AttributeError:
            pass

        if (not pl.isinteractive()) and (not bool(kwargs.get("save", False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        if (power):
            x = self.calc_power(x)

            fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None,
                            facecolor="none",
                            edgecolor="none", frameon=False,
                            FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])

            xaxes = np.arange(self.para[0], dtype=np.int)
            if (vmin is None):
                vmin = np.min(x[:mono].tolist(
                ) + (xaxes * (2 * xaxes + 1) * x)[1:].tolist(), axis=None,
                              out=None)
            if (vmax is None):
                vmax = np.max(x[:mono].tolist(
                ) + (xaxes * (2 * xaxes + 1) * x)[1:].tolist(), axis=None,
                              out=None)
            ax0.loglog(xaxes[1:], (xaxes * (2 * xaxes + 1) * x)[1:], color=[0.0,
                                                                            0.5,
                                                                            0.0],
                       label="graph 0", linestyle='-', linewidth=2.0, zorder=1)
            if (mono):
                ax0.scatter(0.5 * (xaxes[1] + xaxes[2]), x[0], s=20,
                            color=[0.0, 0.5, 0.0], marker='o',
                            cmap=None, norm=None, vmin=None, vmax=None,
                            alpha=None, linewidths=None, verts=None, zorder=1)

            if (other is not None):
                if (isinstance(other, tuple)):
                    other = list(other)
                    for ii in xrange(len(other)):
                        if (isinstance(other[ii], Field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii])
                elif (isinstance(other, Field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other)]
                imax = max(1, len(other) - 1)
                for ii in xrange(len(other)):
                    ax0.loglog(xaxes[1:],
                               (xaxes * (2 * xaxes + 1) * other[ii])[1:],
                               color=[max(0.0, 1.0 - (2 * ii / imax) ** 2),
                                      0.5 * ((2 * ii - imax) / imax)
                                      ** 2, max(0.0, 1.0 - (
                                   2 * (ii - imax) / imax) ** 2)],
                               label="graph " + str(ii + 1), linestyle='-',
                               linewidth=1.0, zorder=-ii)
                    if (mono):
                        ax0.scatter(0.5 * (xaxes[1] + xaxes[2]), other[ii][0],
                                    s=20,
                                    color=[max(0.0, 1.0 - (2 * ii / imax) ** 2),
                                           0.5 * ((2 * ii - imax) / imax) ** 2,
                                           max(
                                               0.0, 1.0 - (
                                               2 * (ii - imax) / imax) ** 2)],
                                    marker='o', cmap=None, norm=None, vmin=None,
                                    vmax=None, alpha=None, linewidths=None,
                                    verts=None, zorder=-ii)
                if (legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1], xaxes[-1])
            ax0.set_xlabel(r"$l$")
            ax0.set_ylim(vmin, vmax)
            ax0.set_ylabel(r"$l(2l+1) C_l$")
            ax0.set_title(title)

        else:
            x = self.cast(x)
            if (vmin is None):
                vmin = np.min(x, axis=None, out=None)
            if (vmax is None):
                vmax = np.max(x, axis=None, out=None)
            if (norm == "log") and (vmin <= 0):
                raise ValueError(about._errors.cstring(
                    "ERROR: nonpositive value(s)."))

            fig = pl.figure(num=None, figsize=(8.5, 5.4), dpi=None,
                            facecolor="none",
                            edgecolor="none", frameon=False,
                            FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.02, 0.05, 0.96, 0.9])

            lon, lat = gl.bounds(self.para[0], nlon=self.para[1])
            lon = (lon - np.pi) * 180 / np.pi
            lat = (lat - np.pi / 2) * 180 / np.pi
            if (norm == "log"):
                n_ = ln(vmin=vmin, vmax=vmax)
            else:
                n_ = None
            sub = ax0.pcolormesh(lon, lat, np.roll(
                np.array(x).reshape((self.para[0], self.para[1]), order='C'),
                self.para[
                    1] // 2, axis=1)[::-1, ::-1], cmap=cmap, norm=n_, vmin=vmin,
                                 vmax=vmax)
            ax0.set_xlim(-180, 180)
            ax0.set_ylim(-90, 90)
            ax0.set_aspect("equal")
            ax0.axis("off")
            if (cbar):
                if (norm == "log"):
                    f_ = lf(10, labelOnlyBase=False)
                    b_ = sub.norm.inverse(np.linspace(0, 1, sub.cmap.N + 1))
                    v_ = np.linspace(sub.norm.vmin, sub.norm.vmax, sub.cmap.N)
                else:
                    f_ = None
                    b_ = None
                    v_ = None
                cb0 = fig.colorbar(sub, ax=ax0, orientation="horizontal",
                                   fraction=0.1, pad=0.05, shrink=0.5,
                                   aspect=25, ticks=[
                        vmin, vmax], format=f_, drawedges=False, boundaries=b_,
                                   values=v_)
                cb0.ax.text(0.5, -1.0, unit, fontdict=None, withdash=False,
                            transform=cb0.ax.transAxes,
                            horizontalalignment="center",
                            verticalalignment="center")
            ax0.set_title(title)

        if (bool(kwargs.get("save", False))):
            fig.savefig(str(kwargs.get("save")), dpi=None, facecolor="none",
                        edgecolor="none", orientation="portrait",
                        papertype=None, format=None, transparent=False,
                        bbox_inches=None, pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()
