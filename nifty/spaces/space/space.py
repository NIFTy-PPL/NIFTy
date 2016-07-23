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

from nifty.config import about
from space_paradict import SpaceParadict


class Space(object):
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

    def __init__(self, dtype=np.dtype('float'), **kwargs):
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
        self.paradict = SpaceParadict(**kwargs)

        # parse dtype
        dtype = np.dtype(dtype)
        self.dtype = dtype
        self._harmonic = None

    @property
    def harmonic(self):
        return self._harmonic

    def __hash__(self):
        # Extract the identifying parts from the vars(self) dict.
        result_hash = 0
        for (key, item) in vars(self).items():
            if key in []:
                continue
            result_hash ^= item.__hash__() ^ int(hash(key)/117)
        return result_hash

    def __eq__(self, x):
        if isinstance(x, type(self)):
            return hash(self) == hash(x)
        else:
            return False

    def copy(self):
        return self.__class__(dtype=self.dtype, **self.paradict.parameters)

    @property
    def shape(self):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: There is no generic shape for the Space base class."))

    @property
    def dim(self):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: There is no generic dim for the Space base class."))

    @property
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
        dof = self.dim
        if issubclass(self.dtype.type, np.complexfloating):
            dof = dof * 2
        return dof

    @property
    def total_volume(self):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: There is no generic volume for the Space base class."))

    def complement_cast(self, x, axes=None):
        return x

    def weight(self, x, power=1, axes=None, inplace=False):
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
        raise NotImplementedError

    def dot_contraction(self, x, axes):
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
        return x.sum(axis=axes)

    def compute_k_array(self, distribution_strategy):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: There is no generic k_array for Space base class."))

    def smooth(self, x, **kwargs):
        raise AttributeError(about._errors.cstring(
            "ERROR: There is no generic smoothing for Space base class."))

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

        if (norm == "log") and (vmin <= 0):
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
        string += "dtype: " + str(self.dtype) + "\n"
        string += "harmonic: " + str(self.harmonic) + "\n"
        return string
