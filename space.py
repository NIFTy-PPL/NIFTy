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

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty_paradict import space_paradict

from nifty.config import about

POINT_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


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
        self._cache_dict = {'check_codomain': {}}
        self.paradict = space_paradict(**kwargs)

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

    def copy(self):
        return Space(dtype=self.dtype, **self.paradict.parameters)

    def getitem(self, data, key):
        return data[key]

    def setitem(self, data, update, key):
        data[key] = update

    def apply_scalar_function(self, x, function, inplace=False):
        return x.apply_scalar_function(function, inplace=inplace)

    @property
    def shape(self):
        return (self.paradict['num'],)

    @property
    def dim(self):
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
        return np.prod(self.shape)

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
    def dof_split(self):
        """
            Computes the number of degrees of freedom of the space, i.e./  the
            number of points for real-valued fields and twice that number for
            complex-valued fields.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        dof = self.shape
        if issubclass(self.dtype.type, np.complexfloating):
            dof = tuple(np.array(dof)*2)
        return dof

    @property
    def vol(self, split=False):
        return np.prod(self.distances)

    @property
    def vol_split(self):
        return self.distances

    @property
    def meta_volume(self):
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
        return self.dim() * self.vol()

    @property
    def meta_volume_split(self):
        mol = self.cast(1, dtype=np.dtype('float'))
        return self.calc_weight(mol, power=1)

    def complement_cast(self, x, axis=None):
        return x

    def enforce_power(self, spec, **kwargs):
        """
            Raises an error since the power spectrum is ill-defined for point
            spaces.
        """
        raise AttributeError(about._errors.cstring(
            "ERROR: the definition of power spectra is ill-defined for " +
            "(unstructured) point spaces."))

    def _enforce_power_helper(self, spec, size, kindex):
        # TODO: Resolve this import by splitting nifty_core into nifty_space
        # and nifty_point_space
        from nifty_field import field

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

        if not isinstance(codomain, Space):
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

#    def get_random_values(self, **kwargs):
#        """
#            Generates random field values according to the specifications given
#            by the parameters.
#
#            Returns
#            -------
#            x : numpy.ndarray
#                Valid field values.
#
#            Other parameters
#            ----------------
#            random : string, *optional*
#                Specifies the probability distribution from which the random
#                numbers are to be drawn.
#                Supported distributions are:
#
#                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
#                - "gau" (normal distribution with zero-mean and a given
#                standard
#                    deviation or variance)
#                - "syn" (synthesizes from a given power spectrum)
#                - "uni" (uniform distribution over [vmin,vmax[)
#
#                (default: None).
#            dev : float, *optional*
#                Standard deviation (default: 1).
#            var : float, *optional*
#                Variance, overriding `dev` if both are specified
#                (default: 1).
#            spec : {scalar, list, numpy.ndarray, nifty.field, function},
#            *optional*
#                Power spectrum (default: 1).
#            pindex : numpy.ndarray, *optional*
#                Indexing array giving the power spectrum index of each band
#                (default: None).
#            kindex : numpy.ndarray, *optional*
#                Scale of each band (default: None).
#            codomain : nifty.space, *optional*
#                A compatible codomain with power indices (default: None).
#            log : bool, *optional*
#                Flag specifying if the spectral binning is performed on
#                logarithmic
#                scale or not; if set, the number of used bins is set
#                automatically (if not given otherwise); by default no binning
#                is done (default: None).
#            nbin : integer, *optional*
#                Number of used spectral bins; if given `log` is set to
#                ``False``;
#                integers below the minimum of 3 induce an automatic setting;
#                by default no binning is done (default: None).
#            binbounds : {list, array}, *optional*
#                User specific inner boundaries of the bins, which are preferred
#                over the above parameters; by default no binning is done
#                (default: None).
#                vmin : {scalar, list, ndarray, field}, *optional*
#                Lower limit of the uniform distribution if ``random == "uni"``
#                (default: 0).
#            vmin : float, *optional*
#                Lower limit for a uniform distribution (default: 0).
#            vmax : float, *optional*
#                Upper limit for a uniform distribution (default: 1).
#        """
#
#        arg = random.parse_arguments(self, **kwargs)
#
#        if arg is None:
#            return self.cast(0)
#
#        # Prepare the empty distributed_data_object
#        sample = distributed_data_object(
#                                    global_shape=self.shape,
#                                    dtype=self.dtype)
#
#        # Case 1: uniform distribution over {-1,+1}/{1,i,-1,-i}
#        if arg['random'] == 'pm1':
#            sample.apply_generator(lambda s: random.pm1(dtype=self.dtype,
#                                                        shape=s))
#
#        # Case 2: normal distribution with zero-mean and a given standard
#        #         deviation or variance
#        elif arg['random'] == 'gau':
#            std = arg['std']
#            if np.isscalar(std) or std is None:
#                processed_std = std
#            else:
#                try:
#                    processed_std = sample.distributor. \
#                        extract_local_data(std)
#                except(AttributeError):
#                    processed_std = std
#
#            sample.apply_generator(lambda s: random.gau(dtype=self.dtype,
#                                                        shape=s,
#                                                        mean=arg['mean'],
#                                                        std=processed_std))
#
#        # Case 3: uniform distribution
#        elif arg['random'] == 'uni':
#            sample.apply_generator(lambda s: random.uni(dtype=self.dtype,
#                                                        shape=s,
#                                                        vmin=arg['vmin'],
#                                                        vmax=arg['vmax']))
#        return sample

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

        if complex_weights_Q:
            real_bincount = x.bincount(weights=weights.real,
                                       minlength=minlength)
            imag_bincount = x.bincount(weights=weights.imag,
                                       minlength=minlength)
            return real_bincount + imag_bincount
        else:
            return x.bincount(weights=weights, minlength=minlength)

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
        string += 'dtype: ' + str(self.dtype) + "\n"
        string += 'discrete: ' + str(self.discrete) + "\n"
        string += 'distances: ' + str(self.distances) + "\n"
        return string
