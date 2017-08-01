# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division
from builtins import zip
#from builtins import str
from builtins import range

import ast
import itertools
import numpy as np

from keepers import Versionable,\
                    Loggable

from d2o import distributed_data_object,\
    STRATEGIES as DISTRIBUTION_STRATEGIES

from .config import nifty_configuration as gc

from .domain_object import DomainObject

from .spaces.power_space import PowerSpace

from . import nifty_utilities as utilities
from .random import Random
from functools import reduce


class Field(Loggable, Versionable, object):
    """ The discrete representation of a continuous field over multiple spaces.

    In NIFTY, Fields are used to store data arrays and carry all the needed
    metainformation (i.e. the domain) for operators to be able to work on them.
    In addition Field has methods to work with power-spectra.

    Parameters
    ----------
    domain : DomainObject
        One of the space types NIFTY supports. RGSpace, GLSpace, HPSpace,
        LMSpace or PowerSpace. It might also be a FieldArray, which is
        an unstructured domain.

    val : scalar, numpy.ndarray, distributed_data_object, Field
        The values the array should contain after init. A scalar input will
        fill the whole array with this scalar. If an array is provided the
        array's dimensions must match the domain's.

    dtype : type
        A numpy.type. Most common are int, float and complex.

    distribution_strategy: optional[{'fftw', 'equal', 'not', 'freeform'}]
        Specifies which distributor will be created and used.
        'fftw'      uses the distribution strategy of pyfftw,
        'equal'     tries to  distribute the data as uniform as possible
        'not'       does not distribute the data at all
        'freeform'  distribute the data according to the given local data/shape

    copy: boolean

    Attributes
    ----------
    val : distributed_data_object

    domain : DomainObject
        See Parameters.
    domain_axes : tuple of tuples
        Enumerates the axes of the Field
    dtype : type
        Contains the datatype stored in the Field.
    distribution_strategy : string
        Name of the used distribution_strategy.

    Raise
    -----
    TypeError
        Raised if
            *the given domain contains something that is not a DomainObject
             instance
            *val is an array that has a different dimension than the domain

    Examples
    --------
    >>> a = Field(RGSpace([4,5]),val=2)
    >>> a.val
    <distributed_data_object>
    array([[2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2]])
    >>> a.dtype
    dtype('int64')

    See Also
    --------
    distributed_data_object

    """

    # ---Initialization methods---

    def __init__(self, domain=None, val=None, dtype=None,
                 distribution_strategy=None, copy=False):
        self.domain = self._parse_domain(domain=domain, val=val)
        self.domain_axes = self._get_axes_tuple(self.domain)

        self.dtype = self._infer_dtype(dtype=dtype,
                                       val=val)

        self.distribution_strategy = self._parse_distribution_strategy(
                                distribution_strategy=distribution_strategy,
                                val=val)

        if val is None:
            self._val = None
        else:
            self.set_val(new_val=val, copy=copy)


    def _parse_domain(self, domain, val=None):
        if domain is None:
            if isinstance(val, Field):
                domain = val.domain
            else:
                domain = ()
        elif isinstance(domain, DomainObject):
            domain = (domain,)
        elif not isinstance(domain, tuple):
            domain = tuple(domain)

        for d in domain:
            if not isinstance(d, DomainObject):
                raise TypeError(
                    "Given domain contains something that is not a "
                    "DomainObject instance.")
        return domain

    def _get_axes_tuple(self, things_with_shape, start=0):
        i = start
        axes_list = []
        for thing in things_with_shape:
            l = []
            for j in range(len(thing.shape)):
                l += [i]
                i += 1
            axes_list += [tuple(l)]
        return tuple(axes_list)

    def _infer_dtype(self, dtype, val):
        if dtype is None:
            try:
                dtype = val.dtype
            except AttributeError:
                try:
                    if val is None:
                        raise TypeError
                    dtype = np.result_type(val)
                except(TypeError):
                    dtype = np.dtype(gc['default_field_dtype'])
        else:
            dtype = np.dtype(dtype)

        dtype = np.result_type(dtype, np.float)

        return dtype

    def _parse_distribution_strategy(self, distribution_strategy, val):
        if distribution_strategy is None:
            if isinstance(val, distributed_data_object):
                distribution_strategy = val.distribution_strategy
            elif isinstance(val, Field):
                distribution_strategy = val.distribution_strategy
            else:
                self.logger.debug("distribution_strategy set to default!")
                distribution_strategy = gc['default_distribution_strategy']
        elif distribution_strategy not in DISTRIBUTION_STRATEGIES['global']:
            raise ValueError(
                    "distribution_strategy must be a global-type "
                    "strategy.")
        return distribution_strategy

    # ---Factory methods---

    @classmethod
    def from_random(cls, random_type, domain=None, dtype=None,
                    distribution_strategy=None, **kwargs):
        """ Draws a random field with the given parameters.

        Parameters
        ----------
        cls : class

        random_type : String
            'pm1', 'normal', 'uniform' are the supported arguments for this
            method.

        domain : DomainObject
            The domain of the output random field

        dtype : type
            The datatype of the output random field

        distribution_strategy : all supported distribution strategies
            The distribution strategy of the output random field

        Returns
        -------
        out : Field
            The output object.

        See Also
        --------
        power_synthesize


        """

        # create a initially empty field
        f = cls(domain=domain, dtype=dtype,
                distribution_strategy=distribution_strategy)

        # now use the processed input in terms of f in order to parse the
        # random arguments
        random_arguments = cls._parse_random_arguments(random_type=random_type,
                                                       f=f,
                                                       **kwargs)

        # extract the distributed_data_object from f and apply the appropriate
        # random number generator to it
        sample = f.get_val(copy=False)
        generator_function = getattr(Random, random_type)
        sample.apply_generator(
            lambda shape: generator_function(dtype=f.dtype,
                                             shape=shape,
                                             **random_arguments))
        return f

    @staticmethod
    def _parse_random_arguments(random_type, f, **kwargs):
        if random_type == "pm1":
            random_arguments = {}

        elif random_type == "normal":
            mean = kwargs.get('mean', 0)
            std = kwargs.get('std', 1)
            random_arguments = {'mean': mean,
                                'std': std}

        elif random_type == "uniform":
            low = kwargs.get('low', 0)
            high = kwargs.get('high', 1)
            random_arguments = {'low': low,
                                'high': high}

        else:
            raise KeyError(
                "unsupported random key '" + str(random_type) + "'.")

        return random_arguments

    # ---Powerspectral methods---

    def power_analyze(self, spaces=None, logarithmic=None, nbin=None,
                      binbounds=None, keep_phase_information=False):
        """ Computes the square root power spectrum for a subspace of `self`.

        Creates a PowerSpace for the space addressed by `spaces` with the given
        binning and computes the power spectrum as a Field over this
        PowerSpace. This can only be done if the subspace to  be analyzed is a
        harmonic space. The resulting field has the same units as the initial
        field, corresponding to the square root of the power spectrum.

        Parameters
        ----------
        spaces : int *optional*
            The subspace for which the powerspectrum shall be computed
            (default : None).
        logarithmic : boolean *optional*
            True if the output PowerSpace should use logarithmic binning.
            {default : None}
        nbin : int *optional*
            The number of bins the resulting PowerSpace shall have
            (default : None).
            if nbin==None : maximum number of bins is used
        binbounds : array-like *optional*
            Inner bounds of the bins (default : None).
            Overrides nbin and logarithmic.
            if binbounds==None : bins are inferred.
        keep_phase_information : boolean, *optional*
            If False, return a real-valued result containing the power spectrum
            of the input Field.
            If True, return a complex-valued result whose real component
            contains the power spectrum computed from the real part of the
            input Field, and whose imaginary component contains the power
            spectrum computed from the imaginary part of the input Field.
            The absolute value of this result should be identical to the output
            of power_analyze with keep_phase_information=False.
            (default : False).

        Raise
        -----
        ValueError
            Raised if
                *len(domain) is != 1 when spaces==None
                *len(spaces) is != 1 if not None
                *the analyzed space is not harmonic

        Returns
        -------
        out : Field
            The output object. It's domain is a PowerSpace and it contains
            the power spectrum of 'self's field.

        See Also
        --------
        power_synthesize, PowerSpace

        """

        # check if all spaces in `self.domain` are either harmonic or
        # power_space instances
        for sp in self.domain:
            if not sp.harmonic and not isinstance(sp, PowerSpace):
                self.logger.info(
                    "Field has a space in `domain` which is neither "
                    "harmonic nor a PowerSpace.")

        # check if the `spaces` input is valid
        spaces = utilities.cast_axis_to_tuple(spaces, len(self.domain))
        if spaces is None:
            spaces = list(range(len(self.domain)))

        if len(spaces) == 0:
            raise ValueError(
                "No space for analysis specified.")

        if keep_phase_information:
            parts_val = self._hermitian_decomposition(
                                              domain=self.domain,
                                              val=self.val,
                                              spaces=spaces,
                                              domain_axes=self.domain_axes,
                                              preserve_gaussian_variance=False)
            parts = [self.copy_empty().set_val(part_val, copy=False)
                     for part_val in parts_val]
        else:
            parts = [self]

        parts = [abs(part)**2 for part in parts]

        for space_index in spaces:
            parts = [self._single_power_analyze(
                                work_field=part,
                                space_index=space_index,
                                logarithmic=logarithmic,
                                nbin=nbin,
                                binbounds=binbounds)
                     for part in parts]

        if keep_phase_information:
            result_field = parts[0] + 1j*parts[1]
        else:
            result_field = parts[0]

        return result_field

    @classmethod
    def _single_power_analyze(cls, work_field, space_index, logarithmic, nbin,
                              binbounds):

        if not work_field.domain[space_index].harmonic:
            raise ValueError(
                "The analyzed space must be harmonic.")

        # Create the target PowerSpace instance:
        # If the associated signal-space field was real, we extract the
        # hermitian and anti-hermitian parts of `self` and put them
        # into the real and imaginary parts of the power spectrum.
        # If it was complex, all the power is put into a real power spectrum.

        distribution_strategy = \
            work_field.val.get_axes_local_distribution_strategy(
                work_field.domain_axes[space_index])

        harmonic_domain = work_field.domain[space_index]
        power_domain = PowerSpace(harmonic_partner=harmonic_domain,
                                  distribution_strategy=distribution_strategy,
                                  logarithmic=logarithmic, nbin=nbin,
                                  binbounds=binbounds)

        power_spectrum = cls._calculate_power_spectrum(
                                field_val=work_field.val,
                                pdomain=power_domain,
                                axes=work_field.domain_axes[space_index])

        # create the result field and put power_spectrum into it
        result_domain = list(work_field.domain)
        result_domain[space_index] = power_domain
        result_dtype = power_spectrum.dtype

        result_field = work_field.copy_empty(
                   domain=result_domain,
                   dtype=result_dtype,
                   distribution_strategy=power_spectrum.distribution_strategy)
        result_field.set_val(new_val=power_spectrum, copy=False)

        return result_field

    @classmethod
    def _calculate_power_spectrum(cls, field_val, pdomain, axes=None):

        pindex = pdomain.pindex
        # MR FIXME: how about iterating over slices, instead of replicating
        # pindex? Would save memory and probably isn't slower.
        if axes is not None:
            pindex = cls._shape_up_pindex(
                            pindex=pindex,
                            target_shape=field_val.shape,
                            target_strategy=field_val.distribution_strategy,
                            axes=axes)
        power_spectrum = pindex.bincount(weights=field_val,
                                         axis=axes)
        rho = pdomain.rho
        if axes is not None:
            new_rho_shape = [1, ] * len(power_spectrum.shape)
            new_rho_shape[axes[0]] = len(rho)
            rho = rho.reshape(new_rho_shape)
        power_spectrum /= rho

        return power_spectrum

    @staticmethod
    def _shape_up_pindex(pindex, target_shape, target_strategy, axes):
        if pindex.distribution_strategy not in \
                DISTRIBUTION_STRATEGIES['global']:
            raise ValueError("pindex's distribution strategy must be "
                             "global-type")

        if pindex.distribution_strategy in DISTRIBUTION_STRATEGIES['slicing']:
            if ((0 not in axes) or
                    (target_strategy is not pindex.distribution_strategy)):
                raise ValueError(
                    "A slicing distributor shall not be reshaped to "
                    "something non-sliced.")

        semiscaled_shape = [1, ] * len(target_shape)
        for i in axes:
            semiscaled_shape[i] = target_shape[i]
        local_data = pindex.get_local_data(copy=False)
        semiscaled_local_data = local_data.reshape(semiscaled_shape)
        result_obj = pindex.copy_empty(global_shape=target_shape,
                                       distribution_strategy=target_strategy)
        result_obj.set_full_data(semiscaled_local_data, copy=False)

        return result_obj

    def power_synthesize(self, spaces=None, real_power=True, real_signal=True,
                         mean=None, std=None, distribution_strategy=None):
        """ Yields a sampled field with `self`**2 as its power spectrum.

        This method draws a Gaussian random field in the harmonic partner
        domain of this fields domains, using this field as power spectrum.

        Parameters
        ----------
        spaces : {tuple, int, None} *optional*
            Specifies the subspace containing all the PowerSpaces which
            should be converted (default : None).
            if spaces==None : Tries to convert the whole domain.
        real_power : boolean *optional*
            Determines whether the power spectrum is treated as intrinsically
            real or complex (default : True).
        real_signal : boolean *optional*
            True will result in a purely real signal-space field
            (default : True).
        mean : float *optional*
            The mean of the Gaussian noise field which is used for the Field
            synthetization (default : None).
            if mean==None : mean will be set to 0
        std : float *optional*
            The standard deviation of the Gaussian noise field which is used
            for the Field synthetization (default : None).
            if std==None : std will be set to 1

        Returns
        -------
        out : Field
            The output object. A random field created with the power spectrum
            stored in the `spaces` in `self`.

        Notes
        -----
        For this the spaces specified by `spaces` must be a PowerSpace.
        This expects this field to be the square root of a power spectrum, i.e.
        to have the unit of the field to be sampled.

        See Also
        --------
        power_analyze

        Raises
        ------
        ValueError : If domain specified by `spaces` is not a PowerSpace.

        """

        # check if the `spaces` input is valid
        spaces = utilities.cast_axis_to_tuple(spaces, len(self.domain))

        if spaces is None:
            spaces = list(range(len(self.domain)))

        for power_space_index in spaces:
            power_space = self.domain[power_space_index]
            if not isinstance(power_space, PowerSpace):
                raise ValueError("A PowerSpace is needed for field "
                                 "synthetization.")

        # create the result domain
        result_domain = list(self.domain)
        for power_space_index in spaces:
            power_space = self.domain[power_space_index]
            harmonic_domain = power_space.harmonic_partner
            result_domain[power_space_index] = harmonic_domain

        # create random samples: one or two, depending on whether the
        # power spectrum is real or complex
        if real_power:
            result_list = [None]
        else:
            result_list = [None, None]

        if distribution_strategy is None:
            distribution_strategy = gc['default_distribution_strategy']

        result_list = [self.__class__.from_random(
                             'normal',
                             mean=mean,
                             std=std,
                             domain=result_domain,
                             dtype=np.complex,
                             distribution_strategy=distribution_strategy)
                       for x in result_list]

        # from now on extract the values from the random fields for further
        # processing without killing the fields.
        # if the signal-space field should be real, hermitianize the field
        # components

        spec = self.val.get_full_data()
        spec = np.sqrt(spec)

        for power_space_index in spaces:
            spec = self._spec_to_rescaler(spec, result_list, power_space_index)
        local_rescaler = spec

        result_val_list = [x.val for x in result_list]

        # apply the rescaler to the random fields
        result_val_list[0].apply_scalar_function(
                                            lambda x: x * local_rescaler.real,
                                            inplace=True)

        if not real_power:
            result_val_list[1].apply_scalar_function(
                                            lambda x: x * local_rescaler.imag,
                                            inplace=True)

        if real_signal:
            result_val_list = [self._hermitian_decomposition(
                                            result_domain,
                                            result_val,
                                            spaces,
                                            result_list[0].domain_axes,
                                            preserve_gaussian_variance=True)[0]
                               for result_val in result_val_list]

        # store the result into the fields
        [x.set_val(new_val=y, copy=False) for x, y in
            zip(result_list, result_val_list)]

        if real_power:
            result = result_list[0]
        else:
            result = result_list[0] + 1j*result_list[1]

        return result

    @staticmethod
    def _hermitian_decomposition(domain, val, spaces, domain_axes,
                                 preserve_gaussian_variance=False):

        flipped_val = val
        for space in spaces:
            flipped_val = domain[space].hermitianize_inverter(
                                                    x=flipped_val,
                                                    axes=domain_axes[space])
        flipped_val = flipped_val.conjugate()
        h = (val + flipped_val)/2.
        a = val - h

        # correct variance
        if preserve_gaussian_variance:
            assert issubclass(val.dtype.type, np.complexfloating),\
                    "complex input field is needed here"
            h *= np.sqrt(2)
            a *= np.sqrt(2)

#            The code below should not be needed in practice, since it would
#            only ever be called when hermitianizing a purely real field.
#            However it might be of educational use and keep us from forgetting
#            how these things are done ...

#            if not issubclass(val.dtype.type, np.complexfloating):
#                # in principle one must not correct the variance for the fixed
#                # points of the hermitianization. However, for a complex field
#                # the input field loses half of its power at its fixed points
#                # in the `hermitian` part. Hence, here a factor of sqrt(2) is
#                # also necessary!
#                # => The hermitianization can be done on a space level since
#                # either nothing must be done (LMSpace) or ALL points need a
#                # factor of sqrt(2)
#                # => use the preserve_gaussian_variance flag in the
#                # hermitian_decomposition method above.
#
#                # This code is for educational purposes:
#                fixed_points = [domain[i].hermitian_fixed_points()
#                                for i in spaces]
#                fixed_points = [[fp] if fp is None else fp
#                                for fp in fixed_points]
#
#                for product_point in itertools.product(*fixed_points):
#                    slice_object = np.array((slice(None), )*len(val.shape),
#                                            dtype=np.object)
#                    for i, sp in enumerate(spaces):
#                        point_component = product_point[i]
#                        if point_component is None:
#                            point_component = slice(None)
#                        slice_object[list(domain_axes[sp])] = point_component
#
#                    slice_object = tuple(slice_object)
#                    h[slice_object] /= np.sqrt(2)
#                    a[slice_object] /= np.sqrt(2)

        return (h, a)

    def _spec_to_rescaler(self, spec, result_list, power_space_index):
        power_space = self.domain[power_space_index]

        # weight the random fields with the power spectrum
        # therefore get the pindex from the power space
        pindex = power_space.pindex
        # take the local data from pindex. This data must be compatible to the
        # local data of the field given the slice of the PowerSpace
        local_distribution_strategy = \
            result_list[0].val.get_axes_local_distribution_strategy(
                result_list[0].domain_axes[power_space_index])

        if pindex.distribution_strategy is not local_distribution_strategy:
            raise AttributeError(
                "The distribution_strategy of pindex does not fit the "
                "slice_local distribution strategy of the synthesized field.")

        # Now use numpy advanced indexing in order to put the entries of the
        # power spectrum into the appropriate places of the pindex array.
        # Do this for every 'pindex-slice' in parallel using the 'slice(None)'s
        local_pindex = pindex.get_local_data(copy=False)

        local_blow_up = [slice(None)]*len(spec.shape)
        # it is important to count from behind, since spec potentially grows
        # with every iteration
        index = self.domain_axes[power_space_index][0]-len(self.shape)
        local_blow_up[index] = local_pindex
        # here, the power_spectrum is distributed into the new shape
        local_rescaler = spec[local_blow_up]
        return local_rescaler

    # ---Properties---

    def set_val(self, new_val=None, copy=False):
        """ Sets the fields distributed_data_object.

        Parameters
        ----------
        new_val : scalar, array-like, Field, None *optional*
            The values to be stored in the field.
            {default : None}

        copy : boolean, *optional*
            If False, Field tries to not copy the input data but use it
            directly.
            {default : False}
        See Also
        --------
        val

        """

        new_val = self.cast(new_val)
        if copy:
            new_val = new_val.copy()
        self._val = new_val
        return self

    def get_val(self, copy=False):
        """ Returns the distributed_data_object associated with this Field.

        Parameters
        ----------
        copy : boolean
            If true, a copy of the Field's underlying distributed_data_object
            is returned.

        Returns
        -------
        out : distributed_data_object

        See Also
        --------
        val

        """

        if self._val is None:
            self.set_val(None)

        if copy:
            return self._val.copy()
        else:
            return self._val

    @property
    def val(self):
        """ Returns the distributed_data_object associated with this Field.

        Returns
        -------
        out : distributed_data_object

        See Also
        --------
        get_val

        """

        return self.get_val(copy=False)

    @val.setter
    def val(self, new_val):
        self.set_val(new_val=new_val, copy=False)

    @property
    def shape(self):
        """ Returns the total shape of the Field's data array.

        Returns
        -------
        out : tuple
            The output object. The tuple contains the dimensions of the spaces
            in domain.

        See Also
        --------
        dim

        """
        if not hasattr(self, '_shape'):
            shape_tuple = tuple(sp.shape for sp in self.domain)
            try:
                global_shape = reduce(lambda x, y: x + y, shape_tuple)
            except TypeError:
                global_shape = ()
            self._shape = global_shape
        return self._shape

    @property
    def dim(self):
        """ Returns the total number of pixel-dimensions the field has.

        Effectively, all values from shape are multiplied.

        Returns
        -------
        out : int
            The dimension of the Field.

        See Also
        --------
        shape

        """

        dim_tuple = tuple(sp.dim for sp in self.domain)
        try:
            return int(reduce(lambda x, y: x * y, dim_tuple))
        except TypeError:
            return 0

    @property
    def dof(self):
        """ Returns the total number of degrees of freedom the Field has. For
        real Fields this is equal to `self.dim`. For complex Fields it is
        2*`self.dim`.

        """

        dof = self.dim
        if issubclass(self.dtype.type, np.complexfloating):
            dof *= 2
        return dof

    @property
    def total_volume(self):
        """ Returns the total volume of all spaces in the domain.
        """

        volume_tuple = tuple(sp.total_volume for sp in self.domain)
        try:
            return reduce(lambda x, y: x * y, volume_tuple)
        except TypeError:
            return 0.

    # ---Special unary/binary operations---

    def cast(self, x=None, dtype=None):
        """ Transforms x to a d2o with the correct dtype and shape.

        Parameters
        ----------
        x : scalar, d2o, Field, array_like
            The input that shall be casted on a d2o of the same shape like the
            domain.

        dtype : type
            The datatype the output shall have. This can be used to override
            the fields dtype.

        Returns
        -------
        out : distributed_data_object
            The output object.

        See Also
        --------
        _actual_cast

        """
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)

        casted_x = x

        for ind, sp in enumerate(self.domain):
            casted_x = sp.pre_cast(casted_x,
                                   axes=self.domain_axes[ind])

        casted_x = self._actual_cast(casted_x, dtype=dtype)

        for ind, sp in enumerate(self.domain):
            casted_x = sp.post_cast(casted_x,
                                    axes=self.domain_axes[ind])

        return casted_x

    def _actual_cast(self, x, dtype=None):
        if isinstance(x, Field):
            x = x.get_val()

        if dtype is None:
            dtype = self.dtype

        return_x = distributed_data_object(
                            global_shape=self.shape,
                            dtype=dtype,
                            distribution_strategy=self.distribution_strategy)
        return_x.set_full_data(x, copy=False)
        return return_x

    def copy(self, domain=None, dtype=None, distribution_strategy=None):
        """ Returns a full copy of the Field.

        If no keyword arguments are given, the returned object will be an
        identical copy of the original Field. By explicit specification one is
        able to define the domain, the dtype and the distribution_strategy of
        the returned Field.

        Parameters
        ----------
        domain : DomainObject
            The new domain the Field shall have.

        dtype : type
            The new dtype the Field shall have.

        distribution_strategy : all supported distribution strategies
            The new distribution strategy the Field shall have.

        Returns
        -------
        out : Field
            The output object. An identical copy of 'self'.

        See Also
        --------
        copy_empty

        """

        copied_val = self.get_val(copy=True)
        new_field = self.copy_empty(
                                domain=domain,
                                dtype=dtype,
                                distribution_strategy=distribution_strategy)
        new_field.set_val(new_val=copied_val, copy=False)
        return new_field

    def copy_empty(self, domain=None, dtype=None, distribution_strategy=None):
        """ Returns an empty copy of the Field.

        If no keyword arguments are given, the returned object will be an
        identical copy of the original Field. The memory for the data array
        is only allocated but not actively set to any value
        (c.f. numpy.ndarray.copy_empty). By explicit specification one is able
        to change the domain, the dtype and the distribution_strategy of the
        returned Field.

        Parameters
        ----------
        domain : DomainObject
            The new domain the Field shall have.

        dtype : type
            The new dtype the Field shall have.

        distribution_strategy : string, all supported distribution strategies
            The distribution strategy the new Field should have.

        Returns
        -------
        out : Field
            The output object.

        See Also
        --------
        copy

        """

        if domain is None:
            domain = self.domain
        else:
            domain = self._parse_domain(domain)

        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)

        if distribution_strategy is None:
            distribution_strategy = self.distribution_strategy

        fast_copyable = True
        try:
            for i in range(len(self.domain)):
                if self.domain[i] is not domain[i]:
                    fast_copyable = False
                    break
        except IndexError:
            fast_copyable = False

        if (fast_copyable and dtype == self.dtype and
                distribution_strategy == self.distribution_strategy):
            new_field = self._fast_copy_empty()
        else:
            new_field = Field(domain=domain,
                              dtype=dtype,
                              distribution_strategy=distribution_strategy)
        return new_field

    def _fast_copy_empty(self):
        # make an empty field
        new_field = EmptyField()
        # repair its class
        new_field.__class__ = self.__class__
        # copy domain, codomain and val
        for key, value in list(self.__dict__.items()):
            if key != '_val':
                new_field.__dict__[key] = value
            else:
                new_field.__dict__[key] = self.val.copy_empty()
        return new_field

    def weight(self, power=1, inplace=False, spaces=None):
        """ Weights the pixels of `self` with their invidual pixel-volume.

        Parameters
        ----------
        power : number
            The pixels get weighted with the volume-factor**power.

        inplace : boolean
            If True, `self` will be weighted and returned. Otherwise, a copy
            is made.

        spaces : tuple of ints
            Determines on which subspace the operation takes place.

        Returns
        -------
        out : Field
            The weighted field.

        """
        if inplace:
            new_field = self
        else:
            new_field = self.copy_empty()

        new_val = self.get_val(copy=False)

        spaces = utilities.cast_axis_to_tuple(spaces, len(self.domain))
        if spaces is None:
            spaces = list(range(len(self.domain)))

        for ind, sp in enumerate(self.domain):
            if ind in spaces:
                new_val = sp.weight(new_val,
                                    power=power,
                                    axes=self.domain_axes[ind],
                                    inplace=inplace)

        new_field.set_val(new_val=new_val, copy=False)
        return new_field

    def vdot(self, x=None, spaces=None, bare=False):
        """ Computes the volume-factor-aware dot product of 'self' with x.

        Parameters
        ----------
        x : Field
            The domain of x must contain `self.domain`

        spaces : tuple of ints
            If the domain of `self` and `x` are not the same, `spaces` specfies
            the mapping.

        bare : boolean
            If true, no volume factors will be included in the computation.

        Returns
        -------
        out : float, complex

        """
        if not isinstance(x, Field):
            raise ValueError("The dot-partner must be an instance of " +
                             "the NIFTy field class")

        # Compute the dot respecting the fact of discrete/continuous spaces
        if bare:
            y = self
        else:
            y = self.weight(power=1)

        if spaces is None:
            x_val = x.get_val(copy=False)
            y_val = y.get_val(copy=False)
            result = (y_val.conjugate() * x_val).sum()
            return result
        else:
            # create a diagonal operator which is capable of taking care of the
            # axes-matching
            from nifty.operators.diagonal_operator import DiagonalOperator
            diagonal = y.val.conjugate()
            diagonalOperator = DiagonalOperator(domain=y.domain,
                                                diagonal=diagonal,
                                                copy=False)
            dotted = diagonalOperator(x, spaces=spaces)
            return dotted.sum(spaces=spaces)

    def norm(self):
        """ Computes the L2-norm of the field values.

        Returns
        -------
        norm : scalar
            The L2-norm of the field values.

        """
        return np.sqrt(np.abs(self.vdot(x=self)))

    def conjugate(self, inplace=False):
        """ Retruns the complex conjugate of the field.

        Parameters
        ----------
        inplace : boolean
            Decides whether the conjugation should be performed inplace.

        Returns
        -------
        cc : field
            The complex conjugated field.

        """

        if inplace:
            work_field = self
        else:
            work_field = self.copy_empty()

        new_val = self.get_val(copy=False)
        new_val = new_val.conjugate()
        work_field.set_val(new_val=new_val, copy=False)

        return work_field

    # ---General unary/contraction methods---

    def __pos__(self):
        """ x.__pos__() <==> +x

        Returns a (positive) copy of `self`.

        """

        return self.copy()

    def __neg__(self):
        """ x.__neg__() <==> -x

        Returns a negative copy of `self`.

        """

        return_field = self.copy_empty()
        new_val = -self.get_val(copy=False)
        return_field.set_val(new_val, copy=False)
        return return_field

    def __abs__(self):
        """ x.__abs__() <==> abs(x)

        Returns an absolute valued copy of `self`.

        """

        new_val = abs(self.get_val(copy=False))
        return_field = self.copy_empty(dtype=new_val.dtype)
        return_field.set_val(new_val, copy=False)
        return return_field

    def _contraction_helper(self, op, spaces):
        # build a list of all axes
        if spaces is None:
            spaces = range(len(self.domain))
        else:
            spaces = utilities.cast_axis_to_tuple(spaces, len(self.domain))

        axes_list = tuple(self.domain_axes[sp_index] for sp_index in spaces)

        try:
            axes_list = reduce(lambda x, y: x+y, axes_list)
        except TypeError:
            axes_list = ()

        # perform the contraction on the d2o
        data = self.get_val(copy=False)
        data = getattr(data, op)(axis=axes_list)

        # check if the result is scalar or if a result_field must be constr.
        if np.isscalar(data):
            return data
        else:
            return_domain = tuple(self.domain[i]
                                  for i in range(len(self.domain))
                                  if i not in spaces)

            return_field = Field(domain=return_domain,
                                 val=data,
                                 copy=False)
            return return_field

    def sum(self, spaces=None):
        return self._contraction_helper('sum', spaces)

    def prod(self, spaces=None):
        return self._contraction_helper('prod', spaces)

    def all(self, spaces=None):
        return self._contraction_helper('all', spaces)

    def any(self, spaces=None):
        return self._contraction_helper('any', spaces)

    def min(self, spaces=None):
        return self._contraction_helper('min', spaces)

    def nanmin(self, spaces=None):
        return self._contraction_helper('nanmin', spaces)

    def max(self, spaces=None):
        return self._contraction_helper('max', spaces)

    def nanmax(self, spaces=None):
        return self._contraction_helper('nanmax', spaces)

    def mean(self, spaces=None):
        return self._contraction_helper('mean', spaces)

    def var(self, spaces=None):
        return self._contraction_helper('var', spaces)

    def std(self, spaces=None):
        return self._contraction_helper('std', spaces)

    # ---General binary methods---

    def _binary_helper(self, other, op, inplace=False):
        # if other is a field, make sure that the domains match
        if isinstance(other, Field):
            try:
                assert len(other.domain) == len(self.domain)
                for index in range(len(self.domain)):
                    assert other.domain[index] == self.domain[index]
            except AssertionError:
                raise ValueError(
                    "domains are incompatible.")
            other = other.get_val(copy=False)

        self_val = self.get_val(copy=False)
        return_val = getattr(self_val, op)(other)

        if inplace:
            working_field = self
        else:
            working_field = self.copy_empty(dtype=return_val.dtype)

        working_field.set_val(return_val, copy=False)
        return working_field

    def __add__(self, other):
        """ x.__add__(y) <==> x+y

        See Also
        --------
        _binary_helper

        """

        return self._binary_helper(other, op='__add__')

    def __radd__(self, other):
        """ x.__radd__(y) <==> y+x

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__radd__')

    def __iadd__(self, other):
        """ x.__iadd__(y) <==> x+=y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__iadd__', inplace=True)

    def __sub__(self, other):
        """ x.__sub__(y) <==> x-y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__sub__')

    def __rsub__(self, other):
        """ x.__rsub__(y) <==> y-x

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__rsub__')

    def __isub__(self, other):
        """ x.__isub__(y) <==> x-=y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__isub__', inplace=True)

    def __mul__(self, other):
        """ x.__mul__(y) <==> x*y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__mul__')

    def __rmul__(self, other):
        """ x.__rmul__(y) <==> y*x

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__rmul__')

    def __imul__(self, other):
        """ x.__imul__(y) <==> x*=y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__imul__', inplace=True)

    def __div__(self, other):
        """ x.__div__(y) <==> x/y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__div__')

    def __truediv__(self, other):
        """ x.__truediv__(y) <==> x/y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__truediv__')

    def __rdiv__(self, other):
        """ x.__rdiv__(y) <==> y/x

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__rdiv__')

    def __rtruediv__(self, other):
        """ x.__rtruediv__(y) <==> y/x

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__rtruediv__')

    def __idiv__(self, other):
        """ x.__idiv__(y) <==> x/=y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__idiv__', inplace=True)

    def __pow__(self, other):
        """ x.__pow__(y) <==> x**y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__pow__')

    def __rpow__(self, other):
        """ x.__rpow__(y) <==> y**x

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__rpow__')

    def __ipow__(self, other):
        """ x.__ipow__(y) <==> x**=y

        See Also
        --------
        _builtin_helper

        """

        return self._binary_helper(other, op='__ipow__', inplace=True)

    def __lt__(self, other):
        """ x.__lt__(y) <==> x<y

        See Also
        --------
        _binary_helper

        """

        return self._binary_helper(other, op='__lt__')

    def __le__(self, other):
        """ x.__le__(y) <==> x<=y

        See Also
        --------
        _binary_helper

        """

        return self._binary_helper(other, op='__le__')

    def __ne__(self, other):
        """ x.__ne__(y) <==> x!=y

        See Also
        --------
        _binary_helper

        """

        if other is None:
            return True
        else:
            return self._binary_helper(other, op='__ne__')

    def __eq__(self, other):
        """ x.__eq__(y) <==> x=y

        See Also
        --------
        _binary_helper

        """

        if other is None:
            return False
        else:
            return self._binary_helper(other, op='__eq__')

    def __ge__(self, other):
        """ x.__ge__(y) <==> x>=y

        See Also
        --------
        _binary_helper

        """

        return self._binary_helper(other, op='__ge__')

    def __gt__(self, other):
        """ x.__gt__(y) <==> x>y

        See Also
        --------
        _binary_helper

        """

        return self._binary_helper(other, op='__gt__')

    def __repr__(self):
        return "<nifty_core.field>"

    def __str__(self):
        minmax = [self.min(), self.max()]
        mean = self.mean()
        return "nifty_core.field instance\n- domain      = " + \
               repr(self.domain) + \
               "\n- val         = " + repr(self.get_val()) + \
               "\n  - min.,max. = " + str(minmax) + \
               "\n  - mean = " + str(mean)

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group.attrs['dtype'] = self.dtype.name
        hdf5_group.attrs['distribution_strategy'] = self.distribution_strategy
        hdf5_group.attrs['domain_axes'] = str(self.domain_axes)
        hdf5_group['num_domain'] = len(self.domain)

        if self._val is None:
            ret_dict = {}
        else:
            ret_dict = {'val': self.val}

        for i in range(len(self.domain)):
            ret_dict['s_' + str(i)] = self.domain[i]

        return ret_dict

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        # create empty field
        new_field = EmptyField()
        # reset class
        new_field.__class__ = cls
        # set values
        temp_domain = []
        for i in range(hdf5_group['num_domain'][()]):
            temp_domain.append(repository.get('s_' + str(i), hdf5_group))
        new_field.domain = tuple(temp_domain)

        new_field.domain_axes = ast.literal_eval(
                                hdf5_group.attrs['domain_axes'])

        try:
            new_field._val = repository.get('val', hdf5_group)
        except(KeyError):
            new_field._val = None

        new_field.dtype = np.dtype(hdf5_group.attrs['dtype'])
        new_field.distribution_strategy =\
            hdf5_group.attrs['distribution_strategy']

        return new_field


class EmptyField(Field):
    def __init__(self):
        pass
