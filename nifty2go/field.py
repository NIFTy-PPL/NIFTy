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
from builtins import range

import numpy as np

from .domain_object import DomainObject

from .spaces.power_space import PowerSpace

from . import nifty_utilities as utilities
from .random import Random
from functools import reduce


class Field(object):
    """ The discrete representation of a continuous field over multiple spaces.

    In NIFTY, Fields are used to store data arrays and carry all the needed
    metainformation (i.e. the domain) for operators to be able to work on them.
    In addition, Field has methods to work with power spectra.

    Parameters
    ----------
    domain : DomainObject
        One of the space types NIFTY supports. RGSpace, GLSpace, HPSpace,
        LMSpace or PowerSpace. It might also be a FieldArray, which is
        an unstructured domain.

    val : scalar, numpy.ndarray, Field
        The values the array should contain after init. A scalar input will
        fill the whole array with this scalar. If an array is provided the
        array's dimensions must match the domain's.

    dtype : type
        A numpy.type. Most common are float and complex.

    copy: boolean

    Attributes
    ----------
    val : numpy.ndarray

    domain : DomainObject
        See Parameters.
    domain_axes : tuple of tuples
        Enumerates the axes of the Field
    dtype : type
        Contains the datatype stored in the Field.

    Raise
    -----
    TypeError
        Raised if
            *the given domain contains something that is not a DomainObject
             instance
            *val is an array that has a different dimension than the domain

    """

    # ---Initialization methods---

    def __init__(self, domain=None, val=None, dtype=None, copy=False):
        self.domain = self._parse_domain(domain=domain, val=val)
        self.domain_axes = self._get_axes_tuple(self.domain)

        shape_tuple = tuple(sp.shape for sp in self.domain)
        if len(shape_tuple) == 0:
            global_shape = ()
        else:
            global_shape = reduce(lambda x, y: x + y, shape_tuple)
        dtype = self._infer_dtype(dtype=dtype, val=val)
        if isinstance(val, Field):
            if self.domain != val.domain:
                raise ValueError("Domain mismatch")
            self._val = np.array(val.val, dtype=dtype, copy=copy)
        elif (np.isscalar(val)):
            self._val = np.full(global_shape, dtype=dtype, fill_value=val)
        elif isinstance(val, np.ndarray):
            if global_shape == val.shape:
                self._val = np.array(val, dtype=dtype, copy=copy)
            else:
                raise ValueError("Shape mismatch")
        elif val is None:
            self._val = np.empty(global_shape, dtype=dtype)
        else:
            raise TypeError("unknown source type")

    def _parse_domain(self, domain, val=None):
        if domain is None:
            if isinstance(val, Field):
                domain = val.domain
            elif np.isscalar(val):
                domain = ()
            else:
                raise TypeError("could not infer domain from value")
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

    def _get_axes_tuple(self, things_with_shape):
        i = 0
        axes_list = []
        for thing in things_with_shape:
            nax = len(thing.shape)
            axes_list += [tuple(range(i, i+nax))]
            i += nax
        return tuple(axes_list)

    def _infer_dtype(self, dtype, val):
        if val is None:
            return np.float64 if dtype is None else dtype
        if dtype is None:
            if isinstance(val, Field):
                return val.dtype
            return np.result_type(val)

    # ---Factory methods---

    @classmethod
    def from_random(cls, random_type, domain=None, dtype=None, **kwargs):
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

        Returns
        -------
        out : Field
            The output object.

        See Also
        --------
        power_synthesize


        """

        f = cls(domain=domain, dtype=dtype)
        generator_function = getattr(Random, random_type)
        f.val = generator_function(dtype=f.dtype, shape=f.shape, **kwargs)
        return f

    # ---Powerspectral methods---

    def power_analyze(self, spaces=None, binbounds=None,
                      keep_phase_information=False):
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
        binbounds : array-like *optional*
            Inner bounds of the bins (default : None).
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
            The output object. Its domain is a PowerSpace and it contains
            the power spectrum of 'self's field.

        See Also
        --------
        power_synthesize, PowerSpace

        """

        # check if all spaces in `self.domain` are either harmonic or
        # power_space instances
        for sp in self.domain:
            if not sp.harmonic and not isinstance(sp, PowerSpace):
                raise TypeError(
                    "Field has a space in `domain` which is neither "
                    "harmonic nor a PowerSpace.")

        # check if the `spaces` input is valid
        spaces = utilities.cast_axis_to_tuple(spaces, len(self.domain))
        if spaces is None:
            spaces = list(range(len(self.domain)))

        if len(spaces) == 0:
            raise ValueError("No space for analysis specified.")

        if keep_phase_information:
            parts_val = self._hermitian_decomposition(
                                              val=self.val,
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
                                binbounds=binbounds)
                     for part in parts]

        if keep_phase_information:
            return parts[0] + 1j*parts[1]
        else:
            return parts[0]

    @classmethod
    def _single_power_analyze(cls, work_field, space_index, binbounds):

        if not work_field.domain[space_index].harmonic:
            raise ValueError(
                "The analyzed space must be harmonic.")

        # Create the target PowerSpace instance:
        # If the associated signal-space field was real, we extract the
        # hermitian and anti-hermitian parts of `self` and put them
        # into the real and imaginary parts of the power spectrum.
        # If it was complex, all the power is put into a real power spectrum.

        harmonic_domain = work_field.domain[space_index]
        power_domain = PowerSpace(harmonic_partner=harmonic_domain,
                                  binbounds=binbounds)
        power_spectrum = cls._calculate_power_spectrum(
                                field_val=work_field.val,
                                pdomain=power_domain,
                                axes=work_field.domain_axes[space_index])

        # create the result field and put power_spectrum into it
        result_domain = list(work_field.domain)
        result_domain[space_index] = power_domain

        return Field(domain=result_domain, val=power_spectrum,
                     dtype=power_spectrum.dtype)

    @classmethod
    def _calculate_power_spectrum(cls, field_val, pdomain, axes=None):

        pindex = pdomain.pindex
        if axes is not None:
            pindex = cls._shape_up_pindex(
                            pindex=pindex,
                            target_shape=field_val.shape,
                            axes=axes)

        power_spectrum = utilities.bincount_axis(pindex, weights=field_val,
                                                 axis=axes)
        rho = pdomain.rho
        if axes is not None:
            new_rho_shape = [1, ] * len(power_spectrum.shape)
            new_rho_shape[axes[0]] = len(rho)
            rho = rho.reshape(new_rho_shape)
        power_spectrum /= rho

        return power_spectrum

    @staticmethod
    def _shape_up_pindex(pindex, target_shape, axes):
        semiscaled_local_shape = [1] * len(target_shape)
        for i in range(len(axes)):
            semiscaled_local_shape[axes[i]] = pindex.shape[i]
        semiscaled_local_data = pindex.reshape(semiscaled_local_shape)
        result_obj = np.empty(target_shape, dtype=pindex.dtype)
        result_obj[()] = semiscaled_local_data
        return result_obj

    def power_synthesize(self, spaces=None, real_power=True, real_signal=True,
                         mean=None, std=None):
        """ Yields a sampled field with `self`**2 as its power spectrum.

        This method draws a Gaussian random field in the harmonic partner
        domain of this field's domains, using this field as power spectrum.

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
        result_list = [self.__class__.from_random(
                             'normal',
                             mean=mean,
                             std=std,
                             domain=result_domain,
                             dtype=np.complex)
                       for x in range(1 if real_power else 2)]

        # from now on extract the values from the random fields for further
        # processing without killing the fields.
        # if the signal-space field should be real, hermitianize the field
        # components
        spec = np.sqrt(self.val)
        for power_space_index in spaces:
            spec = self._spec_to_rescaler(spec, power_space_index)

        # apply the rescaler to the random fields
        result_list[0].val *= spec.real
        if not real_power:
            result_list[1].val *= spec.imag

        if real_signal:
            result_list = [Field(i.domain, self._hermitian_decomposition(
                                     i.val,
                                     preserve_gaussian_variance=True)[0])
                           for i in result_list]

        if real_power:
            result = result_list[0]
            if not issubclass(result_list[0].dtype.type,
                              np.complexfloating):
                result = result.real
        else:
            result = result_list[0] + 1j*result_list[1]

        return result

    @staticmethod
    def _hermitian_decomposition(val, preserve_gaussian_variance=False):
        if preserve_gaussian_variance:
            if not issubclass(val.dtype.type, np.complexfloating):
                raise TypeError("complex input field is needed here")
            return (val.real*np.sqrt(2.), val.imag*np.sqrt(2.))
        else:
            return (val.real.copy(), val.imag.copy())

    def _spec_to_rescaler(self, spec, power_space_index):
        power_space = self.domain[power_space_index]

        local_blow_up = [slice(None)]*len(spec.shape)
        # it is important to count from behind, since spec potentially grows
        # with every iteration
        index = self.domain_axes[power_space_index][0]-len(self.shape)
        local_blow_up[index] = power_space.pindex
        # here, the power_spectrum is distributed into the new shape
        return spec[local_blow_up]

    # ---Properties---

    def set_val(self, new_val=None, copy=False):
        """ Sets the field's data object.

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

        if new_val is None:
            pass
        elif isinstance(new_val, Field):
            if self.domain != new_val.domain:
                raise ValueError("Domain mismatch")
            if copy:
                self._val[()] = new_val.val
            else:
                self._val = np.array(new_val.val, dtype=self.dtype, copy=False)
        elif (np.isscalar(new_val)):
            self._val[()] = new_val
        elif isinstance(new_val, np.ndarray):
            if copy:
                self._val[()] = new_val
            else:
                if self.shape != new_val.shape:
                    raise ValueError("Shape mismatch")
                self._val = np.array(new_val, dtype=self.dtype, copy=False)
        else:
            raise TypeError("unknown source type")
        return self

    def get_val(self, copy=False):
        """ Returns the data object associated with this Field.

        Parameters
        ----------
        copy : boolean
            If true, a copy of the Field's underlying data object
            is returned.

        Returns
        -------
        out : numpy.ndarray

        See Also
        --------
        val
        """
        return self._val.copy() if copy else self._val

    @property
    def val(self):
        """ Returns the data object associated with this Field.
        No copy is made.

        Returns
        -------
        out : numpy.ndarray

        See Also
        --------
        get_val
        """
        return self._val

    @val.setter
    def val(self, new_val):
        self.set_val(new_val=new_val, copy=False)

    @property
    def dtype(self):
        return self._val.dtype

    @property
    def shape(self):
        """ Returns the total shape of the Field's data array.

        Returns
        -------
        out : tuple
            The output object. The tuple contains the dimensions of the spaces
            in domain.
       """
        return self._val.shape

    @property
    def dim(self):
        """ Returns the total number of pixel-dimensions the field has.

        Effectively, all values from shape are multiplied.

        Returns
        -------
        out : int
            The dimension of the Field.
        """
        return self._val.size

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

    @property
    def real(self):
        """ The real part of the field (data is not copied).
        """
        return Field(self.domain, self.val.real)

    @property
    def imag(self):
        """ The imaginary part of the field (data is not copied).
        """
        return Field(self.domain, self.val.imag)

    # ---Special unary/binary operations---

    def copy(self, domain=None, dtype=None):
        """ Returns a full copy of the Field.

        If no keyword arguments are given, the returned object will be an
        identical copy of the original Field. By explicit specification one is
        able to define the domain and the dtype of the returned Field.

        Parameters
        ----------
        domain : DomainObject
            The new domain the Field shall have.

        dtype : type
            The new dtype the Field shall have.

        Returns
        -------
        out : Field
            The output object. An identical copy of 'self'.

        See Also
        --------
        copy_empty

        """

        if domain is None:
            domain = self.domain
        return Field(domain=domain, val=self._val, dtype=dtype, copy=True)

    def copy_empty(self, domain=None, dtype=None):
        """ Returns an empty copy of the Field.

        If no keyword arguments are given, the returned object will be an
        identical copy of the original Field. The memory for the data array
        is only allocated but not actively set to any value
        (c.f. numpy.ndarray.copy_empty). By explicit specification one is able
        to change the domain and the dtype of the returned Field.

        Parameters
        ----------
        domain : DomainObject
            The new domain the Field shall have.

        dtype : type
            The new dtype the Field shall have.

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
        if dtype is None:
            dtype = self.dtype
        return Field(domain=domain, dtype=dtype)

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
                # we need at most one copy, the rest can happen in place
                inplace = True

        return Field(self.domain, new_val, self.dtype)

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
        y = self if bare else self.weight(power=1)

        if spaces is None:
            return np.vdot(y.val.flatten(), x.val.flatten())
        else:
            # create a diagonal operator which is capable of taking care of the
            # axes-matching
            from .operators.diagonal_operator import DiagonalOperator
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
            self.imag *= -1
            return self
        else:
            return Field(self.domain, np.conj(self.val), self.dtype)

    # ---General unary/contraction methods---

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        return Field(self.domain, -self.val, self.dtype)

    def __abs__(self):
        return Field(self.domain, np.abs(self.val), self.dtype)

    def _contraction_helper(self, op, spaces):
        if spaces is None:
            return getattr(self.val, op)()
        # build a list of all axes
        spaces = utilities.cast_axis_to_tuple(spaces, len(self.domain))

        axes_list = tuple(self.domain_axes[sp_index] for sp_index in spaces)

        try:
            axes_list = reduce(lambda x, y: x+y, axes_list)
        except TypeError:
            axes_list = ()

        # perform the contraction on the data
        data = self.get_val(copy=False)
        data = getattr(data, op)(axis=axes_list)

        # check if the result is scalar or if a result_field must be constr.
        if np.isscalar(data):
            return data
        else:
            return_domain = tuple(self.domain[i]
                                  for i in range(len(self.domain))
                                  if i not in spaces)

            return Field(domain=return_domain, val=data, copy=False)

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

    def _binary_helper(self, other, op):
        # if other is a field, make sure that the domains match
        if isinstance(other, Field):
            if other.domain != self.domain:
                raise ValueError("domains are incompatible.")
            return Field(self.domain, getattr(self.val, op)(other.val))

        return Field(self.domain, getattr(self.val, op)(other))

    def __add__(self, other):
        return self._binary_helper(other, op='__add__')

    def __radd__(self, other):
        return self._binary_helper(other, op='__radd__')

    def __iadd__(self, other):
        return self._binary_helper(other, op='__iadd__')

    def __sub__(self, other):
        return self._binary_helper(other, op='__sub__')

    def __rsub__(self, other):
        return self._binary_helper(other, op='__rsub__')

    def __isub__(self, other):
        return self._binary_helper(other, op='__isub__')

    def __mul__(self, other):
        return self._binary_helper(other, op='__mul__')

    def __rmul__(self, other):
        return self._binary_helper(other, op='__rmul__')

    def __imul__(self, other):
        return self._binary_helper(other, op='__imul__')

    def __div__(self, other):
        return self._binary_helper(other, op='__div__')

    def __truediv__(self, other):
        return self._binary_helper(other, op='__truediv__')

    def __rdiv__(self, other):
        return self._binary_helper(other, op='__rdiv__')

    def __rtruediv__(self, other):
        return self._binary_helper(other, op='__rtruediv__')

    def __idiv__(self, other):
        return self._binary_helper(other, op='__idiv__')

    def __pow__(self, other):
        return self._binary_helper(other, op='__pow__')

    def __rpow__(self, other):
        return self._binary_helper(other, op='__rpow__')

    def __ipow__(self, other):
        return self._binary_helper(other, op='__ipow__')

    def __lt__(self, other):
        return self._binary_helper(other, op='__lt__')

    def __le__(self, other):
        return self._binary_helper(other, op='__le__')

    def __ne__(self, other):
        if other is None:
            return True
        else:
            return self._binary_helper(other, op='__ne__')

    def __eq__(self, other):
        if other is None:
            return False
        else:
            return self._binary_helper(other, op='__eq__')

    def __ge__(self, other):
        return self._binary_helper(other, op='__ge__')

    def __gt__(self, other):
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
