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

from __future__ import division, print_function
from builtins import range
import numpy as np
from .spaces.power_space import PowerSpace
from . import nifty_utilities as utilities
from .random import Random
from .domain_tuple import DomainTuple
from functools import reduce
from . import dobj


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

    domain : DomainTuple
        See Parameters.
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

        dtype = self._infer_dtype(dtype=dtype, val=val)
        if isinstance(val, Field):
            if self.domain != val.domain:
                raise ValueError("Domain mismatch")
            self._val = dobj.from_object(val.val, dtype=dtype, copy=copy)
        elif (np.isscalar(val)):
            self._val = dobj.full(self.domain.shape, dtype=dtype, fill_value=val)
        elif isinstance(val, dobj.data_object):
            if self.domain.shape == val.shape:
                self._val = dobj.from_object(val, dtype=dtype, copy=copy)
            else:
                raise ValueError("Shape mismatch")
        elif val is None:
            self._val = dobj.empty(self.domain.shape, dtype=dtype)
        else:
            raise TypeError("unknown source type")

    @staticmethod
    def _parse_domain(domain, val=None):
        if domain is None:
            if isinstance(val, Field):
                return val.domain
            if np.isscalar(val):
                return DomainTuple.make(())  # empty domain tuple
            raise TypeError("could not infer domain from value")
        return DomainTuple.make(domain)

    # MR: this needs some rethinking ... do we need to have at least float64?
    @staticmethod
    def _infer_dtype(dtype, val):
        if val is None or dtype is not None:
            return np.result_type(dtype, np.float64)
        if isinstance(val, Field):
            return val.dtype
        return np.result_type(val, np.float64)

    # ---Factory methods---

    @staticmethod
    def from_random(random_type, domain, dtype=np.float64, **kwargs):
        """ Draws a random field with the given parameters.

        Parameters
        ----------
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

        domain = DomainTuple.make(domain)
        generator_function = getattr(Random, random_type)
        return Field(domain=domain, val=generator_function(dtype=dtype,
                     shape=domain.shape, **kwargs))

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
            The subspace for which the powerspectrum shall be computed.
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
        TypeError
            Raised if any of the input field's domains is not harmonic

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
                print("WARNING: Field has a space in `domain` which is "
                      "neither harmonic nor a PowerSpace.")

        # check if the `spaces` input is valid
        if spaces is None:
            spaces = range(len(self.domain))
        else:
            spaces = utilities.cast_iseq_to_tuple(spaces)

        if len(spaces) == 0:
            raise ValueError("No space for analysis specified.")

        if keep_phase_information:
            parts = [self.real*self.real, self.imag*self.imag]
        else:
            parts = [self.real*self.real + self.imag*self.imag]

        parts = [ part.weight(1,spaces) for part in parts ]
        for space_index in spaces:
            parts = [self._single_power_analyze(field=part,
                                                idx=space_index,
                                                binbounds=binbounds)
                     for part in parts]
        parts = [ part.weight(-1,spaces) for part in parts ]

        return parts[0] + 1j*parts[1] if keep_phase_information else parts[0]

    @staticmethod
    def _single_power_analyze(field, idx, binbounds):
        power_domain = PowerSpace(field.domain[idx], binbounds)
        pindex = power_domain.pindex
        axes = field.domain.axes[idx]
        new_pindex_shape = [1] * len(field.shape)
        for i, ax in enumerate(axes):
            new_pindex_shape[ax] = pindex.shape[i]
        pindex = np.broadcast_to(pindex.reshape(new_pindex_shape), field.shape)

        power_spectrum = dobj.bincount_axis(pindex, weights=field.val,
                                            axis=axes)
        result_domain = list(field.domain)
        result_domain[idx] = power_domain
        return Field(result_domain, power_spectrum)

    def _compute_spec(self, spaces):
        if spaces is None:
            spaces = range(len(self.domain))
        else:
            spaces = utilities.cast_iseq_to_tuple(spaces)

        # create the result domain
        result_domain = list(self.domain)
        for i in spaces:
            if not isinstance(self.domain[i], PowerSpace):
                raise ValueError("A PowerSpace is needed for field "
                                 "synthetization.")
            result_domain[i] = self.domain[i].harmonic_partner

        spec = dobj.sqrt(self.val)
        for i in spaces:
            power_space = self.domain[i]
            local_blow_up = [slice(None)]*len(spec.shape)
            # it is important to count from behind, since spec potentially
            # grows with every iteration
            index = self.domain.axes[i][0]-len(self.shape)
            local_blow_up[index] = power_space.pindex
            # here, the power_spectrum is distributed into the new shape
            spec = spec[local_blow_up]
        return Field(result_domain, val=spec)

    def power_synthesize(self, spaces=None, real_power=True, real_signal=True):
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

        spec = self._compute_spec(spaces)

        # create random samples: one or two, depending on whether the
        # power spectrum is real or complex
        result = [self.from_random('normal', mean=0., std=1.,
                                   domain=spec.domain,
                                   dtype=np.float if real_signal
                                   else np.complex)
                  for x in range(1 if real_power else 2)]

        # MR: dummy call - will be removed soon
        if real_signal:
            self.from_random('normal', mean=0., std=1.,
                             domain=spec.domain, dtype=np.float)

        # apply the rescaler to the random fields
        result[0] *= spec.real
        if not real_power:
            result[1] *= spec.imag

        return result[0] if real_power else result[0] + 1j*result[1]

    def power_synthesize_special(self, spaces=None):
        spec = self._compute_spec(spaces)

        # MR: dummy call - will be removed soon
        self.from_random('normal', mean=0., std=1.,
                         domain=spec.domain, dtype=np.complex)

        return spec.real

    # ---Properties---

    @property
    def val(self):
        """ Returns the data object associated with this Field.
        No copy is made.

        Returns
        -------
        out : numpy.ndarray
        """
        return self._val

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
        return self.domain.shape

    @property
    def dim(self):
        """ Returns the total number of pixel-dimensions the field has.

        Effectively, all values from shape are multiplied.

        Returns
        -------
        out : int
            The dimension of the Field.
        """
        return self.domain.dim

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

    def copy(self):
        """ Returns a full copy of the Field.

        The returned object will be an identical copy of the original Field.

        Returns
        -------
        out : Field
            The output object. An identical copy of 'self'.
        """
        return Field(val=self, copy=True)

    def scalar_weight(self, spaces=None):
        if np.isscalar(spaces):
            return self.domain[spaces].scalar_dvol()

        if spaces is None:
            spaces = range(len(self.domain))
        res = 1.
        for i in spaces:
            tmp = self.domain[i].scalar_dvol()
            if tmp is None:
                return None
            res *= tmp
        return res

    def weight(self, power=1, spaces=None, out=None):
        """ Weights the pixels of `self` with their invidual pixel-volume.

        Parameters
        ----------
        power : number
            The pixels get weighted with the volume-factor**power.

        spaces : tuple of ints
            Determines on which subspace the operation takes place.

        out : Field or None
            if not None, the result is returned in a new Field
            otherwise the contents of "out" are overwritten with the result.
            "out" may be identical to "self"!

        Returns
        -------
        out : Field
            The weighted field.

        """
        if out is None:
            out = self.copy()
        else:
            if out is not self:
                out.copy_content_from(self)

        if spaces is None:
            spaces = range(len(self.domain))
        else:
            spaces = utilities.cast_iseq_to_tuple(spaces)

        fct = 1.
        for ind in spaces:
            wgt = self.domain[ind].dvol()
            if np.isscalar(wgt):
                fct *= wgt
            else:
                new_shape = dobj.ones(len(self.shape), dtype=np.int)
                new_shape[self.domain.axes[ind][0]:
                          self.domain.axes[ind][-1]+1] = wgt.shape
                wgt = wgt.reshape(new_shape)
                out *= wgt**power
        fct = fct**power
        if fct != 1.:
            out *= fct

        return out

    def vdot(self, x=None, spaces=None):
        """ Computes the volume-factor-aware dot product of 'self' with x.

        Parameters
        ----------
        x : Field
            The domain of x must contain `self.domain`

        spaces : tuple of ints
            If the domain of `self` and `x` are not the same, `spaces` defines
            which domains of `x` are mapped to those of `self`.

        Returns
        -------
        out : float, complex

        """
        if not isinstance(x, Field):
            raise ValueError("The dot-partner must be an instance of " +
                             "the NIFTy field class")

        # Compute the dot respecting the fact of discrete/continuous spaces
        tmp = self.scalar_weight(spaces)
        if tmp is None:
            fct = 1.
            y = self.weight(power=1)
        else:
            y = self
            fct = tmp

        if spaces is None:
            return fct*dobj.vdot(y.val.ravel(), x.val.ravel())
        else:
            spaces = utilities.cast_iseq_to_tuple(spaces)
            active_axes = []
            for i in spaces:
                active_axes += self.domain.axes[i]
            res = 0.
            for sl in utilities.get_slice_list(self.shape, active_axes):
                res += dobj.vdot(y.val, x.val[sl])
            return res*fct

    def norm(self):
        """ Computes the L2-norm of the field values.

        Returns
        -------
        norm : float
            The L2-norm of the field values.

        """
        return np.sqrt(np.abs(self.vdot(x=self)))

    def conjugate(self):
        """ Returns the complex conjugate of the field.

        Returns
        -------
        cc : field
            The complex conjugated field.

        """
        return Field(self.domain, self.val.conjugate(), self.dtype)

    # ---General unary/contraction methods---

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        return Field(self.domain, -self.val, self.dtype)

    def __abs__(self):
        return Field(self.domain, dobj.abs(self.val), self.dtype)

    def _contraction_helper(self, op, spaces):
        if spaces is None:
            return getattr(self.val, op)()
        else:
            spaces = utilities.cast_iseq_to_tuple(spaces)

        axes_list = tuple(self.domain.axes[sp_index] for sp_index in spaces)

        if len(axes_list) > 0:
            axes_list = reduce(lambda x, y: x+y, axes_list)

        # perform the contraction on the data
        data = getattr(self.val, op)(axis=axes_list)

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

    def integrate(self, spaces=None):
        tmp = self.weight(1, spaces=spaces)
        return tmp.sum(spaces)

    def prod(self, spaces=None):
        return self._contraction_helper('prod', spaces)

    def all(self, spaces=None):
        return self._contraction_helper('all', spaces)

    def any(self, spaces=None):
        return self._contraction_helper('any', spaces)

    def min(self, spaces=None):
        return self._contraction_helper('min', spaces)

    def max(self, spaces=None):
        return self._contraction_helper('max', spaces)

    def mean(self, spaces=None):
        return self._contraction_helper('mean', spaces)

    def var(self, spaces=None):
        return self._contraction_helper('var', spaces)

    def std(self, spaces=None):
        return self._contraction_helper('std', spaces)

    def copy_content_from(self, other):
        if not isinstance(other, Field):
            raise TypeError("argument must be a Field")
        if other.domain != self.domain:
            raise ValueError("domains are incompatible.")
        self.val[()] = other.val

    # ---General binary methods---

    def _binary_helper(self, other, op):
        # if other is a field, make sure that the domains match
        if isinstance(other, Field):
            if other.domain != self.domain:
                raise ValueError("domains are incompatible.")
            tval = getattr(self.val, op)(other.val)
            return self if tval is self.val else Field(self.domain, tval)

        tval = getattr(self.val, op)(other)
        return self if tval is self.val else Field(self.domain, tval)

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
        return "<nifty2go.Field>"

    def __str__(self):
        minmax = [self.min(), self.max()]
        mean = self.mean()
        return "nifty2go.Field instance\n- domain      = " + \
               repr(self.domain) + \
               "\n- val         = " + repr(self.val) + \
               "\n  - min.,max. = " + str(minmax) + \
               "\n  - mean = " + str(mean)
