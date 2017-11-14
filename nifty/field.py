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
from . import nifty_utilities as utilities
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
            self._val = dobj.full(self.domain.shape, dtype=dtype,
                                  fill_value=val)
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
    def full(domain, val, dtype=None):
        if not np.isscalar(val):
            raise TypeError("val must be a scalar")
        return Field(DomainTuple.make(domain), val, dtype)

    @staticmethod
    def ones(domain, dtype=None):
        return Field(DomainTuple.make(domain), 1., dtype)

    @staticmethod
    def zeros(domain, dtype=None):
        return Field(DomainTuple.make(domain), 0., dtype)

    @staticmethod
    def empty(domain, dtype=None):
        return Field(DomainTuple.make(domain), None, dtype)

    @staticmethod
    def full_like(field, val, dtype=None):
        if not isinstance(field, Field):
            raise TypeError("field must be of Field type")
        return Field.full(field.domain, val, dtype)

    @staticmethod
    def zeros_like(field, dtype=None):
        if not isinstance(field, Field):
            raise TypeError("field must be of Field type")
        if dtype is None:
            dtype = field.dtype
        return Field.zeros(field.domain, dtype)

    @staticmethod
    def ones_like(field, dtype=None):
        if not isinstance(field, Field):
            raise TypeError("field must be of Field type")
        if dtype is None:
            dtype = field.dtype
        return Field.ones(field.domain, dtype)

    @staticmethod
    def empty_like(field, dtype=None):
        if not isinstance(field, Field):
            raise TypeError("field must be of Field type")
        if dtype is None:
            dtype = field.dtype
        return Field.empty(field.domain, dtype)

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
        """

        domain = DomainTuple.make(domain)
        return Field(domain=domain,
                     val=dobj.from_random(random_type, dtype=dtype,
                                          shape=domain.shape, **kwargs))

    def fill(self, fill_value):
        self._val.fill(fill_value)

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
                new_shape = np.ones(len(self.shape), dtype=np.int)
                new_shape[self.domain.axes[ind][0]:
                          self.domain.axes[ind][-1]+1] = wgt.shape
                wgt = wgt.reshape(new_shape)
                # FIXME only temporary
                if ind == 0:
                    wgt = dobj.local_data(dobj.from_global_data(wgt))
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
            return fct*dobj.vdot(y.val, x.val)
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
            return_domain = tuple(dom
                                  for i, dom in enumerate(self.domain)
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
        dobj.local_data(self.val)[()] = dobj.local_data(other.val)[()]

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
