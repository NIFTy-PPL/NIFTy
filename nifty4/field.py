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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division
from builtins import range
import numpy as np
from . import utilities
from .domain_tuple import DomainTuple
from functools import reduce
from . import dobj

__all__ = ["Field", "sqrt", "exp", "log", "conjugate"]


class Field(object):
    """ The discrete representation of a continuous field over multiple spaces.

    In NIFTy, Fields are used to store data arrays and carry all the needed
    metainformation (i.e. the domain) for operators to be able to work on them.

    Parameters
    ----------
    domain : None, DomainTuple, tuple of Domain, or Domain

    val : None, Field, data_object, or scalar
        The values the array should contain after init. A scalar input will
        fill the whole array with this scalar. If a data_object is provided,
        its dimensions must match the domain's.

    dtype : type
        A numpy.type. Most common are float and complex.

    Notes
    -----
    If possible, do not invoke the constructor directly, but use one of the
    many convenience functions for Field conatruction!
    """

    def __init__(self, domain=None, val=None, dtype=None, copy=False,
                 locked=False):
        self._domain = self._infer_domain(domain=domain, val=val)

        dtype = self._infer_dtype(dtype=dtype, val=val)
        if isinstance(val, Field):
            if self._domain != val._domain:
                raise ValueError("Domain mismatch")
            self._val = dobj.from_object(val.val, dtype=dtype, copy=copy,
                                         set_locked=locked)
        elif (np.isscalar(val)):
            self._val = dobj.full(self._domain.shape, dtype=dtype,
                                  fill_value=val)
        elif isinstance(val, dobj.data_object):
            if self._domain.shape == val.shape:
                self._val = dobj.from_object(val, dtype=dtype, copy=copy,
                                             set_locked=locked)
            else:
                raise ValueError("Shape mismatch")
        elif val is None:
            self._val = dobj.empty(self._domain.shape, dtype=dtype)
        else:
            raise TypeError("unknown source type")

        if locked:
            dobj.lock(self._val)

    @staticmethod
    def full(domain, val, dtype=None):
        """Creates a Field with a given domain, filled with a constant value.

        Parameters
        ----------
        domain : Domain, tuple of Domain, or DomainTuple
            domain of the new Field
        val : float/complex/int scalar
            fill value. Data type of the field is inferred from val.

        Returns
        -------
        Field
            the newly created field
        """
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
        """Creates a Field from a template, filled with a constant value.

        Parameters
        ----------
        field : Field
            the template field, from which the domain is inferred
        val : float/complex/int scalar
            fill value. Data type of the field is inferred from val.

        Returns
        -------
        Field
            the newly created field
        """
        if not isinstance(field, Field):
            raise TypeError("field must be of Field type")
        return Field.full(field._domain, val, dtype)

    @staticmethod
    def zeros_like(field, dtype=None):
        if not isinstance(field, Field):
            raise TypeError("field must be of Field type")
        if dtype is None:
            dtype = field.dtype
        return Field.zeros(field._domain, dtype)

    @staticmethod
    def ones_like(field, dtype=None):
        if not isinstance(field, Field):
            raise TypeError("field must be of Field type")
        if dtype is None:
            dtype = field.dtype
        return Field.ones(field._domain, dtype)

    @staticmethod
    def empty_like(field, dtype=None):
        if not isinstance(field, Field):
            raise TypeError("field must be of Field type")
        if dtype is None:
            dtype = field.dtype
        return Field.empty(field._domain, dtype)

    @staticmethod
    def from_global_data(domain, arr, sum_up=False):
        """Returns a Field constructed from `domain` and `arr`.

        Parameters
        ----------
        domain : DomainTuple, tuple of Domain, or Domain
            the domain of the new Field
        arr : numpy.ndarray
            The data content to be used for the new Field.
            Its shape must match the shape of `domain`.
        sum_up : bool, optional
            If True, the contents of `arr` are summed up over all MPI tasks
            (if any), and the sum is used as data content.
            If False, the contens of `arr` are used directly, and must be
            identical on all MPI tasks.
        """
        return Field(domain, dobj.from_global_data(arr, sum_up))

    @staticmethod
    def from_local_data(domain, arr):
        domain = DomainTuple.make(domain)
        return Field(domain, dobj.from_local_data(domain.shape, arr))

    def to_global_data(self):
        """Returns an array containing the full data of the field.

        Returns
        -------
        numpy.ndarray : array containing all field entries.
            Its shape is identical to `self.shape`.

        Notes
        -----
        Do not write to the returned array! Depending on whether MPI is
        active or not, this may or may not change the field's data content.
        """
        return dobj.to_global_data(self._val)

    @property
    def local_data(self):
        """numpy.ndarray : locally residing field data

        Returns a handle to the part of the array data residing on the local
        task (or to the entore array if MPI is not active).

        Notes
        -----
        If the field is not locked, the array data can be modified.
        Use with care!
        """
        return dobj.local_data(self._val)

    def cast_domain(self, new_domain):
        """Returns a field with the same data, but a different domain

        Parameters
        ----------
        new_domain : Domain, tuple of Domain, or DomainTuple
            The domain for the returned field. Must be shape-compatible to
            `self`.

        Returns
        -------
        Field
            Field living on `new_domain`, but with the same data as `self`.

        Notes
        -----
        No copy is made. If needed, use an additional copy() invocation.
        """
        return Field(new_domain, self._val)

    @staticmethod
    def _infer_domain(domain, val=None):
        if domain is None:
            if isinstance(val, Field):
                return val._domain
            if np.isscalar(val):
                return DomainTuple.make(())  # empty domain tuple
            raise TypeError("could not infer domain from value")
        return DomainTuple.make(domain)

    @staticmethod
    def _infer_dtype(dtype, val):
        if dtype is not None:
            return dtype
        if val is None:
            raise ValueError("could not infer dtype")
        if isinstance(val, Field):
            return val.dtype
        return np.result_type(val)

    @staticmethod
    def from_random(random_type, domain, dtype=np.float64, **kwargs):
        """ Draws a random field with the given parameters.

        Parameters
        ----------
        random_type : 'pm1', 'normal', or 'uniform'
            The random distribution to use.
        domain : DomainTuple
            The domain of the output random field
        dtype : type
            The datatype of the output random field

        Returns
        -------
        Field
            The newly created Field.
        """
        domain = DomainTuple.make(domain)
        return Field(domain=domain,
                     val=dobj.from_random(random_type, dtype=dtype,
                                          shape=domain.shape, **kwargs))

    def fill(self, fill_value):
        """Fill `self` uniformly with `fill_value`

        Parameters
        ----------
        fill_value: float or complex or int
            The value to fill the field with.
        """
        self._val.fill(fill_value)

    def lock(self):
        """Write-protect the data content of `self`.

        After this call, it will no longer be possible to change the data
        entries of `self`. This is convenient if, for example, a
        DiagonalOperator wants to ensure that its diagonal cannot be modified
        inadvertently, without making copies.

        Notes
        -----
        This will not only prohibit modifications to the entries of `self`, but
        also to the entries of any other Field or numpy array pointing to the
        same data. If an unlocked instance is needed, use copy().

        The fact that there is no `unlock()` method is deliberate.
        """
        dobj.lock(self._val)
        return self

    @property
    def locked(self):
        """bool : True iff the field's data content has been locked"""
        return dobj.locked(self._val)

    @property
    def val(self):
        """dobj.data_object : the data object storing the field's entries

        Notes
        -----
        This property is intended for low-level, internal use only. Do not use
        from outside of NIFTy's core; there should be better alternatives.
        """
        return self._val

    @property
    def dtype(self):
        """type : the data type of the field's entries"""
        return self._val.dtype

    @property
    def domain(self):
        """DomainTuple : the field's domain"""
        return self._domain

    @property
    def shape(self):
        """tuple of int : the concatenated shapes of all sub-domains"""
        return self._domain.shape

    @property
    def size(self):
        """int : total number of pixels in the field"""
        return self._domain.size

    @property
    def real(self):
        """Field : The real part of the field"""
        if not np.issubdtype(self.dtype, np.complexfloating):
            return self
        return Field(self._domain, self.val.real)

    @property
    def imag(self):
        """Field : The imaginary part of the field"""
        if not np.issubdtype(self.dtype, np.complexfloating):
            raise ValueError(".imag called on a non-complex Field")
        return Field(self._domain, self.val.imag)

    def copy(self):
        """ Returns a full copy of the Field.

        The returned object will be an identical copy of the original Field.
        The copy will be writeable, even if `self` was locked.

        Returns
        -------
        Field
            An identical, but unlocked copy of 'self'.
        """
        return Field(val=self, copy=True)

    def locked_copy(self):
        """ Returns a read-only version of the Field.

        If `self` is locked, returns `self`. Otherwise returns a locked copy
        of `self`.

        Returns
        -------
        Field
            A read-only version of `self`.
        """
        return self if self.locked else Field(val=self, copy=True, locked=True)

    def scalar_weight(self, spaces=None):
        """Returns the uniform volume element for a sub-domain of `self`.

        Parameters
        ----------
        spaces : int, tuple of int or None
            indices of the sub-domains of the field's domain to be considered.
            If `None`, the entire domain is used.

        Returns
        -------
        float or None
            if the requested sub-domain has a uniform volume element, it is
            returned. Otherwise, `None` is returned.
        """
        if np.isscalar(spaces):
            return self._domain[spaces].scalar_dvol

        if spaces is None:
            spaces = range(len(self._domain))
        res = 1.
        for i in spaces:
            tmp = self._domain[i].scalar_dvol
            if tmp is None:
                return None
            res *= tmp
        return res

    def total_volume(self, spaces=None):
        """Returns the total volume of a sub-domain of `self`.

        Parameters
        ----------
        spaces : int, tuple of int or None
            indices of the sub-domains of the field's domain to be considered.
            If `None`, the entire domain is used.

        Returns
        -------
        float
            the total volume of the requested sub-domain.
        """
        if np.isscalar(spaces):
            return self._domain[spaces].total_volume

        if spaces is None:
            spaces = range(len(self._domain))
        res = 1.
        for i in spaces:
            res *= self._domain[i].total_volume
        return res

    def weight(self, power=1, spaces=None, out=None):
        """ Weights the pixels of `self` with their invidual pixel-volume.

        Parameters
        ----------
        power : number
            The pixels get weighted with the volume-factor**power.

        spaces : None, int or tuple of int
            Determines on which sub-domain the operation takes place.
            If None, the entire domain is used.

        out : Field or None
            if not None, the result is returned in a new Field
            otherwise the contents of "out" are overwritten with the result.
            "out" may be identical to "self"!

        Returns
        -------
        Field
            The weighted field.
        """
        if out is None:
            out = self.copy()
        else:
            if out is not self:
                out.copy_content_from(self)

        spaces = utilities.parse_spaces(spaces, len(self._domain))

        fct = 1.
        for ind in spaces:
            wgt = self._domain[ind].dvol
            if np.isscalar(wgt):
                fct *= wgt
            else:
                new_shape = np.ones(len(self.shape), dtype=np.int)
                new_shape[self._domain.axes[ind][0]:
                          self._domain.axes[ind][-1]+1] = wgt.shape
                wgt = wgt.reshape(new_shape)
                if dobj.distaxis(self._val) >= 0 and ind == 0:
                    # we need to distribute the weights along axis 0
                    wgt = dobj.local_data(dobj.from_global_data(wgt))
                out.local_data[()] *= wgt**power
        fct = fct**power
        if fct != 1.:
            out *= fct

        return out

    def vdot(self, x=None, spaces=None):
        """ Computes the dot product of 'self' with x.

        Parameters
        ----------
        x : Field
            x must live on the same domain as `self`.

        spaces : None, int or tuple of int (default: None)
            The dot product is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        float, complex, either scalar (for full dot products)
                              or Field (for partial dot products)
        """
        if not isinstance(x, Field):
            raise ValueError("The dot-partner must be an instance of " +
                             "the NIFTy field class")

        if x._domain != self._domain:
            raise ValueError("Domain mismatch")

        ndom = len(self._domain)
        spaces = utilities.parse_spaces(spaces, ndom)

        if len(spaces) == ndom:
            return dobj.vdot(self.val, x.val)
        # If we arrive here, we have to do a partial dot product.
        # For the moment, do this the explicit, non-optimized way
        return (self.conjugate()*x).sum(spaces=spaces)

    def norm(self):
        """ Computes the L2-norm of the field values.

        Returns
        -------
        float
            The L2-norm of the field values.
        """
        return np.sqrt(np.abs(self.vdot(x=self)))

    def conjugate(self):
        """ Returns the complex conjugate of the field.

        Returns
        -------
        Field
            The complex conjugated field.
        """
        return Field(self._domain, self.val.conjugate())

    # ---General unary/contraction methods---

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        return Field(self._domain, -self.val)

    def __abs__(self):
        return Field(self._domain, dobj.abs(self.val))

    def _contraction_helper(self, op, spaces):
        if spaces is None:
            return getattr(self.val, op)()

        spaces = utilities.parse_spaces(spaces, len(self._domain))

        axes_list = tuple(self._domain.axes[sp_index] for sp_index in spaces)

        if len(axes_list) > 0:
            axes_list = reduce(lambda x, y: x+y, axes_list)

        # perform the contraction on the data
        data = getattr(self.val, op)(axis=axes_list)

        # check if the result is scalar or if a result_field must be constr.
        if np.isscalar(data):
            return data
        else:
            return_domain = tuple(dom
                                  for i, dom in enumerate(self._domain)
                                  if i not in spaces)

            return Field(domain=return_domain, val=data, copy=False)

    def sum(self, spaces=None):
        """Sums up over the sub-domains given by `spaces`.

        Parameters
        ----------
        spaces : None, int or tuple of int (default: None)
            The summation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field or scalar
            The result of the summation. If it is carried out over the entire
            domain, this is a scalar, otherwise a Field.
        """
        return self._contraction_helper('sum', spaces)

    def integrate(self, spaces=None):
        """Integrates over the sub-domains given by `spaces`.

        Integration is performed by summing over `self` multiplied by its
        volume factors.

        Parameters
        ----------
        spaces : None, int or tuple of int (default: None)
            The summation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field or scalar
            The result of the integration. If it is carried out over the
            entire domain, this is a scalar, otherwise a Field.
        """
        swgt = self.scalar_weight(spaces)
        if swgt is not None:
            res = self.sum(spaces)
            res *= swgt
            return res
        tmp = self.weight(1, spaces=spaces)
        return tmp.sum(spaces)

    def prod(self, spaces=None):
        """Computes the product over the sub-domains given by `spaces`.

        Parameters
        ----------
        spaces : None, int or tuple of int (default: None)
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field or scalar
            The result of the product. If it is carried out over the entire
            domain, this is a scalar, otherwise a Field.
        """
        return self._contraction_helper('prod', spaces)

    def all(self, spaces=None):
        return self._contraction_helper('all', spaces)

    def any(self, spaces=None):
        return self._contraction_helper('any', spaces)

    def min(self, spaces=None):
        """Determines the minimum over the sub-domains given by `spaces`.

        Parameters
        ----------
        spaces : None, int or tuple of int (default: None)
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field or scalar
            The result of the operation. If it is carried out over the entire
            domain, this is a scalar, otherwise a Field.
        """
        return self._contraction_helper('min', spaces)

    def max(self, spaces=None):
        """Determines the maximum over the sub-domains given by `spaces`.

        Parameters
        ----------
        spaces : None, int or tuple of int (default: None)
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field or scalar
            The result of the operation. If it is carried out over the entire
            domain, this is a scalar, otherwise a Field.
        """
        return self._contraction_helper('max', spaces)

    def mean(self, spaces=None):
        """Determines the mean over the sub-domains given by `spaces`.

        ``x.mean(spaces)`` is equivalent to
        ``x.integrate(spaces)/x.total_volume(spaces)``.

        Parameters
        ----------
        spaces : None, int or tuple of int (default: None)
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field or scalar
            The result of the operation. If it is carried out over the entire
            domain, this is a scalar, otherwise a Field.
        """
        if self.scalar_weight(spaces) is not None:
            return self._contraction_helper('mean', spaces)
        # MR FIXME: not very efficient
        tmp = self.weight(1)
        return tmp.sum(spaces)*(1./tmp.total_volume(spaces))

    def var(self, spaces=None):
        """Determines the variance over the sub-domains given by `spaces`.

        Parameters
        ----------
        spaces : None, int or tuple of int (default: None)
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field or scalar
            The result of the operation. If it is carried out over the entire
            domain, this is a scalar, otherwise a Field.
        """
        if self.scalar_weight(spaces) is not None:
            return self._contraction_helper('var', spaces)
        # MR FIXME: not very efficient or accurate
        m1 = self.mean(spaces)
        if np.issubdtype(self.dtype, np.complexfloating):
            sq = abs(self)**2
            m1 = abs(m1)**2
        else:
            sq = self**2
            m1 **= 2
        return sq.mean(spaces) - m1

    def std(self, spaces=None):
        """Determines the standard deviation over the sub-domains given by
        `spaces`.

        ``x.std(spaces)`` is equivalent to ``sqrt(x.var(spaces))``.

        Parameters
        ----------
        spaces : None, int or tuple of int (default: None)
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field or scalar
            The result of the operation. If it is carried out over the entire
            domain, this is a scalar, otherwise a Field.
        """
        if self.scalar_weight(spaces) is not None:
            return self._contraction_helper('std', spaces)
        return sqrt(self.var(spaces))

    def copy_content_from(self, other):
        if not isinstance(other, Field):
            raise TypeError("argument must be a Field")
        if other._domain != self._domain:
            raise ValueError("domains are incompatible.")
        self.local_data[()] = other.local_data[()]

    def _binary_helper(self, other, op):
        # if other is a field, make sure that the domains match
        if isinstance(other, Field):
            if other._domain != self._domain:
                raise ValueError("domains are incompatible.")
            tval = getattr(self.val, op)(other.val)
            return self if tval is self.val else Field(self._domain, tval)

        if np.isscalar(other) or isinstance(other, dobj.data_object):
            tval = getattr(self.val, op)(other)
            return self if tval is self.val else Field(self._domain, tval)

        return NotImplemented

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
        return self._binary_helper(other, op='__ne__')

    def __eq__(self, other):
        return self._binary_helper(other, op='__eq__')

    def __ge__(self, other):
        return self._binary_helper(other, op='__ge__')

    def __gt__(self, other):
        return self._binary_helper(other, op='__gt__')

    def __repr__(self):
        return "<nifty4.Field>"

    def __str__(self):
        return "nifty4.Field instance\n- domain      = " + \
               self._domain.__str__() + \
               "\n- val         = " + repr(self.val)


# Arithmetic functions working on Fields

def _math_helper(x, function, out):
    if not isinstance(x, Field):
        raise TypeError("This function only accepts Field objects.")
    if out is not None:
        if not isinstance(out, Field) or x._domain != out._domain:
            raise ValueError("Bad 'out' argument")
        function(x.val, out=out.val)
        return out
    else:
        return Field(domain=x._domain, val=function(x.val))


def sqrt(x, out=None):
    return _math_helper(x, dobj.sqrt, out)


def exp(x, out=None):
    return _math_helper(x, dobj.exp, out)


def log(x, out=None):
    return _math_helper(x, dobj.log, out)


def tanh(x, out=None):
    return _math_helper(x, dobj.tanh, out)


def conjugate(x, out=None):
    return _math_helper(x, dobj.conjugate, out)
