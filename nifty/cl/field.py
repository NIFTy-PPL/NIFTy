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
# Copyright(C) 2013-2020 Max-Planck-Society
# Copyright(C) 2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce

import numpy as np

from . import utilities
from .any_array import AnyArray
from .domain_tuple import DomainTuple
from .operators.operator import Operator


class Field(Operator):
    """The discrete representation of a continuous field over multiple spaces.

    Stores data arrays and carries all the needed meta-information (i.e. the
    domain) for operators to be able to operate on them.

    Parameters
    ----------
    domain : DomainTuple
        The domain of the new Field.
    val : AnyArray and all allowed wrappees of AnyArray (e.g. np.ndarray, cupy.ndarray, scalars)
        This object's shape must match the domain shape
        After construction, the object will no longer be writeable!

    Notes
    -----
    If possible, do not invoke the constructor directly, but use one of the
    many convenience functions for instantiation!
    """

    _scalar_dom = DomainTuple.scalar_domain()

    def __init__(self, domain, val):
        if not isinstance(domain, DomainTuple):
            raise TypeError("domain must be of type DomainTuple")
        val = AnyArray(val)
        val.lock()
        if domain.shape != val.shape:
            raise ValueError(f"shape mismatch between val and domain\n{domain.shape}\n{val.shape}")
        self._domain = domain
        self._val = val

    @staticmethod
    def scalar(val, device_id=-1):
        return Field(Field._scalar_dom, AnyArray(val).at(device_id))

    # prevent implicit conversion to bool
    def __nonzero__(self):
        raise TypeError("Field does not support implicit conversion to bool")

    def __bool__(self):
        raise TypeError("Field does not support implicit conversion to bool")

    @staticmethod
    def full(domain, val, device_id=-1):
        """Creates a Field with a given domain, filled with a constant value.

        Parameters
        ----------
        domain : Domain, tuple of Domain, or DomainTuple
            Domain of the new Field.
        val : float/complex/int scalar
            Fill value. Data type of the field is inferred from val.
        device_id : int
            The device on which the Field shall be allocated. -1 corresponds to
            host (cpu), 0 is the first device (gpu), 1 the second and so on.
            Default: -1.

        Returns
        -------
        Field
            The newly created Field.
        """
        domain = DomainTuple.make(domain)
        return Field(domain, AnyArray.full(domain.shape, val, device_id))

    @staticmethod
    def from_raw(domain, arr):
        """Returns a Field constructed from `domain` and `arr`.

        Parameters
        ----------
        domain : DomainTuple, tuple of Domain, or Domain
            The domain of the new Field.
        arr : AnyArray
            The data content to be used for the new Field.
            Its shape must match the shape of `domain`.
        """
        domain = DomainTuple.make(domain)
        if np.isscalar(arr):
            arr = np.broadcast_to(arr, domain.shape)
        return Field(domain, AnyArray(arr))

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
            Field defined on `new_domain`, but with the same data as `self`.
        """
        return Field(DomainTuple.make(new_domain), self._val)

    @staticmethod
    def from_random(domain, random_type='normal', dtype=np.float64, device_id=-1, **kwargs):
        """Draws a random field with the given parameters.

        Parameters
        ----------
        random_type : 'pm1', 'normal', or 'uniform'
            The random distribution to use.
        domain : DomainTuple
            The domain of the output random Field.
        dtype : type
            The datatype of the output random Field.
            If the datatype is complex, each real and imaginary part
            have variance 1
        device_id : int
            Device id on which the Field shall be located. Note that Fields are
            always generated on the host and then potentially transferred to
            device. Default: -1

        Returns
        -------
        Field
            The newly created Field.
        """
        from .random import Random
        domain = DomainTuple.make(domain)
        generator_function = getattr(Random, random_type)
        arr = generator_function(dtype=dtype, shape=domain.shape, **kwargs)
        return Field(domain, AnyArray(arr).at(device_id, check_fail=False))

    @property
    def val(self):
        """AnyArray : the array storing the field's entries.

        Notes
        -----
        The returned array is read-only.
        """
        return self._val

    def val_rw(self):
        """AnyArray : a copy of the array storing the field's entries.
        """
        return self._val.copy()

    @property
    def raw(self):
        """ndarray : the raw array (numpy.ndarray or cupy.ndarray) storing the field's entries.

        Notes
        -----
        The returned array is read-only.
        """
        return self._val._val

    def asnumpy(self):
        """np.ndarray : the array storing the field's entries.

        Notes
        -----
        The returned array is read-only.
        """
        return self._val.asnumpy()

    def asnumpy_rw(self):
        """np.ndarray : a copy of the array storing the field's entries.
        """
        return self._val.asnumpy().copy()

    def at(self, device_id, *, check_fail=True):
        """Field : a Field stored on a potentially different devicethe array storing the field's entries.

        Parameters
        ----------
        device_id : int or dict
            The device on which the Field shall be allocated. -1 corresponds to
            host (cpu), 0 is the first device (gpu), 1 the second and so on.
            Default: -1.
        """
        if self._val.device_id == device_id:
            return self
        return Field(self._domain, self._val.at(device_id, check_fail=check_fail))

    @property
    def device_id(self):
        return self._val.device_id

    @property
    def dtype(self):
        """type : the data type of the field's entries"""
        return self._val.dtype

    def astype(self, dtype):
        return Field(self._domain, self._val.astype(dtype))

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
        if utilities.iscomplextype(self.dtype):
            return Field(self._domain, self._val.real)
        return self

    @property
    def imag(self):
        """Field : The imaginary part of the field"""
        if not utilities.iscomplextype(self.dtype):
            raise ValueError(".imag called on a non-complex Field")
        return Field(self._domain, self._val.imag)

    def scalar_weight(self, spaces=None):
        """Returns the uniform volume element for a sub-domain of `self`.

        Parameters
        ----------
        spaces : int, tuple of int or None
            Indices of the sub-domains of the field's domain to be considered.
            If `None`, the entire domain is used.

        Returns
        -------
        float or None
            If the requested sub-domain has a uniform volume element, it is
            returned. Otherwise, `None` is returned.
        """
        return self._domain.scalar_weight(spaces)

    def total_volume(self, spaces=None):
        """Returns the total volume of the field's domain or of a subspace of it.

        Parameters
        ----------
        spaces : int, tuple of int or None
            Indices of the sub-domains of the field's domain to be considered.
            If `None`, the total volume of the whole domain is returned.

        Returns
        -------
        float
            the total volume of the requested (sub-)domain.
        """
        return self._domain.total_volume(spaces)

    def weight(self, power=1, spaces=None):
        """Weights the pixels of `self` with their invidual pixel volumes.

        Parameters
        ----------
        power : number
            The pixel values get multiplied with their volume-factor**power.

        spaces : None, int or tuple of int
            Determines on which sub-domain the operation takes place.
            If None, the entire domain is used.

        Returns
        -------
        Field
            The weighted field on the same device as `self`.
        """
        aout = self.val.copy()

        spaces = utilities.parse_spaces(spaces, len(self._domain))

        fct = 1.
        for ind in spaces:
            wgt = self._domain[ind].dvol
            if np.isscalar(wgt):
                fct *= wgt
            else:
                wgt = AnyArray(wgt).at(self.device_id, check_fail=False)
                new_shape = np.ones(len(self.shape), dtype=np.int64)
                new_shape[self._domain.axes[ind][0]:
                          self._domain.axes[ind][-1]+1] = wgt.shape
                wgt = wgt.reshape(new_shape)
                aout *= wgt**power
        fct = fct**power
        if fct != 1.:
            aout *= fct

        return Field(self._domain, AnyArray(aout).at(self._val.device_id))

    def outer(self, x):
        """Computes the outer product of 'self' with x.

        Parameters
        ----------
        x : :class:`nifty.cl.field.Field`

        Returns
        -------
        Field
            Defined on the product space of self.domain and x.domain.
        """
        if not isinstance(x, Field):
            raise TypeError("The multiplier must be an instance of " +
                            "the Field class")
        from .operators.outer_product_operator import OuterProduct
        return OuterProduct(x.domain, self)(x)

    def vdot(self, x, spaces=None):
        """Computes the dot product of 'self' with x.

        Parameters
        ----------
        x : :class:`nifty.cl.field.Field`
            x must be defined on the same domain as `self`.

        spaces : None, int or tuple of int
            The dot product is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.
            Default: None.

        Returns
        -------
        float, complex, either scalar (for full dot products) or Field (for partial dot products).
        """
        if not isinstance(x, Field):
            raise TypeError("The dot-partner must be an instance of " +
                            "the Field class")

        utilities.check_object_identity(x._domain, self._domain)

        ndom = len(self._domain)
        spaces = utilities.parse_spaces(spaces, ndom)

        if len(spaces) == ndom:
            return Field.scalar(self._val.vdot(x._val), device_id=self.device_id)
        # If we arrive here, we have to do a partial dot product.
        # For the moment, do this the explicit, non-optimized way
        return (self.conjugate()*x).sum(spaces=spaces)

    def s_vdot(self, x):
        """Computes the dot product of 'self' with x.

        Parameters
        ----------
        x : :class:`nifty.cl.field.Field`
            x must be defined on the same domain as `self`.

        Returns
        -------
        float or complex
            The dot product
        """
        if not isinstance(x, Field):
            raise TypeError("The dot-partner must be an instance of " +
                            "the Field class")

        utilities.check_object_identity(x._domain, self._domain)

        return self._val.vdot(x._val)

    def norm(self, ord=2):
        """Computes the L2-norm of the field values.

        Parameters
        ----------
        ord : int
            Accepted values: 1, 2, ..., np.inf. Default: 2.

        Returns
        -------
        float
            The L2-norm of the field values.
        """
        return self._val.norm(ord=ord)

    def conjugate(self):
        """Returns the complex conjugate of the field.

        Returns
        -------
        Field
            The complex conjugated field.
        """
        if utilities.iscomplextype(self._val.dtype):
            return Field(self._domain, self._val.conjugate())
        return self

    # ---General unary/contraction methods---

    def __pos__(self):
        return self

    def __neg__(self):
        return Field(self._domain, -self._val)

    def __abs__(self):
        return Field(self._domain, abs(self._val))

    def _contraction_helper(self, op, spaces):
        if spaces is None:
            res = Field.scalar(getattr(self._val, op)())
            return res.at(self.device_id, check_fail=False)

        spaces = utilities.parse_spaces(spaces, len(self._domain))

        axes_list = tuple(self._domain.axes[sp_index] for sp_index in spaces)

        if len(axes_list) > 0:
            axes_list = reduce(lambda x, y: x+y, axes_list)

        # perform the contraction on the data
        data = getattr(self._val, op)(axis=axes_list)

        # check if the result is scalar or if a result_field must be constr.
        if np.isscalar(data):
            return Field.scalar(data).at(self.device_id, check_fail=False)
        else:
            return_domain = tuple(dom
                                  for i, dom in enumerate(self._domain)
                                  if i not in spaces)

            res = Field(DomainTuple.make(return_domain), data)
            assert res.device_id == self.device_id
            return res

    def scale(self, factor):
        if factor == 1:
            return self
        return factor*self

    def sum(self, spaces=None):
        """Sums up over the sub-domains given by `spaces`.

        Parameters
        ----------
        spaces : None, int or tuple of int
            The summation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field
            The result of the summation.
        """
        return self._contraction_helper('sum', spaces)

    def s_sum(self):
        """Returns the sum over all entries

        Returns
        -------
        scalar
            The result of the summation.
        """
        return self._val.sum()

    def integrate(self, spaces=None):
        """Integrates over the sub-domains given by `spaces`.

        Integration is performed by summing over `self` multiplied by its
        volume factors.

        Parameters
        ----------
        spaces : None, int or tuple of int
            The summation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field
            The result of the integration.
        """
        swgt = self.scalar_weight(spaces)
        if swgt is not None:
            res = self.sum(spaces)
            res = res*swgt
            return res
        tmp = self.weight(1, spaces=spaces)
        return tmp.sum(spaces)

    def s_integrate(self):
        """Integrates over the Field.

        Integration is performed by summing over `self` multiplied by its
        volume factors.

        Returns
        -------
        Scalar
            The result of the integration.
        """
        swgt = self.scalar_weight()
        if swgt is not None:
            return self.s_sum()*swgt
        tmp = self.weight(1)
        return tmp.s_sum()

    def prod(self, spaces=None):
        """Computes the product over the sub-domains given by `spaces`.

        Parameters
        ----------
        spaces : None, int or tuple of int
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.
            Default: None.

        Returns
        -------
        Field
            The result of the product.
        """
        return self._contraction_helper('prod', spaces)

    def s_prod(self):
        return self._val.prod()

    def all(self, spaces=None):
        return self._contraction_helper('all', spaces)

    def s_all(self):
        return self._val.all()

    def any(self, spaces=None):
        return self._contraction_helper('any', spaces)

    def s_any(self):
        return self._val.any()

#     def min(self, spaces=None):
#         """Determines the minimum over the sub-domains given by `spaces`.
#
#         Parameters
#         ----------
#         spaces : None, int or tuple of int (default: None)
#             The operation is only carried out over the sub-domains in this
#             tuple. If None, it is carried out over all sub-domains.
#
#         Returns
#         -------
#         Field
#             The result of the operation.
#         """
#         return self._contraction_helper('min', spaces)
#
#     def max(self, spaces=None):
#         """Determines the maximum over the sub-domains given by `spaces`.
#
#         Parameters
#         ----------
#         spaces : None, int or tuple of int (default: None)
#             The operation is only carried out over the sub-domains in this
#             tuple. If None, it is carried out over all sub-domains.
#
#         Returns
#         -------
#         Field
#             The result of the operation.
#         """
#         return self._contraction_helper('max', spaces)

    def mean(self, spaces=None):
        """Determines the mean over the sub-domains given by `spaces`.

        ``x.mean(spaces)`` is equivalent to
        ``x.integrate(spaces)/x.total_volume(spaces)``.

        Parameters
        ----------
        spaces : None, int or tuple of int
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.

        Returns
        -------
        Field
            The result of the operation.
        """
        if self.scalar_weight(spaces) is not None:
            return self._contraction_helper('mean', spaces)
        # MR FIXME: not very efficient
        # MR FIXME: do we need "spaces" here?
        tmp = self.weight(1, spaces)
        return tmp.sum(spaces)*(1./tmp.total_volume(spaces))

    def s_mean(self):
        """Determines the field mean

        ``x.s_mean()`` is equivalent to
        ``x.s_integrate()/x.total_volume()``.

        Returns
        -------
        scalar
            The result of the operation.
        """
        return self.s_integrate()/self.total_volume()

    def var(self, spaces=None):
        """Determines the variance over the sub-domains given by `spaces`.

        Parameters
        ----------
        spaces : None, int or tuple of int
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.
            Default: None.

        Returns
        -------
        Field
            The result of the operation.
        """
        if self.scalar_weight(spaces) is not None:
            return self._contraction_helper('var', spaces)
        # MR FIXME: not very efficient or accurate
        m1 = self.mean(spaces)
        from .operators.contraction_operator import ContractionOperator
        op = ContractionOperator(self._domain, spaces)
        m1 = op.adjoint_times(m1)
        if utilities.iscomplextype(self.dtype):
            sq = abs(self-m1)**2
        else:
            sq = (self-m1)**2
        return sq.mean(spaces)

    def s_var(self):
        """Determines the field variance

        Returns
        -------
        scalar
            The result of the operation.
        """
        if self.scalar_weight() is not None:
            return self._val.var()
        # MR FIXME: not very efficient or accurate
        m1 = self.s_mean()
        if utilities.iscomplextype(self.dtype):
            sq = abs(self-m1)**2
        else:
            sq = (self-m1)**2
        return sq.s_mean()

    def std(self, spaces=None):
        """Determines the standard deviation over the sub-domains given by
        `spaces`.

        ``x.std(spaces)`` is equivalent to ``sqrt(x.var(spaces))``.

        Parameters
        ----------
        spaces : None, int or tuple of int
            The operation is only carried out over the sub-domains in this
            tuple. If None, it is carried out over all sub-domains.
            Default: None.

        Returns
        -------
        Field
            The result of the operation.
        """
        if self.scalar_weight(spaces) is not None:
            return self._contraction_helper('std', spaces)
        return self.var(spaces).ptw("sqrt")

    def s_std(self):
        """Determines the standard deviation of the Field.

        ``x.s_std()`` is equivalent to ``sqrt(x.s_var())``.

        Returns
        -------
        scalar
            The result of the operation.
        """
        if self.scalar_weight() is not None:
            return self._val.std()
        return np.sqrt(self.s_var())

    def __repr__(self):
        return "<nifty.cl.Field>"

    def __str__(self):
        return "\n".join(["nifty.cl.Field instance",
                          f"- domain      = {self._domain.__str__()}",
                          f"- val         = {repr(self._val)}",
                          f"- nbytes      = {self._val.nbytes*1e-6:.3f} MB"])

    def extract(self, dom):
        utilities.check_object_identity(dom, self._domain)
        return self

    def extract_part(self, dom):
        utilities.check_object_identity(dom, self._domain)
        return self

    def unite(self, other):
        return self+other

    def flexible_addsub(self, other, neg):
        return self-other if neg else self+other

    def _binary_op(self, other, op):
        # if other is a field, make sure that the domains match
        f = getattr(self._val, op)
        if isinstance(other, Field):
            utilities.check_object_identity(other._domain, self._domain)
            return Field(self._domain, f(other._val))
        if np.isscalar(other):
            return Field(self._domain, f(other))
        return NotImplemented

    def ptw(self, op, *args, **kwargs):
        return Field(self._domain, self._val.ptw(op, *args, **kwargs))

    def ptw_with_deriv(self, op, *args, **kwargs):
        val, deriv = self._val.ptw_with_deriv(op, *args, **kwargs)
        return Field(self._domain, val), Field(self._domain, deriv)


for op in ["__add__", "__radd__",
           "__sub__", "__rsub__",
           "__mul__", "__rmul__",
           "__truediv__", "__rtruediv__",
           "__floordiv__", "__rfloordiv__",
           "__pow__", "__rpow__",
           "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"]:
    def func(op):
        def func2(self, other):
            return self._binary_op(other, op)
        return func2
    setattr(Field, op, func(op))

for op in ["__iadd__", "__isub__", "__imul__", "__idiv__",
           "__itruediv__", "__ifloordiv__", "__ipow__"]:
    def func(op):
        def func2(self, other):
            raise TypeError(
                "In-place operations are deliberately not supported")
        return func2
    setattr(Field, op, func(op))
