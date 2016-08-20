from __future__ import division
import numpy as np
import pylab as pl

from d2o import distributed_data_object, \
    STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.config import about, \
    nifty_configuration as gc, \
    dependency_injector as gdi

from nifty.field_types import FieldType,\
                              FieldArray

from nifty.spaces.space import Space

import nifty.nifty_utilities as utilities

POINT_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']
COMM = getattr(gdi[gc['mpi_module']], gc['default_comm'])


class Field(object):
    # ---Initialization methods---
    def __init__(self, domain=None, val=None, dtype=None, field_type=None,
                 datamodel=None, copy=False):
        if isinstance(val, Field):
            if domain is None:
                domain = val.domain
            if dtype is None:
                dtype = val.dtype
            if field_type is None:
                field_type = val.field_type
            if datamodel is None:
                datamodel = val.datamodel

        self.domain = self._parse_domain(domain=domain)
        self.domain_axes = self._get_axes_tuple(self.domain)

        self.field_type = self._parse_field_type(field_type)

        try:
            start = len(reduce(lambda x, y: x+y, self.domain_axes))
        except TypeError:
            start = 0
        self.field_type_axes = self._get_axes_tuple(self.field_type,
                                                    start=start)

        self.dtype = self._infer_dtype(dtype=dtype,
                                       domain=self.domain,
                                       field_type=self.field_type)

        self.datamodel = self._parse_datamodel(datamodel=datamodel,
                                               val=val)

        self.set_val(new_val=val, copy=copy)

    def _parse_domain(self, domain):
        if domain is None:
            domain = ()
        elif not isinstance(domain, tuple):
            domain = (domain,)
        for d in domain:
            if not isinstance(d, Space):
                raise TypeError(about._errors.cstring(
                    "ERROR: Given domain contains something that is not a "
                    "nifty.space."))
        return domain

    def _parse_field_type(self, field_type):
        if field_type is None:
            field_type = ()
        elif not isinstance(field_type, tuple):
            field_type = (field_type,)
        for ft in field_type:
            if not isinstance(ft, FieldType):
                raise TypeError(about._errors.cstring(
                    "ERROR: Given object is not a nifty.FieldType."))
        return field_type

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

    def _infer_dtype(self, dtype=None, domain=None, field_type=None):
        if dtype is None:
            dtype_tuple = (np.dtype(gc['default_field_dtype']),)
        else:
            dtype_tuple = (np.dtype(dtype),)
        if domain is not None:
            dtype_tuple += tuple(np.dtype(sp.dtype) for sp in domain)
        if field_type is not None:
            dtype_tuple += tuple(np.dtype(ft.dtype) for ft in field_type)

        dtype = reduce(lambda x, y: np.result_type(x, y), dtype_tuple)
        return dtype

    def _parse_datamodel(self, datamodel, val):
        if datamodel in DISTRIBUTION_STRATEGIES['all']:
            pass
        elif isinstance(val, distributed_data_object):
            datamodel = val.distribution_strategy
        else:
            datamodel = gc['default_datamodel']

        return datamodel

    # ---Properties---
    def set_val(self, new_val=None, copy=False):
        new_val = self.cast(new_val)
        if copy:
            new_val = new_val.copy()
        self._val = new_val
        return self._val

    def get_val(self, copy=False):
        if copy:
            return self._val.copy()
        else:
            return self._val

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, new_val):
        self._val = self.cast(new_val)

    @property
    def shape(self):
        shape_tuple = ()
        shape_tuple += tuple(sp.shape for sp in self.domain)
        shape_tuple += tuple(ft.shape for ft in self.field_type)
        try:
            global_shape = reduce(lambda x, y: x + y, shape_tuple)
        except TypeError:
            global_shape = ()

        return global_shape

    @property
    def dim(self):
        dim_tuple = ()
        dim_tuple += tuple(sp.dim for sp in self.domain)
        dim_tuple += tuple(ft.dim for ft in self.field_type)
        try:
            return reduce(lambda x, y: x * y, dim_tuple)
        except TypeError:
            return 0

    @property
    def dof(self):
        dof = self.dim
        if issubclass(self.dtype.type, np.complexfloating):
            dof *= 2
        return dof

    @property
    def total_volume(self):
        volume_tuple = tuple(sp.total_volume for sp in self.domain)
        try:
            return reduce(lambda x, y: x * y, volume_tuple)
        except TypeError:
            return 0

    # ---Special unary/binary operations---
    def cast(self, x=None, dtype=None):
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)

        casted_x = self._actual_cast(x, dtype=dtype)

        for ind, sp in enumerate(self.domain):
            casted_x = sp.complement_cast(casted_x,
                                          axes=self.domain_axes[ind])

        for ind, ft in enumerate(self.field_type):
            casted_x = ft.complement_cast(casted_x,
                                          axes=self.field_type_axes[ind])

        return casted_x

    def _actual_cast(self, x, dtype=None):
        if isinstance(x, Field):
            x = x.get_val()

        if dtype is None:
            dtype = self.dtype

        x = distributed_data_object(x,
                                    global_shape=self.shape,
                                    dtype=dtype,
                                    distribution_strategy=self.datamodel)

        return x

    def copy(self, domain=None, dtype=None, field_type=None,
             datamodel=None):
        copied_val = self.get_val(copy=True)
        new_field = self.copy_empty(domain=domain,
                                    dtype=dtype,
                                    field_type=field_type,
                                    datamodel=datamodel)
        new_field.set_val(new_val=copied_val, copy=False)
        return new_field

    def copy_empty(self, domain=None, dtype=None, field_type=None,
                   datamodel=None):
        if domain is None:
            domain = self.domain
        else:
            domain = self._parse_domain(domain)

        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)

        if field_type is None:
            field_type = self.field_type
        else:
            field_type = self._parse_field_type(field_type)

        if datamodel is None:
            datamodel = self.datamodel

        fast_copyable = True
        try:
            for i in xrange(len(self.domain)):
                if self.domain[i] is not domain[i]:
                    fast_copyable = False
                    break
            for i in xrange(len(self.field_type)):
                if self.field_type[i] is not field_type[i]:
                    fast_copyable = False
                    break
        except IndexError:
            fast_copyable = False

        if (fast_copyable and dtype == self.dtype and
                datamodel == self.datamodel):
            new_field = self._fast_copy_empty()
        else:
            new_field = Field(domain=domain,
                              dtype=dtype,
                              field_type=field_type,
                              datamodel=datamodel)
        return new_field

    def _fast_copy_empty(self):
        # make an empty field
        new_field = EmptyField()
        # repair its class
        new_field.__class__ = self.__class__
        # copy domain, codomain and val
        for key, value in self.__dict__.items():
            if key != 'val':
                new_field.__dict__[key] = value
            else:
                new_field.__dict__[key] = self.val.copy_empty()
        return new_field

    def weight(self, power=1, inplace=False, spaces=None):
        if inplace:
            new_field = self
        else:
            new_field = self.copy_empty()

        new_val = self.get_val(copy=False)

        if spaces is None:
            spaces = range(len(self.domain))
        else:
            spaces = utilities.cast_axis_to_tuple(spaces, len(self.domain))

        for ind, sp in enumerate(self.domain):
            if ind in spaces:
                new_val = sp.weight(new_val,
                                    power=power,
                                    axes=self.domain_axes[ind],
                                    inplace=inplace)

        new_field.set_val(new_val=new_val, copy=False)
        return new_field

    def dot(self, x=None, bare=False):
        if isinstance(x, Field):
            try:
                assert len(x.domain) == len(self.domain)
                for index in xrange(len(self.domain)):
                    assert x.domain[index] == self.domain[index]
                for index in xrange(len(self.field_type)):
                    assert x.field_type[index] == self.field_type[index]
            except AssertionError:
                raise ValueError(about._errors.cstring(
                    "ERROR: domains are incompatible."))
            # extract the data from x and try to dot with this
            x = x.get_val(copy=False)

        # Compute the dot respecting the fact of discrete/continous spaces
        if bare:
            y = self
        else:
            y = self.weight(power=1)

        y = y.get_val(copy=False)

        # Cast the input in order to cure dtype and shape differences
        x = self.cast(x)

        dotted = x.conjugate() * y

        return dotted.sum()

    def norm(self, q=2):
        """
            Computes the Lq-norm of the field values.

            Parameters
            ----------
            q : scalar
                Parameter q of the Lq-norm (default: 2).

            Returns
            -------
            norm : scalar
                The Lq-norm of the field values.

        """
        if q == 2:
            return (self.dot(x=self)) ** (1 / 2)
        else:
            return self.dot(x=self ** (q - 1)) ** (1 / q)

    def conjugate(self, inplace=False):
        """
            Computes the complex conjugate of the field.

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
        return self.copy()

    def __neg__(self):
        return_field = self.copy_empty()
        new_val = -self.get_val(copy=False)
        return_field.set_val(new_val, copy=False)
        return return_field

    def __abs__(self):
        return_field = self.copy_empty()
        new_val = abs(self.get_val(copy=False))
        return_field.set_val(new_val, copy=False)
        return return_field

    def _contraction_helper(self, op, spaces, types):
        # build a list of all axes
        if spaces is None:
            spaces = xrange(len(self.domain))
        else:
            spaces = utilities.cast_axis_to_tuple(spaces, len(self.domain))

        if types is None:
            types = xrange(len(self.field_type))
        else:
            types = utilities.cast_axis_to_tuple(types, len(self.field_type))

        axes_list = ()
        axes_list += tuple(self.domain_axes[sp_index] for sp_index in spaces)
        axes_list += tuple(self.field_type_axes[ft_index] for
                           ft_index in types)
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
                                  for i in xrange(len(self.domain))
                                  if i not in spaces)
            return_field_type = tuple(self.field_type[i]
                                      for i in xrange(len(self.field_type))
                                      if i not in types)
            return_field = Field(domain=return_domain,
                                 val=data,
                                 field_type=return_field_type,
                                 copy=False)
            return return_field

    def sum(self, spaces=None, types=None):
        return self._contraction_helper('sum', spaces, types)

    def prod(self, spaces=None, types=None):
        return self._contraction_helper('prod', spaces, types)

    def all(self, spaces=None, types=None):
        return self._contraction_helper('all', spaces, types)

    def any(self, spaces=None, types=None):
        return self._contraction_helper('any', spaces, types)

    def min(self, spaces=None, types=None):
        return self._contraction_helper('min', spaces, types)

    def nanmin(self, spaces=None, types=None):
        return self._contraction_helper('nanmin', spaces, types)

    def max(self, spaces=None, types=None):
        return self._contraction_helper('max', spaces, types)

    def nanmax(self, spaces=None, types=None):
        return self._contraction_helper('nanmax', spaces, types)

    def mean(self, spaces=None, types=None):
        return self._contraction_helper('mean', spaces, types)

    def var(self, spaces=None, types=None):
        return self._contraction_helper('var', spaces, types)

    def std(self, spaces=None, types=None):
        return self._contraction_helper('std', spaces, types)

    # ---General binary methods---

    def _binary_helper(self, other, op, inplace=False):
        # if other is a field, make sure that the domains match
        if isinstance(other, Field):
            try:
                assert len(other.domain) == len(self.domain)
                for index in xrange(len(self.domain)):
                    assert other.domain[index] == self.domain[index]
                for index in xrange(len(self.field_type)):
                    assert other.field_type[index] == self.field_type[index]
            except AssertionError:
                raise ValueError(about._errors.cstring(
                    "ERROR: domains are incompatible."))
            other = other.get_val(copy=False)

        self_val = self.get_val(copy=False)
        return_val = getattr(self_val, op)(other)

        if inplace:
            working_field = self
        else:
            working_field = self.copy_empty()

        working_field.set_val(return_val, copy=False)
        return working_field

    def __add__(self, other):
        return self._binary_helper(other, op='__add__')

    def __radd__(self, other):
        return self._binary_helper(other, op='__radd__')

    def __iadd__(self, other):
        return self._binary_helper(other, op='__iadd__', inplace=True)

    def __sub__(self, other):
        return self._binary_helper(other, op='__sub__')

    def __rsub__(self, other):
        return self._binary_helper(other, op='__rsub__')

    def __isub__(self, other):
        return self._binary_helper(other, op='__isub__', inplace=True)

    def __mul__(self, other):
        return self._binary_helper(other, op='__mul__')

    def __rmul__(self, other):
        return self._binary_helper(other, op='__rmul__')

    def __imul__(self, other):
        return self._binary_helper(other, op='__imul__', inplace=True)

    def __div__(self, other):
        return self._binary_helper(other, op='__div__')

    def __rdiv__(self, other):
        return self._binary_helper(other, op='__rdiv__')

    def __idiv__(self, other):
        return self._binary_helper(other, op='__idiv__', inplace=True)

    def __pow__(self, other):
        return self._binary_helper(other, op='__pow__')

    def __rpow__(self, other):
        return self._binary_helper(other, op='__rpow__')

    def __ipow__(self, other):
        return self._binary_helper(other, op='__ipow__', inplace=True)

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


class EmptyField(Field):
    def __init__(self):
        pass
