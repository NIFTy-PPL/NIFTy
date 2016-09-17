# -*- coding: utf-8 -*-

import abc

from nifty.config import about
from nifty.field import Field
from nifty.spaces import Space
from nifty.field_types import FieldType
import nifty.nifty_utilities as utilities


class LinearOperator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def _parse_domain(self, domain):
        if domain is None:
            domain = ()
        elif isinstance(domain, Space):
            domain = (domain,)
        elif not isinstance(domain, tuple):
            domain = tuple(domain)

        for d in domain:
            if not isinstance(d, Space):
                raise TypeError(about._errors.cstring(
                    "ERROR: Given object contains something that is not a "
                    "nifty.space."))
        return domain

    def _parse_field_type(self, field_type):
        if field_type is None:
            field_type = ()
        elif isinstance(field_type, FieldType):
            field_type = (field_type,)
        elif not isinstance(field_type, tuple):
            field_type = tuple(field_type)

        for ft in field_type:
            if not isinstance(ft, FieldType):
                raise TypeError(about._errors.cstring(
                    "ERROR: Given object is not a nifty.FieldType."))
        return field_type

    @abc.abstractproperty
    def domain(self):
        raise NotImplementedError

    @abc.abstractproperty
    def target(self):
        raise NotImplementedError

    @abc.abstractproperty
    def field_type(self):
        raise NotImplementedError

    @abc.abstractproperty
    def field_type_target(self):
        raise NotImplementedError

    @abc.abstractproperty
    def implemented(self):
        raise NotImplementedError

    @abc.abstractproperty
    def unitary(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.times(*args, **kwargs)

    def times(self, x, spaces=None, types=None):
        spaces, types = self._check_input_compatibility(x, spaces, types)

        if not self.implemented:
            x = x.weight(spaces=spaces)

        y = self._times(x, spaces, types)
        return y

    def inverse_times(self, x, spaces=None, types=None):
        spaces, types = self._check_input_compatibility(x, spaces, types,
                                                        inverse=True)

        y = self._inverse_times(x, spaces, types)
        if not self.implemented:
            y = y.weight(power=-1, spaces=spaces)
        return y

    def adjoint_times(self, x, spaces=None, types=None):
        if self.unitary:
            return self.inverse_times(x, spaces, types)

        spaces, types = self._check_input_compatibility(x, spaces, types,
                                                        inverse=True)

        if not self.implemented:
            x = x.weight(spaces=spaces)
        y = self._adjoint_times(x, spaces, types)
        return y

    def adjoint_inverse_times(self, x, spaces=None, types=None):
        if self.unitary:
            return self.times(x, spaces, types)

        spaces, types = self._check_input_compatibility(x, spaces, types)

        y = self._adjoint_inverse_times(x, spaces, types)
        if not self.implemented:
            y = y.weight(power=-1, spaces=spaces)
        return y

    def inverse_adjoint_times(self, x, spaces=None, types=None):
        if self.unitary:
            return self.times(x, spaces, types)

        spaces, types = self._check_input_compatibility(x, spaces, types)

        y = self._inverse_adjoint_times(x, spaces, types)
        if not self.implemented:
            y = y.weight(power=-1, spaces=spaces)
        return y

    def _times(self, x, spaces, types):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'times'."))

    def _adjoint_times(self, x, spaces, types):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'adjoint_times'."))

    def _inverse_times(self, x, spaces, types):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'inverse_times'."))

    def _adjoint_inverse_times(self, x, spaces, types):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'adjoint_inverse_times'."))

    def _inverse_adjoint_times(self, x, spaces, types):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'inverse_adjoint_times'."))

    def _check_input_compatibility(self, x, spaces, types, inverse=False):
        if not isinstance(x, Field):
            raise ValueError(about._errors.cstring(
                "ERROR: supplied object is not a `nifty.Field`."))

        # sanitize the `spaces` and `types` input
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))
        types = utilities.cast_axis_to_tuple(types, len(x.field_type))

        # if the operator's domain is set to something, there are two valid
        # cases:
        # 1. Case:
        #   The user specifies with `spaces` that the operators domain should
        #   be applied to certain spaces in the domain-tuple of x.
        # 2. Case:
        #   The domains of self and x match completely.

        if not inverse:
            self_domain = self.domain
            self_field_type = self.field_type
        else:
            self_domain = self.target
            self_field_type = self.field_type_target

        if spaces is None:
            if self_domain != () and self_domain != x.domain:
                raise ValueError(about._errors.cstring(
                    "ERROR: The operator's and and field's domains don't "
                    "match."))
        else:
            for i, space_index in enumerate(spaces):
                if x.domain[space_index] != self_domain[i]:
                    raise ValueError(about._errors.cstring(
                        "ERROR: The operator's and and field's domains don't "
                        "match."))

        if types is None:
            if self_field_type != () and self_field_type != x.field_type:
                raise ValueError(about._errors.cstring(
                    "ERROR: The operator's and and field's field_types don't "
                    "match."))
        else:
            for i, field_type_index in enumerate(types):
                if x.field_type[field_type_index] != self_field_type[i]:
                    raise ValueError(about._errors.cstring(
                        "ERROR: The operator's and and field's field_type "
                        "don't match."))

        return (spaces, types)

    def __repr__(self):
        return str(self.__class__)
