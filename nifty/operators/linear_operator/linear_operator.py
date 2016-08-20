# -*- coding: utf-8 -*-

from nifty.config import about
from nifty.field import Field
from nifty.spaces import Space
from nifty.field_types import FieldType
import nifty.nifty_utilities as utilities

from linear_operator_paradict import LinearOperatorParadict


class LinearOperator(object):

    def __init__(self, domain=None, target=None,
                 field_type=None, field_type_target=None,
                 implemented=False, symmetric=False, unitary=False):
        self.paradict = LinearOperatorParadict()

        self._implemented = bool(implemented)

        self.domain = self._parse_domain(domain)
        self.target = self._parse_domain(target)

        self.field_type = self._parse_field_type(field_type)
        self.field_type_target = self._parse_field_type(field_type_target)

    def _parse_domain(self, domain):
        if domain is None:
            domain = ()
        elif not isinstance(domain, tuple):
            domain = (domain,)
        for d in domain:
            if not isinstance(d, Space):
                raise TypeError(about._errors.cstring(
                    "ERROR: Given object contains something that is not a "
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

    @property
    def implemented(self):
        return self._implemented

    def __call__(self, *args, **kwargs):
        return self.times(*args, **kwargs)

    def times(self, x, spaces=None, types=None):
        spaces, types = self._check_input_compatibility(x, spaces, types)

        if not self.implemented:
            x = x.weight(spaces=spaces)

        y = self._times(x, spaces, types)
        return y

    def inverse_times(self, x, spaces=None, types=None):
        spaces, types = self._check_input_compatibility(x, spaces, types)

        y = self._inverse_times(x, spaces, types)
        if not self.implemented:
            y = y.weight(power=-1, spaces=spaces)
        return y

    def adjoint_times(self, x, spaces=None, types=None):
        spaces, types = self._check_input_compatibility(x, spaces, types)

        if not self.implemented:
            x = x.weight(spaces=spaces)
        y = self._adjoint_times(x, spaces, types)
        return y

    def adjoint_inverse_times(self, x, spaces=None, types=None):
        spaces, types = self._check_input_compatibility(x, spaces, types)

        y = self._adjoint_inverse_times(x, spaces, types)
        if not self.implemented:
            y = y.weight(power=-1, spaces=spaces)
        return y

    def inverse_adjoint_times(self, x, spaces=None, types=None):
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

    def _check_input_compatibility(self, x, spaces, types):
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
        #   be applied to a certain domain in the domain-tuple of x. This is
        #   only valid if len(self.domain)==1.
        # 2. Case:
        #   The domains of self and x match completely.

        if spaces is None:
            if self.domain != () and self.domain != x.domain:
                raise ValueError(about._errors.cstring(
                    "ERROR: The operator's and and field's domains don't "
                    "match."))
        else:
            if len(self.domain) > 1:
                raise ValueError(about._errors.cstring(
                    "ERROR: Specifying `spaces` for operators with multiple "
                    "domain spaces is not valid."))
            elif len(spaces) != len(self.domain):
                raise ValueError(about._errors.cstring(
                    "ERROR: Length of `spaces` does not match the number of "
                    "spaces in the operator's domain."))
            elif len(spaces) == 1:
                if x.domain[spaces[0]] != self.domain[0]:
                    raise ValueError(about._errors.cstring(
                        "ERROR: The operator's and and field's domains don't "
                        "match."))

        if types is None:
            if self.field_type != () and self.field_type != x.field_type:
                raise ValueError(about._errors.cstring(
                    "ERROR: The operator's and and field's field_types don't "
                    "match."))
        else:
            if len(self.field_type) > 1:
                raise ValueError(about._errors.cstring(
                    "ERROR: Specifying `types` for operators with multiple "
                    "field-types is not valid."))
            elif len(types) != len(self.field_type):
                raise ValueError(about._errors.cstring(
                    "ERROR: Length of `types` does not match the number of "
                    "the operator's field-types."))
            elif len(types) == 1:
                if x.field_type[types[0]] != self.field_type[0]:
                    raise ValueError(about._errors.cstring(
                        "ERROR: The operator's and and field's field_type "
                        "don't match."))
        return (spaces, types)

    def __repr__(self):
        return str(self.__class__)
