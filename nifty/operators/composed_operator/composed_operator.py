# -*- coding: utf-8 -*-

from nifty.operators.linear_operator import LinearOperator


class ComposedOperator(LinearOperator):
    # ---Overwritten properties and methods---
    def __init__(self, operators):
        self._operator_store = ()
        for op in operators:
            if not isinstance(op, LinearOperator):
                raise TypeError("The elements of the operator list must be"
                                "instances of the LinearOperator-baseclass")
            self._operator_store += (op,)

    def _check_input_compatibility(self, x, spaces, types, inverse=False):
        """
        The input check must be disabled for the ComposedOperator, since it
        is not easily forecasteable what the output of an operator-call
        will look like.
        """
        return (spaces, types)

    # ---Mandatory properties and methods---
    @property
    def domain(self):
        if not hasattr(self, '_domain'):
            self._domain = ()
            for op in self._operator_store:
                self._domain += op.domain
        return self._domain

    @property
    def target(self):
        if not hasattr(self, '_target'):
            self._target = ()
            for op in self._operator_store:
                self._target += op.target
        return self._target

    @property
    def field_type(self):
        if not hasattr(self, '_field_type'):
            self._field_type = ()
            for op in self._operator_store:
                self._field_type += op.field_type
        return self._field_type

    @property
    def field_type_target(self):
        if not hasattr(self, '_field_type_target'):
            self._field_type_target = ()
            for op in self._operator_store:
                self._field_type_target += op.field_type_target
        return self._field_type_target

    @property
    def implemented(self):
        return True

    @property
    def unitary(self):
        return False

    def _times(self, x, spaces, types):
        space_index = 0
        type_index = 0
        if spaces is None:
            spaces = range(len(self.domain))
        if types is None:
            types = range(len(self.field_type))
        for op in self._operator_store:
            active_spaces = spaces[space_index:space_index+len(op.domain)]
            space_index += len(op.domain)

            active_types = types[type_index:type_index+len(op.field_type)]
            type_index += len(op.field_type)

            x = op(x, spaces=active_spaces, types=active_types)
        return x
