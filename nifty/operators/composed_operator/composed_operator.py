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

    def _check_input_compatibility(self, x, spaces, inverse=False):
        """
        The input check must be disabled for the ComposedOperator, since it
        is not easily forecasteable what the output of an operator-call
        will look like.
        """
        return spaces

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
    def implemented(self):
        return True

    @property
    def unitary(self):
        return False

    def _times(self, x, spaces):
        return self._times_helper(x, spaces, func='times')

    def _adjoint_times(self, x, spaces):
        return self._inverse_times_helper(x, spaces, func='adjoint_times')

    def _inverse_times(self, x, spaces):
        return self._inverse_times_helper(x, spaces, func='inverse_times')

    def _adjoint_inverse_times(self, x, spaces):
        return self._times_helper(x, spaces, func='adjoint_inverse_times')

    def _inverse_adjoint_times(self, x, spaces):
        return self._times_helper(x, spaces, func='inverse_adjoint_times')

    def _times_helper(self, x, spaces, func):
        space_index = 0
        if spaces is None:
            spaces = range(len(self.domain))
        for op in self._operator_store:
            active_spaces = spaces[space_index:space_index+len(op.domain)]
            space_index += len(op.domain)

            x = getattr(op, func)(x, spaces=active_spaces)
        return x

    def _inverse_times_helper(self, x, spaces, func):
        space_index = 0
        if spaces is None:
            spaces = range(len(self.target))[::-1]
        for op in reversed(self._operator_store):
            active_spaces = spaces[space_index:space_index+len(op.target)]
            space_index += len(op.target)

            x = getattr(op, func)(x, spaces=active_spaces)
        return x
