# -*- coding: utf-8 -*-

from nifty.config import about
from operator_paradict import OperatorParadict


class LinearOperator(object):

    def __init__(self, domain=None, target=None,
                 field_type=None, field_type_target=None,
                 implemented=False, symmetric=False, unitary=False,
                 **kwargs):
        self.paradict = OperatorParadict(**kwargs)

        self.implemented = implemented
        self.symmetric = symmetric
        self.unitary = unitary

    @property
    def implemented(self):
        return self._implemented

    @implemented.setter
    def implemented(self, b):
        self._implemented = bool(b)

    @property
    def symmetric(self):
        return self._symmetric

    @symmetric.setter
    def symmetric(self, b):
        self._symmetric = bool(b)

    @property
    def unitary(self):
        return self._unitary

    @unitary.setter
    def unitary(self, b):
        self._unitary = bool(b)

    def times(self, x, spaces=None, types=None):
        raise NotImplementedError

    def adjoint_times(self, x, spaces=None, types=None):
        raise NotImplementedError

    def inverse_times(self, x, spaces=None, types=None):
        raise NotImplementedError

    def adjoint_inverse_times(self, x, spaces=None, types=None):
        raise NotImplementedError

    def inverse_adjoint_times(self, x, spaces=None, types=None):
        raise NotImplementedError

    def _times(self, x, **kwargs):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'times'."))

    def _adjoint_times(self, x, **kwargs):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'adjoint_times'."))

    def _inverse_times(self, x, **kwargs):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'inverse_times'."))

    def _adjoint_inverse_times(self, x, **kwargs):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'adjoint_inverse_times'."))

    def _inverse_adjoint_times(self, x, **kwargs):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: no generic instance method 'inverse_adjoint_times'."))

    def _check_input_compatibility(self, x, spaces, types):
        # assert: x is a field
        # if spaces is None -> assert f.domain == self.domain
        # -> same for field_type
        # else: check if self.domain/self.field_type == one entry.
        #


