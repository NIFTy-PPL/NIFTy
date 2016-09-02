# -*- coding: utf-8 -*-

import abc

from nifty import about

from prober import Prober


class OperatorProber(Prober):

    # ---Overwritten properties and methods---

    def __init__(self, operator, random='pm1', distribution_strategy=None,
                 compute_variance=False):
        super(OperatorProber, self).__init__(
                                 random=random,
                                 distribution_strategy=distribution_strategy,
                                 compute_variance=compute_variance)

        self.operator = operator

    # ---Mandatory properties and methods---

    @abc.abstractproperty
    def domain(self):
        if self.is_inverse:
            return self.operator.target
        else:
            return self.operator.domain

    @abc.abstractproperty
    def field_type(self):
        if self.is_inverse:
            return self.operator.field_type_target
        else:
            return self.operator.field_type

    # ---Added properties and methods---

    @abc.abstractproperty
    def valid_operator_class(self):
        raise NotImplementedError

    @property
    def is_inverse(self):
        raise NotImplementedError

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, operator):
        if not isinstance(operator, self.valid_operator_class):
            raise ValueError(about._errors.cstring(
                    "ERROR: The given operator is not an instance of the "
                    "LinearOperator class."))
        self._operator = operator
