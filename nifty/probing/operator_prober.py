# -*- coding: utf-8 -*-

import abc


from prober import Prober


class OperatorProber(Prober):

    # ---Overwritten properties and methods---

    def __init__(self, operator, probe_count=8, random_type='pm1',
                 distribution_strategy=None, compute_variance=False):
        super(OperatorProber, self).__init__(
                                 probe_count=probe_count,
                                 random_type=random_type,
                                 compute_variance=compute_variance)

        if not isinstance(operator, self.valid_operator_class):
            raise TypeError("Operator must be an instance of %s" %
                            str(self.valid_operator_class))
        self._operator = operator

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        if self.is_inverse:
            return self.operator.target
        else:
            return self.operator.domain

    @property
    def field_type(self):
        if self.is_inverse:
            return self.operator.field_type_target
        else:
            return self.operator.field_type

    @property
    def distribution_strategy(self):
        return self.operator.distribution_strategy

    # ---Added properties and methods---

    @abc.abstractproperty
    def is_inverse(self):
        raise NotImplementedError

    @abc.abstractproperty
    def valid_operator_class(self):
        raise NotImplementedError

    @property
    def operator(self):
        return self._operator
