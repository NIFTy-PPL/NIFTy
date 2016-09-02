# -*- coding: utf-8 -*-

import abc

from nifty.config import about,\
                         nifty_configuration as gc

from nifty.field import Field
from nifty.operators import LinearOperator

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES


class Prober(object):
    def __init__(self, operator, random='pm1', distribution_strategy=None):

        self.operator = operator
        self.random = random

        if distribution_strategy is None:
            distribution_strategy = gc['default_distribution_strategy']
        elif distribution_strategy not in DISTRIBUTION_STRATEGIES['global']:
            raise ValueError(about._errors.cstring(
                    "ERROR: distribution_strategy must be a global-type "
                    "strategy."))
        self._distribution_strategy = distribution_strategy

    @property
    def distribution_strategy(self):
        return self._distribution_strategy

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, operator):
        if not isinstance(operator, LinearOperator):
            raise ValueError(about._errors.cstring(
                    "ERROR: The given operator is not an instance of the "
                    "LinearOperator class."))
        self._operator = operator

    @property
    def random(self):
        return self._random

    @random.setter
    def random(self, random):
        if random not in ["pm1", "normal"]:
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported random key '" + str(random) + "'."))
        else:
            self._random = random

    @abc.abstractmethod
    def probing_function(self):
        """A callable which is constructed from self.operator."""
        raise NotImplementedError

    def generate_probe(self, inverse=False):
        if not inverse:
            domain = self.operator.domain
            field_type = self.operator.field_type
        else:
            domain = self.operator.target
            field_type = self.operator.field_type_target

        f = Field.from_random(random=self.random,
                              domain=domain,
                              field_type=field_type,
                              distribution_strategy=self.distribution_strategy)
        return f

    def probe(self, compute_variance=False):
        raise NotImplementedError
