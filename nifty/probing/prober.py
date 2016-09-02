# -*- coding: utf-8 -*-

import abc

from nifty.config import about,\
                         nifty_configuration as gc

from nifty.field import Field

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES


class Prober(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, random='pm1', distribution_strategy=None,
                 compute_variance=False):

        self.random = random

        if distribution_strategy is None:
            distribution_strategy = gc['default_distribution_strategy']
        elif distribution_strategy not in DISTRIBUTION_STRATEGIES['global']:
            raise ValueError(about._errors.cstring(
                    "ERROR: distribution_strategy must be a global-type "
                    "strategy."))
        self._distribution_strategy = distribution_strategy

        self.compute_variance = bool(compute_variance)

    @abc.abstractproperty
    def domain(self):
        raise NotImplemented

    @abc.abstractproperty
    def field_type(self):
        raise NotImplemented

    @abc.abstractmethod
    def probe(self):
        """ controls the generation, evaluation and finalization of probes """
        raise NotImplementedError

    @abc.abstractmethod
    def probing_function(self, probe):
        """ processes probes """
        raise NotImplementedError

    @property
    def distribution_strategy(self):
        return self._distribution_strategy

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

    def generate_probe(self):
        """ a random-probe generator """
        f = Field.from_random(random=self.random,
                              domain=self.domain,
                              field_type=self.field_type,
                              distribution_strategy=self.distribution_strategy)
        return f

    def get_probe(self, index=None):
        """ layer of abstraction for potential probe-caching """
        return self.generate_probe()

    def finalize(self, probes_count, sum_of_probes, sum_of_squares):
        mean_of_probes = sum_of_probes/probes_count
        if self.compute_variance:
            # variance = 1/(n-1) (sum(x^2) - 1/n*sum(x)^2)
            variance = ((sum_of_squares - sum_of_probes*mean_of_probes) /
                        (probes_count-1))
        else:
            variance = None

        return (mean_of_probes, variance)

    def __call__(self):
        return self.probe()
