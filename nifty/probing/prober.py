# -*- coding: utf-8 -*-

import abc

import numpy as np

from nifty.config import about,\
                         nifty_configuration as gc

from nifty.field import Field

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES


class Prober(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, probe_count=8, random_type='pm1',
                 distribution_strategy=None, compute_variance=False):

        self.probe_count = probe_count

        self.random_type = random_type

        if distribution_strategy is None:
            distribution_strategy = gc['default_distribution_strategy']
        self.distribution_strategy = distribution_strategy

        self.compute_variance = bool(compute_variance)

    # ---Properties---

    @abc.abstractproperty
    def domain(self):
        raise NotImplemented

    @abc.abstractproperty
    def field_type(self):
        raise NotImplemented

    @property
    def distribution_strategy(self):
        return self._distribution_strategy

    @distribution_strategy.setter
    def distribution_strategy(self, distribution_strategy):
        distribution_strategy = str(distribution_strategy)
        if distribution_strategy not in DISTRIBUTION_STRATEGIES['global']:
            raise ValueError(about._errors.cstring(
                    "ERROR: distribution_strategy must be a global-type "
                    "strategy."))
        self._distribution_strategy = distribution_strategy

    @property
    def probe_count(self):
        return self._probe_count

    @probe_count.setter
    def probe_count(self, probe_count):
        self._probe_count = int(probe_count)

    @property
    def random_type(self):
        return self._random_type

    @random_type.setter
    def random_type(self, random_type):
        if random_type not in ["pm1", "normal"]:
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported random type: '" + str(random) + "'."))
        else:
            self._random_type = random_type

    # ---Probing methods---

    def probing_run(self):
        """ controls the generation, evaluation and finalization of probes """
        sum_of_probes = 0
        sum_of_squares = 0

        for index in xrange(self.probe_count):
            current_probe = self.get_probe(index)
            pre_result = self.process_probe(current_probe, index)
            result = self.finish_probe(pre_result)

            sum_of_probes += result
            if self.compute_variance:
                sum_of_squares += result.conjugate() * result

        mean_and_variance = self.finalize(sum_of_probes, sum_of_squares)
        return mean_and_variance

    def get_probe(self, index):
        """ layer of abstraction for potential probe-caching """
        return self.generate_probe()

    def generate_probe(self):
        """ a random-probe generator """
        f = Field.from_random(random_type=self.random_type,
                              domain=self.domain,
                              field_type=self.field_type,
                              distribution_strategy=self.distribution_strategy)
        uid = np.random.randint(1e18)
        return (uid, f)

    def process_probe(self, probe, index):
        return self.evaluate_probe(probe)

    @abc.abstractmethod
    def evaluate_probe(self, probe):
        """ processes a probe """
        raise NotImplementedError

    def finish_probe(self, probe):
        return probe

    def finalize(self, sum_of_probes, sum_of_squares):
        probe_count = self.probe_count
        mean_of_probes = sum_of_probes/probe_count
        if self.compute_variance:
            # variance = 1/(n-1) (sum(x^2) - 1/n*sum(x)^2)
            variance = ((sum_of_squares - sum_of_probes*mean_of_probes) /
                        (probe_count-1))
        else:
            variance = None

        return (mean_of_probes, variance)

    def __call__(self):
        return self.probe()
