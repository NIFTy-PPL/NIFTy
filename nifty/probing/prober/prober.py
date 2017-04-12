# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import abc

import numpy as np

from nifty.field import Field
import nifty.nifty_utilities as utilities

from nifty import nifty_configuration as nc

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES


class Prober(object):
    """
    See the following webpages for the principles behind the usage of
    mixin-classes

    https://www.python.org/download/releases/2.2.3/descrintro/#cooperation
    https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, domain=None, distribution_strategy=None, probe_count=8,
                 random_type='pm1', compute_variance=False):

        self._domain = utilities.parse_domain(domain)
        self._distribution_strategy = \
            self._parse_distribution_strategy(distribution_strategy)
        self._probe_count = self._parse_probe_count(probe_count)
        self._random_type = self._parse_random_type(random_type)
        self.compute_variance = bool(compute_variance)

    # ---Properties---

    @property
    def domain(self):
        return self._domain

    @property
    def distribution_strategy(self):
        return self._distribution_strategy

    def _parse_distribution_strategy(self, distribution_strategy):
        if distribution_strategy is None:
            distribution_strategy = nc['default_distribution_strategy']
        else:
            distribution_strategy = str(distribution_strategy)
        if distribution_strategy not in DISTRIBUTION_STRATEGIES['global']:
            raise ValueError("distribution_strategy must be a global-type "
                             "strategy.")
        self._distribution_strategy = distribution_strategy

    @property
    def probe_count(self):
        return self._probe_count

    def _parse_probe_count(self, probe_count):
        return int(probe_count)

    @property
    def random_type(self):
        return self._random_type

    def _parse_random_type(self, random_type):
        if random_type not in ["pm1", "normal"]:
            raise ValueError(
                "unsupported random type: '" + str(random_type) + "'.")
        return random_type

    # ---Probing methods---

    def probing_run(self, callee):
        """ controls the generation, evaluation and finalization of probes """
        self.reset()
        for index in xrange(self.probe_count):
            current_probe = self.get_probe(index)
            pre_result = self.process_probe(callee, current_probe, index)
            self.finish_probe(current_probe, pre_result)

    def reset(self):
        pass

    def get_probe(self, index):
        """ layer of abstraction for potential probe-caching """
        return self.generate_probe()

    def generate_probe(self):
        """ a random-probe generator """
        f = Field.from_random(random_type=self.random_type,
                              domain=self.domain,
                              distribution_strategy=self.distribution_strategy)
        uid = np.random.randint(1e18)
        return (uid, f)

    def process_probe(self, callee, probe, index):
        """ layer of abstraction for potential result-caching/recycling """
        return self.evaluate_probe(callee, probe[1])

    def evaluate_probe(self, callee, probe, **kwargs):
        """ processes a probe """
        return callee(probe, **kwargs)

    def finish_probe(self, probe, pre_result):
        pass

    def __call__(self, callee):
        return self.probing_run(callee)
