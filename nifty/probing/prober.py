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
#
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from builtins import str
from builtins import range
from builtins import object
import numpy as np
from ..field import Field, DomainTuple
from .. import utilities


class Prober(object):
    """
    See the following webpages for the principles behind the usage of
    mixin-classes

    https://www.python.org/download/releases/2.2.3/descrintro/#cooperation
    https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
    """

    def __init__(self, domain=None, probe_count=8,
                 random_type='pm1', probe_dtype=np.float,
                 compute_variance=False, ncpu=1):

        self._domain = DomainTuple.make(domain)
        self._probe_count = self._parse_probe_count(probe_count)
        self._ncpu = self._parse_probe_count(ncpu)
        self._random_type = self._parse_random_type(random_type)
        self.compute_variance = bool(compute_variance)
        self.probe_dtype = np.dtype(probe_dtype)
        self._uid_counter = 0

    # ---Properties---

    @property
    def domain(self):
        return self._domain

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

    def gen_parallel_probe(self,callee):
        for i in range(self.probe_count):
            yield (callee, self.get_probe(i))

    def probing_run(self, callee):
        """ controls the generation, evaluation and finalization of probes """
        self.reset()
        if self._ncpu == 1:
            for index in range(self.probe_count):
                current_probe = self.get_probe(index)
                pre_result = self.process_probe(callee, current_probe, index)
                self.finish_probe(current_probe, pre_result)
        else:
            from multiprocess import Pool
            pool = Pool(self._ncpu)
            for i in pool.imap_unordered(self.evaluate_probe_parallel,
                                         self.gen_parallel_probe(callee)):
                self.finish_probe(i[0],i[1])

    def evaluate_probe_parallel(self, argtuple):
        callee = argtuple[0]
        probe = argtuple[1]
        return (probe, callee(probe[1]))

    def reset(self):
        pass

    def get_probe(self, index):
        """ layer of abstraction for potential probe-caching """
        return self.generate_probe()

    def generate_probe(self):
        """ a random-probe generator """
        f = Field.from_random(random_type=self.random_type,
                              domain=self.domain,
                              dtype=self.probe_dtype.type)
        uid = self._uid_counter
        self._uid_counter += 1
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
