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

from __future__ import division
from builtins import object
from ...sugar import create_composed_fft_operator


class TraceProberMixin(object):
    def __init__(self, *args, **kwargs):
        self.reset()
        self.__evaluate_probe_in_signal_space = False
        super(TraceProberMixin, self).__init__(*args, **kwargs)

    def reset(self):
        self.__sum_of_probings = 0
        self.__sum_of_squares = 0
        self.__trace = None
        self.__trace_variance = None
        super(TraceProberMixin, self).reset()

    def finish_probe(self, probe, pre_result):
        if self.__evaluate_probe_in_signal_space:
            fft = create_composed_fft_operator(self._domain, all_to='position')
            result = fft(probe[1]).vdot(fft(pre_result), bare=True)
        else:
            result = probe[1].vdot(pre_result, bare=True)

        self.__sum_of_probings += result
        if self.compute_variance:
            self.__sum_of_squares += result.conjugate() * result
        super(TraceProberMixin, self).finish_probe(probe, pre_result)

    @property
    def trace(self):
        if self.__trace is None:
            self.__trace = self.__sum_of_probings/self.probe_count
        return self.__trace

    @property
    def trace_variance(self):
        if not self.compute_variance:
            raise AttributeError("self.compute_variance is set to False")
        if self.__trace_variance is None:
            # variance = 1/(n-1) (sum(x^2) - 1/n*sum(x)^2)
            n = self.probe_count
            sum_pr = self.__sum_of_probings
            mean = self.trace
            sum_sq = self.__sum_of_squares

            self.__trace_variance = (sum_sq - sum_pr*mean) / (n-1)
        return self.__trace_variance
