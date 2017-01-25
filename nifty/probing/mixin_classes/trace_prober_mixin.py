# -*- coding: utf-8 -*-

from mixin_base import MixinBase


class TraceProberMixin(MixinBase):
    def __init__(self):
        self.reset()
        super(TraceProberMixin, self).__init__()

    def reset(self):
        self.__sum_of_probings = 0
        self.__sum_of_squares = 0
        self.__trace = None
        self.__trace_variance = None
        super(TraceProberMixin, self).reset()

    def finish_probe(self, probe, pre_result):
        result = probe[1].dot(pre_result, bare=True)
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

            self.__trace_variance = ((sum_sq - sum_pr*mean) / (n-1))
        return self.__trace_variance
