# -*- coding: utf-8 -*-

from nifty.operators import EndomorphicOperator

from operator_prober import OperatorProber

__all__ = ['TraceProber', 'InverseTraceProber',
           'AdjointTraceProber', 'AdjointInverseTraceProber']


class TraceTypeProber(OperatorProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def valid_operator_class(self):
        return EndomorphicOperator

    # --- ->Mandatory from Prober---

    def finish_probe(self, probe, pre_result):
        return probe[1].conjugate().weight(power=-1).dot(pre_result)


class TraceProber(TraceTypeProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def is_inverse(self):
        return False

    # --- ->Mandatory from Prober---

    def evaluate_probe(self, probe):
        """ processes a probe """
        return self.operator.times(probe[1])


class InverseTraceProber(TraceTypeProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def is_inverse(self):
        return True

    # --- ->Mandatory from Prober---

    def evaluate_probe(self, probe):
        """ processes a probe """
        return self.operator.inverse_times(probe[1])


class AdjointTraceProber(TraceTypeProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def is_inverse(self):
        return False

    # --- ->Mandatory from Prober---

    def evaluate_probe(self, probe):
        """ processes a probe """
        return self.operator.adjoint_times(probe[1])


class AdjointInverseTraceProber(TraceTypeProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def is_inverse(self):
        return True

    # --- ->Mandatory from Prober---

    def evaluate_probe(self, probe):
        """ processes a probe """
        return self.operator.adjoint_inverse_times(probe[1])
