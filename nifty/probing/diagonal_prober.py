# -*- coding: utf-8 -*-

from nifty.operators import EndomorphicOperator

from operator_prober import OperatorProber


class DiagonalProber(OperatorProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def is_inverse(self):
        return False

    @property
    def valid_operator_class(self):
        return EndomorphicOperator

    # --- ->Mandatory from Prober---

    def evaluate_probe(self, probe):
        """ processes a probe """
        return self.operator.times(probe[1])

    def finish_probe(self, probe, pre_result):
        return probe[1].conjugate()*pre_result
