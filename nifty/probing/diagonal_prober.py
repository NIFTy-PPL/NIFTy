# -*- coding: utf-8 -*-

from nifty.operators import EndomorphicOperator

from operator_prober import OperatorProber

__all__ = ['DiagonalProber', 'InverseDiagonalProber',
           'AdjointDiagonalProber', 'AdjointInverseDiagonalProber']


class DiagonalTypeProber(OperatorProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def valid_operator_class(self):
        return EndomorphicOperator

    # --- ->Mandatory from Prober---

    def finish_probe(self, probe, pre_result):
        return probe[1].conjugate()*pre_result


class DiagonalProber(DiagonalTypeProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def is_inverse(self):
        return False

    # --- ->Mandatory from Prober---

    def evaluate_probe(self, probe):
        """ processes a probe """
        return self.operator.times(probe[1])


class InverseDiagonalProber(DiagonalTypeProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def is_inverse(self):
        return True

    # --- ->Mandatory from Prober---

    def evaluate_probe(self, probe):
        """ processes a probe """
        return self.operator.inverse_times(probe[1])


class AdjointDiagonalProber(DiagonalTypeProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def is_inverse(self):
        return False

    # --- ->Mandatory from Prober---

    def evaluate_probe(self, probe):
        """ processes a probe """
        return self.operator.adjoint_times(probe[1])


class AdjointInverseDiagonalProber(DiagonalTypeProber):

    # ---Mandatory properties and methods---
    # --- ->Mandatory from OperatorProber---

    @property
    def is_inverse(self):
        return True

    # --- ->Mandatory from Prober---

    def evaluate_probe(self, probe):
        """ processes a probe """
        return self.operator.adjoint_inverse_times(probe[1])
