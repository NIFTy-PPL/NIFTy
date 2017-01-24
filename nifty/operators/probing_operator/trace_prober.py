# -*- coding: utf-8 -*-

from prober import Prober


class TraceProber(Prober):

    # ---Mandatory properties and methods---
    def finish_probe(self, probe, pre_result):
        return probe[1].conjugate().weight(power=-1).dot(pre_result)
