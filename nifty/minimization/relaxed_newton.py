# -*- coding: utf-8 -*-

from .quasi_newton_minimizer import QuasiNewtonMinimizer


class RelaxedNewton(QuasiNewtonMinimizer):
    def _get_descend_direction(self, energy):
        descend_direction = energy.curvature.inverse_times(energy.gradient)
        norm = descend_direction.norm()
        if norm != 1:
            return descend_direction / -norm
        else:
            return descend_direction * -1
