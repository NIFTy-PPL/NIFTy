# -*- coding: utf-8 -*-

from .quasi_newton_minimizer import QuasiNewtonMinimizer


class SteepestDescent(QuasiNewtonMinimizer):
    def _get_descend_direction(self, gradient, gradient_norm):
        return gradient/(-gradient_norm)
