# -*- coding: utf-8 -*-

from .quasi_newton_minimizer import QuasiNewtonMinimizer
from .line_searching import LineSearchStrongWolfe


class RelaxedNewton(QuasiNewtonMinimizer):
    def __init__(self, line_searcher=LineSearchStrongWolfe(), callback=None,
                 convergence_tolerance=1E-4, convergence_level=3,
                 iteration_limit=None):
        super(RelaxedNewton, self).__init__(
                                line_searcher=line_searcher,
                                callback=callback,
                                convergence_tolerance=convergence_tolerance,
                                convergence_level=convergence_level,
                                iteration_limit=iteration_limit)

        self.line_searcher.prefered_initial_step_size = 1.

    def _get_descend_direction(self, energy):
        gradient = energy.gradient
        curvature = energy.curvature
        descend_direction = curvature.inverse_times(gradient)
        return descend_direction * -1
        #norm = descend_direction.norm()
#        if norm != 1:
#            return descend_direction / -norm
#        else:
#            return descend_direction * -1
