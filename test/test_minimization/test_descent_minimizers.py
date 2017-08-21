import unittest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from nifty import Field, DiagonalOperator, RGSpace, HPSpace
from nifty import SteepestDescent, RelaxedNewton, VL_BFGS

from itertools import product
from test.common import expand

from quadratic_potential import QuadraticPotential

from nifty import logger

minimizers = [SteepestDescent, RelaxedNewton, VL_BFGS]
spaces = [RGSpace([1024, 1024], distances=0.123), HPSpace(32)]


class Test_DescentMinimizers(unittest.TestCase):

    @expand([[minimizer] for minimizer in minimizers])
    def test_interface(self, minimizer):
        iteration_limit = 100
        convergence_level = 4
        convergence_tolerance = 1E-6
        callback = lambda z: z
        minimizer = minimizer(iteration_limit=iteration_limit,
                              convergence_tolerance=convergence_tolerance,
                              convergence_level=convergence_level,
                              callback=callback)

        assert_equal(minimizer.iteration_limit, iteration_limit)
        assert_equal(minimizer.convergence_level, convergence_level)
        assert_equal(minimizer.convergence_tolerance, convergence_tolerance)
        assert(minimizer.callback is callback)

    @expand(product(minimizers, spaces))
    def test_minimization(self, minimizer_class, space):
        np.random.seed(42)
        starting_point = Field.from_random('normal', domain=space)*10
        covariance_diagonal = Field.from_random('uniform', domain=space) + 0.5
        covariance = DiagonalOperator(space, diagonal=covariance_diagonal)
        energy = QuadraticPotential(position=starting_point,
                                    eigenvalues=covariance)
        minimizer = minimizer_class(iteration_limit=30,
                                    convergence_tolerance=1e-10)

        (energy, convergence) = minimizer(energy)

        assert_almost_equal(energy.value, 0, decimal=5)
        assert_almost_equal(energy.position.val.get_full_data(), 0., decimal=5)
