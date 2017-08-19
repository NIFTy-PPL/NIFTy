import unittest

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from nifty import Field, DiagonalOperator, RGSpace, HPSpace
from nifty import ConjugateGradient, QuadraticEnergy

from test.common import expand

spaces = [RGSpace([1024, 1024], distances=0.123), HPSpace(32)]


class Test_ConjugateGradient(unittest.TestCase):

    def test_interface(self):
        iteration_limit = 100
        convergence_level = 4
        convergence_tolerance = 1E-6
        callback = lambda z: z
        minimizer = ConjugateGradient(
                                iteration_limit=iteration_limit,
                                convergence_tolerance=convergence_tolerance,
                                convergence_level=convergence_level,
                                callback=callback)

        assert_equal(minimizer.iteration_limit, iteration_limit)
        assert_equal(minimizer.convergence_level, convergence_level)
        assert_equal(minimizer.convergence_tolerance, convergence_tolerance)
        assert(minimizer.callback is callback)

    @expand([[space] for space in spaces])
    def test_minimization(self, space):
        np.random.seed(42)
        starting_point = Field.from_random('normal', domain=space)*10
        covariance_diagonal = Field.from_random('uniform', domain=space) + 0.5
        covariance = DiagonalOperator(space, diagonal=covariance_diagonal)
        required_result = Field(space, val=1.)

        minimizer = ConjugateGradient()
        energy = QuadraticEnergy(A=covariance, b=required_result,
                                 position=starting_point)

        (energy, convergence) = minimizer(energy)

        assert_allclose(energy.position.val.get_full_data(),
                            1./covariance_diagonal.val.get_full_data(),
                            rtol=1e-3)
