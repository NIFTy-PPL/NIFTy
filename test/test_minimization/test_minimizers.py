import unittest

import numpy as np
from numpy.testing import assert_allclose

import nifty as ift

from itertools import product
from test.common import expand

spaces = [ift.RGSpace([1024], distances=0.123), ift.HPSpace(32)]
minimizers = [ift.SteepestDescent, ift.RelaxedNewton, ift.VL_BFGS,
              ift.ConjugateGradient, ift.NonlinearCG]


class Test_Minimizers(unittest.TestCase):

    @expand(product(minimizers, spaces))
    def test_quadratic_minimization(self, minimizer_class, space):
        np.random.seed(42)
        starting_point = ift.Field.from_random('normal', domain=space)*10
        covariance_diagonal = ift.Field.from_random(
                                  'uniform', domain=space) + 0.5
        covariance = ift.DiagonalOperator(space, diagonal=covariance_diagonal)
        required_result = ift.Field(space, val=1.)

        IC = ift.DefaultIterationController(tol_abs_gradnorm=1e-5)
        minimizer = minimizer_class(controller=IC)
        energy = ift.QuadraticEnergy(A=covariance, b=required_result,
                                     position=starting_point)

        (energy, convergence) = minimizer(energy)
        assert convergence == IC.CONVERGED
        assert_allclose(energy.position.val.get_full_data(),
                        1./covariance_diagonal.val.get_full_data(),
                        rtol=1e-3, atol=1e-3)
