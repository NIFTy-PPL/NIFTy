# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import nifty4 as ift
from itertools import product
from test.common import expand
from nose.plugins.skip import SkipTest

spaces = [ift.RGSpace([1024], distances=0.123), ift.HPSpace(32)]
minimizers = [ift.SteepestDescent, ift.RelaxedNewton, ift.VL_BFGS,
              ift.ConjugateGradient, ift.NonlinearCG,
              ift.NewtonCG, ift.L_BFGS_B]

minimizers2 = [ift.RelaxedNewton, ift.VL_BFGS, ift.NonlinearCG,
               ift.NewtonCG, ift.L_BFGS_B]


class Test_Minimizers(unittest.TestCase):

    @expand(product(minimizers, spaces))
    def test_quadratic_minimization(self, minimizer_class, space):
        np.random.seed(42)
        starting_point = ift.Field.from_random('normal', domain=space)*10
        covariance_diagonal = ift.Field.from_random(
                                  'uniform', domain=space) + 0.5
        covariance = ift.DiagonalOperator(covariance_diagonal)
        required_result = ift.Field.ones(space, dtype=np.float64)

        IC = ift.GradientNormController(tol_abs_gradnorm=1e-5,
                                        iteration_limit=1000)
        try:
            minimizer = minimizer_class(controller=IC)
            energy = ift.QuadraticEnergy(A=covariance, b=required_result,
                                         position=starting_point)

            (energy, convergence) = minimizer(energy)
        except NotImplementedError:
            raise SkipTest

        assert_equal(convergence, IC.CONVERGED)
        assert_allclose(energy.position.to_global_data(),
                        1./covariance_diagonal.to_global_data(),
                        rtol=1e-3, atol=1e-3)

    @expand(product(minimizers2))
    def test_rosenbrock(self, minimizer_class):
        try:
            from scipy.optimize import rosen, rosen_der, rosen_hess_prod
        except ImportError:
            raise SkipTest
        np.random.seed(42)
        space = ift.UnstructuredDomain((2,))
        starting_point = ift.Field.from_random('normal', domain=space)*10

        class RBEnergy(ift.Energy):
            def __init__(self, position):
                super(RBEnergy, self).__init__(position)

            @property
            def value(self):
                return rosen(self._position.to_global_data().copy())

            @property
            def gradient(self):
                inp = self._position.to_global_data().copy()
                out = ift.Field.from_global_data(space, rosen_der(inp))
                return out

            @property
            def curvature(self):
                class RBCurv(ift.EndomorphicOperator):
                    def __init__(self, loc):
                        self._loc = loc.to_global_data().copy()

                    @property
                    def domain(self):
                        return space

                    @property
                    def capability(self):
                        return self.TIMES

                    def apply(self, x, mode):
                        self._check_input(x, mode)
                        inp = x.to_global_data().copy()
                        out = ift.Field.from_global_data(
                            space, rosen_hess_prod(self._loc.copy(), inp))
                        return out

                t1 = ift.GradientNormController(tol_abs_gradnorm=1e-5,
                                                iteration_limit=1000)
                t2 = ift.ConjugateGradient(controller=t1)
                return ift.InversionEnabler(RBCurv(self._position),
                                            inverter=t2)

        IC = ift.GradientNormController(tol_abs_gradnorm=1e-5,
                                        iteration_limit=10000)
        try:
            minimizer = minimizer_class(controller=IC)
            energy = RBEnergy(position=starting_point)

            (energy, convergence) = minimizer(energy)
        except NotImplementedError:
            raise SkipTest

        assert_equal(convergence, IC.CONVERGED)
        assert_allclose(energy.position.to_global_data(), 1.,
                        rtol=1e-3, atol=1e-3)
