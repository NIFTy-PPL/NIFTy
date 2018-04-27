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

from __future__ import print_function
import numpy as np
import nifty4 as ift


IC = ift.GradientNormController(
        tol_abs_gradnorm=1e-6, iteration_limit=1000)
calls = 0


def mprint(*args):
    if ift.dobj.rank == 0:
        print(*args)


def test_rosenbrock_convex(minimizer):
    np.random.seed(42)
    space = ift.UnstructuredDomain((2,))
    starting_point = ift.Field.from_random('normal', domain=space)*10

    class RBEnergy(ift.Energy):
        def __init__(self, position, a=1., b=100.):
            super(RBEnergy, self).__init__(position)
            self.a = a
            self.b = b
            global calls
            calls += 1

        @property
        def value(self):
            x, y = self.position.to_global_data()
            return (self.a-x)*(self.a-x)+self.b*(y-x*x)*(y-x*x)

        @property
        def gradient(self):
            x, y = self.position.to_global_data()
            v0 = -2*(self.a-x)-4*self.b*x*(y-x*x)
            v1 = 2*self.b*(y-x*x)
            return ift.Field.from_global_data(space, np.array([v0, v1]))

        @property
        def curvature(self):
            class RBCurv(ift.EndomorphicOperator):
                def __init__(self, loc, a, b):
                    self._x, self._y = loc.to_global_data()
                    self.a = a
                    self.b = b

                @property
                def domain(self):
                    return space

                @property
                def capability(self):
                    return self.TIMES

                def apply(self, x, mode):
                    x = x.to_global_data()
                    v0 = (2+self.b*8*self._x**2)*x[0]
                    v0 -= self.b*4*self._x*x[1]
                    v1 = -self.b*4*self._x*x[0]
                    v1 += 2*self.b*x[1]
                    global calls
                    calls += 1
                    return ift.Field.from_global_data(space,
                                                      np.array([v0, v1]))
            t1 = ift.GradientNormController(tol_abs_gradnorm=1e-6,
                                            iteration_limit=1000)
            t2 = ift.ConjugateGradient(controller=t1)
            return ift.InversionEnabler(RBCurv(self._position, self.a, self.b),
                                        inverter=t2)

    energy = RBEnergy(position=starting_point)
    minimizer = minimizer(controller=IC)

    energy, convergence = minimizer(energy)
    return energy


def test_Ndim_rosenbrocklike_convex(minimizer, Ndim=3):
    np.random.seed(42)
    space = ift.UnstructuredDomain((Ndim,))
    starting_point = ift.Field.from_random('normal', domain=space)*10

    class RBLikeEnergy(ift.Energy):
        def __init__(self, position, a=1., b=100.):
            super(RBLikeEnergy, self).__init__(position)
            self.a = a
            self.b = b
            global calls
            calls += 1

        @property
        def value(self):
            x = self.position.to_global_data()
            t1 = self.a-x[0]
            t2 = x[1:]-x[:-1]**3
            return 0.5*t1**2+0.5*self.b*np.dot(t2, t2)

        @property
        def gradient(self):
            res = np.zeros(space.shape)
            x = self.position.to_global_data()
            t1 = self.a-x[0]
            t2 = x[1:]-x[:-1]**3
            res[0] = -t1
            res[1:] += self.b*t2
            res[:-1] += -3*self.b*x[:-1]**2*t2
            return ift.Field.from_global_data(space, res)

        @property
        def curvature(self):
            class RBCurv(ift.EndomorphicOperator):
                def __init__(self, loc, a, b):
                    self._x = loc.to_global_data()
                    self.a = a
                    self.b = b

                @property
                def domain(self):
                    return space

                @property
                def capability(self):
                    return self.TIMES

                def apply(self, z, mode):
                    y = z.to_global_data()
                    res = np.zeros(space.shape)
                    x = self._x
                    res[0] = y[0]
                    dt2 = y[1:] - 3*x[:-1]**2*y[:-1]
                    res[1:] += self.b*dt2
                    res[:-1] += -self.b*3*x[:-1]**2*dt2
                    global calls
                    calls += 1
                    return ift.Field.from_global_data(space, res)
            t1 = ift.GradientNormController(tol_abs_gradnorm=1e-6,
                                            iteration_limit=1000)
            t2 = ift.ConjugateGradient(controller=t1)
            return ift.InversionEnabler(RBCurv(self._position, self.a, self.b),
                                        inverter=t2)

    energy = RBLikeEnergy(position=starting_point)
    minimizer = minimizer(controller=IC)

    energy, convergence = minimizer(energy)
    return energy


def verbose_test(func, minimizer):
    mprint("Testing", minimizer)
    global calls
    calls = 0
    E = func(minimizer)
    mprint("Used", calls, "calls.")
    mprint("Final energy:", E.value)

if __name__ == "__main__":
    mprint("Standard Rosenbrock function\n")
    verbose_test(test_rosenbrock_convex, ift.Yango)
    verbose_test(test_rosenbrock_convex, ift.RelaxedNewton)
    verbose_test(test_rosenbrock_convex, ift.NonlinearCG)
    verbose_test(test_rosenbrock_convex, ift.L_BFGS)

    mprint("\nHigher-dimensional Rosenbrock function\n")
    verbose_test(test_Ndim_rosenbrocklike_convex, ift.Yango)
    verbose_test(test_Ndim_rosenbrocklike_convex, ift.RelaxedNewton)
    verbose_test(test_Ndim_rosenbrocklike_convex, ift.NonlinearCG)
    verbose_test(test_Ndim_rosenbrocklike_convex, ift.L_BFGS)
