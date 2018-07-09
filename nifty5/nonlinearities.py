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

from __future__ import absolute_import, division, print_function
from .compat import *
from .sugar import full, exp, tanh


class Linear(object):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return full(x.domain, 1.)

    def hessian(self, x):
        return full(x.domain, 0.)


class Exponential(object):
    def __call__(self, x):
        return exp(x)

    def derivative(self, x):
        return exp(x)

    def hessian(self, x):
        return exp(x)


class Tanh(object):
    def __call__(self, x):
        return tanh(x)

    def derivative(self, x):
        return (1. - tanh(x)**2)

    def hessian(self, x):
        return - 2. * tanh(x) * (1. - tanh(x)**2)


class PositiveTanh(object):
    def __call__(self, x):
        return 0.5 * tanh(x) + 0.5

    def derivative(self, x):
        return 0.5 * (1. - tanh(x)**2)

    def hessian(self, x):
        return - tanh(x) * (1. - tanh(x)**2)
