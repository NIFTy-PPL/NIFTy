## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
    ..                  __   ____   __
    ..                /__/ /   _/ /  /_
    ..      __ ___    __  /  /_  /   _/  __   __
    ..    /   _   | /  / /   _/ /  /   /  / /  /
    ..   /  / /  / /  / /  /   /  /_  /  /_/  /
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  demo
    ..                               /______/

    NIFTY demo applying a Wiener filter using explicit matrices.

"""
from __future__ import division
from nifty import *                                                   # version 0.7.0


# some signal space; e.g., a one-dimensional regular grid
x_space = rg_space([128,])                                               # define signal space

k_space = x_space.get_codomain()                                      # get conjugate space

# some power spectrum
power = (lambda k: 42 / (k + 1) ** 3)

S = power_operator(k_space, spec=power)                               # define signal covariance
s = S.get_random_field(domain=x_space)                                # generate signal
s -= s.val.mean()

R = response_operator(x_space, sigma=0.0, mask=1.0, assign=None)      # define response
d_space = R.target                                                    # get data space

# some noise variance
delta = 1000 * (np.arange(1, x_space.dim() + 1) / x_space.dim()) ** 5
N = diagonal_operator(d_space, diag=delta, bare=True)                 # define noise covariance
n = N.get_random_field(domain=d_space)                                # generate noise

d = R(s) + n                                                          # compute data

j = R.adjoint_times(N.inverse_times(d))                               # define information source


class M_operator(operator):

    def _multiply(self, x):
        N, R = self.para
        return R.adjoint_times(N.inverse_times(R.times(x)))


C = explicify(S, newdomain=x_space, newtarget=x_space)                # explicify S
M = M_operator(x_space, sym=True, uni=False, imp=True, para=(N, R))
M = explicify(M)                                                      # explicify M
D = (C.inverse() + M).inverse()                                       # define information propagator

m = D(j)                                                              # reconstruct map

vminmax = {"vmin":1.5 * s.val.min(), "vmax":1.5 * s.val.max()}
s.plot(title="signal", **vminmax)                                     # plot signal
d_ = field(x_space, val=d.val, target=k_space)
d_.plot(title="data", **vminmax)                                      # plot data
m.plot(title="reconstructed map", error=D.diag(bare=True), **vminmax) # plot map
D.plot(title="information propagator", bare=True)                     # plot information propagator

