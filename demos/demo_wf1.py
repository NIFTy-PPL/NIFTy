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

    NIFTY demo applying a Wiener filter using conjugate gradient.

"""
from __future__ import division
from nifty import *                                              # version 0.8.0


# some signal space; e.g., a two-dimensional regular grid
x_space = rg_space([128, 128])                                   # define signal space
#x_space = rg_space(512)
#x_space = hp_space(32)
#x_space = gl_space(96)

k_space = x_space.get_codomain()                                 # get conjugate space

# some power spectrum
power = (lambda k: 42 / (k + 1) ** 3)

S = power_operator(k_space, spec=power)                          # define signal covariance
s = S.get_random_field(domain=x_space)                           # generate signal

R = response_operator(x_space, sigma=0.0, mask=1.0, assign=None) # define response
d_space = R.target                                               # get data space

# some noise variance; e.g., signal-to-noise ratio of 1
N = diagonal_operator(d_space, diag=s.var(), bare=True)          # define noise covariance
n = N.get_random_field(domain=d_space)                           # generate noise

d = R(s) + n                                                     # compute data

j = R.adjoint_times(N.inverse_times(d))                          # define information source
D = propagator_operator(S=S, N=N, R=R)                           # define information propagator

m = D(j, W=S, tol=1E-3, note=True)                               # reconstruct map

s.plot(title="signal", save = 'plot_s.png')                                           # plot signal
d_ = field(x_space, val=d.val, target=k_space)
d_.plot(title="data", vmin=s.min(), vmax=s.max(), save = 'plot_d.png')                # plot data
m.plot(title="reconstructed map", vmin=s.min(), vmax=s.max(), save = 'plot_m.png')    # plot map

