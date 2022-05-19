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
# Copyright(C) 2019-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

import nifty8 as ift


def convtest(test_signal, delta, func):
    domain = test_signal.domain

    # Create Convolution Operator
    conv_op = ift.FuncConvolutionOperator(domain, func)

    # Convolve, Adjoint-Convolve
    conv_signal = conv_op(test_signal)
    cac_signal = conv_op.adjoint_times(conv_signal)

    print(test_signal.integrate(), conv_signal.integrate(),
          cac_signal.integrate())

    # generate kernel image
    conv_delta = conv_op(delta)

    # Plot results
    plot = ift.Plot()
    plot.add(test_signal, title='Signal')
    plot.add(conv_signal, title='Signal Convolved')
    plot.add(cac_signal, title='Signal, Conv, Adj-Conv')
    plot.add(conv_delta, title='Kernel')
    plot.output()


def main():
    # Healpix test
    nside = 64
    npix = 12 * nside * nside

    domain = ift.HPSpace(nside)

    # Define test signal (some point sources)
    signal_vals = np.zeros(npix, dtype=np.float64)
    for i in range(0, npix, npix//12 + 27):
        signal_vals[i] = 500.

    signal = ift.makeField(domain, signal_vals)

    delta_vals = np.zeros(npix, dtype=np.float64)
    delta_vals[0] = 1.0
    delta = ift.makeField(domain, delta_vals)

    # Define kernel function
    def func(theta):
        ct = np.cos(theta)
        return 1. * np.logical_and(ct > 0.7, ct <= 0.8)

    convtest(signal, delta, func)

    domain = ift.RGSpace((100, 100))
    # Define test signal (some point sources)
    signal_vals = np.zeros(domain.shape, dtype=np.float64)
    signal_vals[35, 70] = 5000.
    signal_vals[45, 8] = 5000.
    signal = ift.makeField(domain, signal_vals)

    # Define delta signal, generate kernel image
    delta_vals = np.zeros(domain.shape, dtype=np.float64)
    delta_vals[0, 0] = 1.0
    delta = ift.makeField(domain, delta_vals)

    convtest(signal, delta, lambda d: 1. * np.logical_and(d > 0.1, d <= 0.2))


if __name__ == '__main__':
    main()
