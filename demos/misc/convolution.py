import numpy as np
import nifty5 as ift


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
    plot.add(signal, title='Signal')
    plot.add(conv_signal, title='Signal Convolved')
    plot.add(cac_signal, title='Signal, Conv, Adj-Conv')
    plot.add(conv_delta, title='Kernel')
    plot.output()


# Healpix test
nside = 64
npix = 12 * nside * nside

domain = ift.HPSpace(nside)

# Define test signal (some point sources)
signal_vals = np.zeros(npix, dtype=np.float64)
for i in range(0, npix, npix//12 + 27):
    signal_vals[i] = 500.

signal = ift.from_global_data(domain, signal_vals)

delta_vals = np.zeros(npix, dtype=np.float64)
delta_vals[0] = 1.0
delta = ift.from_global_data(domain, delta_vals)


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
signal = ift.from_global_data(domain, signal_vals)

# Define delta signal, generate kernel image
delta_vals = np.zeros(domain.shape, dtype=np.float64)
delta_vals[0, 0] = 1.0
delta = ift.from_global_data(domain, delta_vals)


# Define kernel function
def func(dist):
    return 1. * np.logical_and(dist > 0.1, dist <= 0.2)


convtest(signal, delta, func)
