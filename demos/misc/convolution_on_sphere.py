import numpy as np
import nifty5 as ift

# Define domains
nside = 64
npix = 12 * nside * nside

domain = ift.HPSpace(nside)
dom_tuple = ift.DomainTuple.make(domain)
codom = domain.get_default_codomain()

# Define test signal (some point sources)
signal_vals = np.zeros(npix, dtype=np.float64)
for i in range(0, npix, npix//12 + 27):
    signal_vals[i] = 1.
signal = ift.from_global_data(dom_tuple, signal_vals)


# Define kernel function
def func(theta):
    ct = np.cos(theta)
    return 1. * np.logical_and(ct > 0.7, ct <= 0.8)


# Create Convolution Operator
conv_op = ift.SphericalFuncConvolutionOperator(dom_tuple, func)

# Convolve, Adjoint-Convolve
conv_signal = conv_op(signal)
cac_signal = conv_op.adjoint_times(conv_signal)

print(signal.integrate(), conv_signal.integrate(), cac_signal.integrate())

# Define delta signal, generate kernel image
delta_vals = np.zeros(npix, dtype=np.float64)
delta_vals[0] = 1.0
delta = ift.from_global_data(domain, delta_vals)
conv_delta = conv_op(delta)

# Plot results
plot = ift.Plot()
plot.add(signal, title='Signal')
plot.add(conv_signal, title='Signal Convolved')
plot.add(cac_signal, title='Signal, Conv, Adj-Conv')
plot.add(conv_delta, title='Kernel')
plot.output()
