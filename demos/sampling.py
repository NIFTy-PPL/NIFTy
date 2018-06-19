import nifty5 as ift
import numpy as np
import matplotlib.pyplot as plt
from nifty5.sugar import create_power_operator

np.random.seed(42)

x_space = ift.RGSpace(1024)
h_space = x_space.get_default_codomain()

d_space = x_space
N_hat = np.full(d_space.shape, 10.)
N_hat[400:450] = 0.0001
N_hat = ift.Field.from_global_data(d_space, N_hat)
N = ift.DiagonalOperator(N_hat)

FFT = ift.HarmonicTransformOperator(h_space, x_space)
R = ift.ScalingOperator(1., x_space)


def ampspec(k): return 1. / (1. + k**2.)


S = ift.ScalingOperator(1., h_space)
A = create_power_operator(h_space, ampspec)
s_h = S.draw_sample()
sky = FFT * A
s_x = sky(s_h)
n = N.draw_sample()
d = R(s_x) + n

R_p = R * FFT * A
j = R_p.adjoint(N.inverse(d))
D_inv = ift.SandwichOperator.make(R_p, N.inverse) + S.inverse


N_samps = 200
N_iter = 100

tol = 1e-3
IC = ift.GradientNormController(tol_abs_gradnorm=tol, iteration_limit=N_iter)
inverter = ift.ConjugateGradient(IC)
curv = ift.library.WienerFilterCurvature(S=S, N=N, R=R_p, inverter=inverter,
                                         sampling_inverter=inverter)
m_xi = curv.inverse_times(j)
samps_long = [curv.draw_sample(from_inverse=True) for i in range(N_samps)]

tol = 1e2
IC = ift.GradientNormController(tol_abs_gradnorm=tol, iteration_limit=N_iter)
inverter = ift.ConjugateGradient(IC)
curv = ift.library.WienerFilterCurvature(S=S, N=N, R=R_p, inverter=inverter,
                                         sampling_inverter=inverter)
samps_short = [curv.draw_sample(from_inverse=True) for i in range(N_samps)]

# Compute mean
sc = ift.StatCalculator()
for samp in samps_long:
    sc.add(samp)
m_x = sky(sc.mean + m_xi)

plt.plot(d.to_global_data(), '+', label="data", alpha=.5)
plt.plot(s_x.to_global_data(), label="original")
plt.plot(m_x.to_global_data(), label="reconstruction")
plt.legend()
plt.savefig('reconstruction.png')
plt.close()

pltdict = {'alpha': .3, 'linewidth': .2}
for i in range(N_samps):
    if i == 0:
        plt.plot(sky(samps_short[i]).to_global_data(), color='b',
                 label='Short samples (residuals)',
                 **pltdict)
        plt.plot(sky(samps_long[i]).to_global_data(), color='r',
                 label='Long samples (residuals)',
                 **pltdict)
    else:
        plt.plot(sky(samps_short[i]).to_global_data(), color='b', **pltdict)
        plt.plot(sky(samps_long[i]).to_global_data(), color='r', **pltdict)
plt.plot((s_x - m_x).to_global_data(), color='k', label='signal - mean')
plt.legend()
plt.savefig('samples_residuals.png')
plt.close()

D_hat_old = ift.full(x_space, 0.).to_global_data()
D_hat_new = ift.full(x_space, 0.).to_global_data()
for i in range(N_samps):
    D_hat_old += sky(samps_short[i]).to_global_data()**2
    D_hat_new += sky(samps_long[i]).to_global_data()**2
plt.plot(np.sqrt(D_hat_old / N_samps), 'r--', label='Short uncertainty')
plt.plot(-np.sqrt(D_hat_old / N_samps), 'r--')
plt.fill_between(range(len(D_hat_new)), -np.sqrt(D_hat_new / N_samps), np.sqrt(
    D_hat_new / N_samps), facecolor='0.5', alpha=0.5,
    label='Long uncertainty')
plt.plot((s_x - m_x).to_global_data(), color='k', label='signal - mean')
plt.legend()
plt.savefig('uncertainty.png')
plt.close()
