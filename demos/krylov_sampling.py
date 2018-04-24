import nifty4 as ift
import numpy as np
import matplotlib.pyplot as plt
from nifty4.sugar import create_power_operator

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
D_inv = ift.SandwichOperator(R_p, N.inverse) + S.inverse


N_samps = 200
N_iter = 100
IC = ift.GradientNormController(tol_abs_gradnorm=1e-3, iteration_limit=N_iter)
m, samps = ift.library.generate_krylov_samples(D_inv, S, j, N_samps, IC)
m_x = sky(m)
inverter = ift.ConjugateGradient(IC)
curv = ift.library.WienerFilterCurvature(S=S, N=N, R=R_p, inverter=inverter)
samps_old = [curv.draw_sample(from_inverse=True) for i in range(N_samps)]

plt.plot(d.to_global_data(), '+', label="data", alpha=.5)
plt.plot(s_x.to_global_data(), label="original")
plt.plot(m_x.to_global_data(), label="reconstruction")
plt.legend()
plt.savefig('Krylov_reconstruction.png')
plt.close()

pltdict = {'alpha': .3, 'linewidth': .2}
for i in range(N_samps):
    if i == 0:
        plt.plot(sky(samps_old[i]).to_global_data(), color='b',
                 label='Traditional samples (residuals)',
                 **pltdict)
        plt.plot(sky(samps[i]).to_global_data(), color='r',
                 label='Krylov samples (residuals)',
                 **pltdict)
    else:
        plt.plot(sky(samps_old[i]).to_global_data(), color='b', **pltdict)
        plt.plot(sky(samps[i]).to_global_data(), color='r', **pltdict)
plt.plot((s_x - m_x).to_global_data(), color='k', label='signal - mean')
plt.legend()
plt.savefig('Krylov_samples_residuals.png')
plt.close()

D_hat_old = ift.Field.zeros(x_space).to_global_data()
D_hat_new = ift.Field.zeros(x_space).to_global_data()
for i in range(N_samps):
    D_hat_old += sky(samps_old[i]).to_global_data()**2
    D_hat_new += sky(samps[i]).to_global_data()**2
plt.plot(np.sqrt(D_hat_old / N_samps), 'r--', label='Traditional uncertainty')
plt.plot(-np.sqrt(D_hat_old / N_samps), 'r--')
plt.fill_between(range(len(D_hat_new)), -np.sqrt(D_hat_new / N_samps), np.sqrt(
    D_hat_new / N_samps), facecolor='0.5', alpha=0.5,
    label='Krylov uncertainty')
plt.plot((s_x - m_x).to_global_data(), color='k', label='signal - mean')
plt.legend()
plt.savefig('Krylov_uncertainty.png')
plt.close()
