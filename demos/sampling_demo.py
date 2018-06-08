import nifty4 as ift
import numpy as np
import matplotlib.pyplot as plt
from nifty4.sugar import create_power_operator

np.random.seed(41)

x_space = ift.RGSpace(1024)
h_space = x_space.get_default_codomain()

d_space = x_space
N_hat = np.full(d_space.shape, .1)
N_hat[300:350] = 1e-13
N_hat = ift.Field.from_global_data(d_space, N_hat)
N = ift.DiagonalOperator(N_hat)

FFT = ift.HarmonicTransformOperator(h_space, x_space)
R = ift.ScalingOperator(1., x_space)


def ampspec(k): return 1. / (0.2 + k**2.)


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
N_iter = 300
IC = ift.GradientNormController(tol_abs_gradnorm=1e-5, iteration_limit=N_iter)
inverter = ift.ConjugateGradient(IC)
sampling_inverter = ift.ConjugateGradient(IC)
D_inv_1 = ift.SamplingEnabler(ift.SandwichOperator.make(R_p, N.inverse), S.inverse, inverter, sampling_inverter)


D_inv_2 = ift.SamplingEnabler2(D_inv, inverter, sampling_inverter)


samps_1 = [D_inv_1.draw_sample(from_inverse=True) for i in range(N_samps)] #GOOD
samps_2 = [D_inv_2.draw_sample(from_inverse=True) for i in range(N_samps)] #BAD

m = D_inv_1.inverse_times(j)
m_x = sky(m)


pltdict = {'alpha': .3, 'linewidth': .3}
for i in range(N_samps):
    if i == 0:
        plt.plot(sky(samps_2[i]).to_global_data()[290:360], color='b',
                 label='Default samples (residuals)',
                 **pltdict)
        plt.plot(sky(samps_1[i]).to_global_data()[290:360], color='r',
                 label='Conservative samples (residuals)',
                 **pltdict)
    else:
        plt.plot(sky(samps_2[i]).to_global_data()[290:360], color='b',
                 **pltdict)
        plt.plot(sky(samps_1[i]).to_global_data()[290:360], color='r', **pltdict)
plt.plot((s_x - m_x).to_global_data()[290:360], color='k',
         label='signal - mean')
plt.title('Comparison of conservative vs default samples near area of low noise')
plt.legend()
plt.savefig('residual_sample_comparison.png')
plt.close()

D_hat_old = ift.Field.full(x_space, 0.).to_global_data()
D_hat_new = ift.Field.full(x_space, 0.).to_global_data()
for i in range(N_samps):
    D_hat_old += sky(samps_2[i]).to_global_data()**2
    D_hat_new += sky(samps_1[i]).to_global_data()**2
plt.plot(np.sqrt(D_hat_old / N_samps), 'r--', label='Default uncertainty')
plt.plot(-np.sqrt(D_hat_old / N_samps), 'r--')
plt.fill_between(range(len(D_hat_new)), -np.sqrt(D_hat_new / N_samps), np.sqrt(
    D_hat_new / N_samps), facecolor='0.5', alpha=0.5,
    label='Conservative uncertainty')
plt.plot((s_x - m_x).to_global_data(), alpha=0.6, color='k',
         label='signal - mean')
plt.title('Comparison of uncertainty in position space')
plt.legend()
plt.savefig('uncertainty_x.png')
plt.close()

p_space = ift.PowerSpace(h_space)
p_new = ift.power_analyze(samps_1[0])
for i in range(1, N_samps):
    p_new += ift.power_analyze(samps_1[i])
p_new /= N_samps
p_old = ift.power_analyze(samps_2[0])
for i in range(1, N_samps):
    p_old += ift.power_analyze(samps_2[i])
p_old /= N_samps
plt.title("power_analyze")
plt.plot(p_old.to_global_data(), label='default')
plt.plot(p_new.to_global_data(), label='conservative')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('sample_mean_power.png')
plt.close()
