import nifty4 as ift
import numpy as np
import matplotlib.pyplot as plt
from nifty4.sugar import create_power_operator

np.random.seed(42)

x_space = ift.RGSpace(1024)
h_space = x_space.get_default_codomain()

d_space = x_space
N_hat = ift.Field(d_space, 10.)
N_hat.val[400:450] = 0.0001
N = ift.DiagonalOperator(N_hat, d_space)

FFT = ift.HarmonicTransformOperator(h_space, target=x_space)
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
D_inv = R_p.adjoint * N.inverse * R_p + S.inverse


def sample(D_inv, S, j,  N_samps, N_iter):
    space = D_inv.domain
    x = ift.Field.zeros(space)
    r = j.copy()
    p = r.copy()
    d = p.vdot(D_inv(p))
    y = []
    for i in range(N_samps):
        y += [S.draw_sample()]
    for k in range(1, 1 + N_iter):
        gamma = r.vdot(r) / d
        if gamma == 0.:
            break
        x += gamma * p
        print(p.vdot(D_inv(j)))
        for i in range(N_samps):
            y[i] -= p.vdot(D_inv(y[i])) * p / d
            y[i] += np.random.randn() / np.sqrt(d) * p
        #r_new = j - D_inv(x)
        r_new = r - gamma * D_inv(p)
        beta = r_new.vdot(r_new) / (r.vdot(r))
        r = r_new
        p = r + beta * p
        d = p.vdot(D_inv(p))
        if d == 0.:
            break
    return x, y


N_samps = 200
N_iter = 10
m, samps = sample(D_inv, S, j, N_samps, N_iter)
m_x = sky(m)
IC = ift.GradientNormController(iteration_limit=N_iter)
inverter = ift.ConjugateGradient(IC)
curv = ift.library.WienerFilterCurvature(S=S, N=N, R=R_p, inverter=inverter)
samps_old = []
for i in range(N_samps):
    samps_old += [curv.draw_sample(from_inverse=True)]

plt.plot(d.val, '+', label="data", alpha=.5)
plt.plot(s_x.val, label="original")
plt.plot(m_x.val, label="reconstruction")
plt.legend()
plt.savefig('Krylov_reconstruction.png')
plt.close()

pltdict = {'alpha': .3, 'linewidth': .2}
for i in range(N_samps):
    if i == 0:
        plt.plot(sky(samps_old[i]).val, color='b', **pltdict, label='Traditional samples (residuals)')
        plt.plot(sky(samps[i]).val, color='r', **pltdict, label='Krylov samples (residuals)')
    else:
        plt.plot(sky(samps_old[i]).val, color='b', **pltdict)
        plt.plot(sky(samps[i]).val, color='r', **pltdict)
plt.plot((s_x - m_x).val, color='k', label='signal - mean')
plt.legend()
plt.savefig('Krylov_samples_residuals.png')
plt.close()

D_hat_old = ift.Field.zeros(x_space).val
D_hat_new = ift.Field.zeros(x_space).val
for i in range(N_samps):
    D_hat_old += sky(samps_old[i]).val**2
    D_hat_new += sky(samps[i]).val**2
plt.plot(np.sqrt(D_hat_old / N_samps), 'r--', label='Traditional uncertainty')
plt.plot(-np.sqrt(D_hat_old / N_samps), 'r--')
plt.fill_between(range(len(D_hat_new)), -np.sqrt(D_hat_new / N_samps), np.sqrt(
    D_hat_new / N_samps), facecolor='0.5', alpha=0.5, label='Krylov unvertainty')
plt.plot((s_x - m_x).val, color='k', label='signal - mean')
plt.legend()
plt.savefig('Krylov_uncertainty.png')
plt.close()
