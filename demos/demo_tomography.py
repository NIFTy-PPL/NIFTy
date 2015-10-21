# -*- coding: utf-8 -*-

from nifty import *

about.warnings.off()

shape = (256, 256)
x_space = rg_space(shape)
k_space = x_space.get_codomain()

power = lambda k: 42/((1+k*shape[0])**2)

S = power_operator(k_space, codomain=x_space, spec=power)
s = S.get_random_field(domain=x_space)


def make_los(n=10, angle=0, d=1):
    starts_list = []
    ends_list = []
    for i in xrange(n):
        starts_list += [[(-0.2)*d, (-0.2 + 1.2*i/n)*d]]
        ends_list += [[(1.2)*d, (-0.2 + 1.2*i/n)*d]]
    starts_list = np.array(starts_list)
    ends_list = np.array(ends_list)

    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    starts_list = rot_matrix.dot(starts_list.T-0.5*d).T+0.5*d
    ends_list = rot_matrix.dot(ends_list.T-0.5*d).T+0.5*d
    return (starts_list, ends_list)

temp_coords = (np.empty((0,2)), np.empty((0,2)))
n = 256
m = 256
for alpha in [np.pi/n*j for j in xrange(n)]:
    temp = make_los(n=m, angle=alpha)
    temp_coords = np.concatenate([temp_coords, temp], axis=1)

starts = list(temp_coords[0].T)
ends = list(temp_coords[1].T)

#n_points = 360.
#starts = [[(np.cos(i/n_points*np.pi)+1)*shape[0]/2.,
#           (np.sin(i/n_points*np.pi)+1)*shape[0]/2.] for i in xrange(int(n_points))]
#starts = list(np.array(starts).T)
#
#ends = [[(np.cos(i/n_points*np.pi + np.pi)+1)*shape[0]/2.,
#         (np.sin(i/n_points*np.pi + np.pi)+1)*shape[0]/2.] for i in xrange(int(n_points))]
#ends = list(np.array(ends).T)

R = los_response(x_space, starts=starts, ends=ends, sigmas_up=0.1, sigmas_low=0.1)
d_space = R.target

N = diagonal_operator(d_space, diag=s.var(), bare=True)
n = N.get_random_field(domain=d_space)

d = R(s) + n
j = R.adjoint_times(N.inverse_times(d))
D = propagator_operator(S=S, N=N, R=R)


m = D(j, W=S, tol=1E-14, limii=50, note=True)

s.plot(title="signal", save='1_plot_s.png')
s.plot(save='plot_s_power.png', power=True, other=power)
j.plot(save='plot_j.png')
#d_ = field(x_space, val=d.val, target=k_space)
#d_.plot(title="data", vmin=s.min(), vmax=s.max(), save='plot_d.png')
m.plot(title="reconstructed map", vmin=s.min(), vmax=s.max(), save='1_plot_m.png')
m.plot(title="reconstructed map", save='2_plot_m.png')
m.plot(save='plot_m_power.png', power=True, other=power)