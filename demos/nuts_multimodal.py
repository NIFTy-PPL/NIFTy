# %%
from functools import partial
import jax.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import jifty1 as jft
from jifty1 import hmc

def loggaussian(x, mu, sigma):
    return -0.5 * (x-mu)**2 / sigma

def sum_of_gaussians(x, separation, sigma1, sigma2):
    return - np.logaddexp(loggaussian(x, 0, sigma1), loggaussian(x, separation, sigma2))

ham = partial(sum_of_gaussians, separation = 10., sigma1 = 1., sigma2 = 1.)

N = 100000
SEED = 43
EPS = 0.3

subplots = (2,2)
fig_width_pt = 426 # pt (a4paper, and such)
# fig_width_pt = 360 # pt
inches_per_pt = 1 / 72.27
fig_width_in = 0.9 * fig_width_pt * inches_per_pt
fig_height_in = fig_width_in * 0.618 * (subplots[0] / subplots[1])
fig_dims = (fig_width_in, fig_height_in*1.5)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(subplots[0], subplots[1], sharex='col', figsize=fig_dims, gridspec_kw={'width_ratios': [1, 2]})

nuts_sampler = hmc.NUTSChain(
    initial_position = np.array(3.),
    potential_energy = ham,
    diag_mass_matrix = .2,
    eps = EPS,
    maxdepth = 15,
    rngseed = SEED,
    compile = True,
    dbg_info = True
)

(_last_pos, _key, nuts_pos, nuts_unintegrated_momenta, nuts_momentum_samples,
 nuts_depths, nuts_trees) = (nuts_sampler.generate_n_samples(N))

nuts_sampler.plot_1d_hist(ax1, bins=30, density=True)

nuts_sampler.plot_response_ts(ax2, linewidth=0.5)

ax1.set_title(rf'$m={nuts_sampler.diag_mass_matrix:1.2f}$')
ax2.set_title(rf'$m={nuts_sampler.diag_mass_matrix:1.2f}$')

nuts_sampler = hmc.NUTSChain(
    initial_position = np.array(3.),
    potential_energy = ham,
    diag_mass_matrix = .02,
    eps = EPS,
    maxdepth = 15,
    rngseed = SEED,
    compile = True,
    dbg_info = True
)

(_last_pos, _key, nuts_pos, nuts_unintegrated_momenta, nuts_momentum_samples,
 nuts_depths, nuts_trees) = (nuts_sampler.generate_n_samples(N))

nuts_sampler.plot_1d_hist(ax3, bins=30, density=True)

nuts_sampler.plot_response_ts(ax4, linewidth=0.5)

ax3.set_title(rf'$m={nuts_sampler.diag_mass_matrix:1.2f}$')
ax4.set_title(rf'$m={nuts_sampler.diag_mass_matrix:1.2f}$')

xs = np.linspace(-10, 20, num=500)
Z = np.trapz(np.exp(-ham(xs)), xs)
ax1.plot(xs, np.exp(-ham(xs)) / Z, linewidth=0.5, c='r')
ax3.plot(xs, np.exp(-ham(xs)) / Z, linewidth=0.5, c='r')

ax1.set_ylabel('frequency')
ax2.set_ylabel('position')
ax3.set_xlabel('position')
ax3.set_ylabel('frequency')
ax4.set_xlabel('time')
ax4.set_ylabel('position')

#fig.suptitle("sum of two Gaussians, with different choices of mass matrix")

fig.tight_layout()

fig.savefig("multimodal.pdf", bbox_inches='tight')

print("final figure saved as multimodal.pdf")