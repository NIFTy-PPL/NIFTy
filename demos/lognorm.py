# %%
from jax.config import config

config.update("jax_enable_x64", True)

import sys
import matplotlib.pyplot as plt
from functools import partial
from jax import numpy as np
from jax import lax, random
from jax import value_and_grad, jit

import jifty1 as jft
from jifty1 import hmc

seed = 42
key = random.PRNGKey(seed)


# %%
def cartesian_product(arrays, out=None):
    import numpy as onp

    # Generalized N-dimensional products
    arrays = [onp.asarray(x) for x in arrays]
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    if out is None:
        out = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        out[..., i] = a
    return out.reshape(-1, la)


def helper_phi_b(b, x):
    return b * x[0] * np.exp(b * x[1])


# %%
b = 2.

signal_response = partial(helper_phi_b, b)
nll = jft.Gaussian(0., lambda x: x / np.sqrt(1.)) @ signal_response

ham = jft.StandardHamiltonian(nll)
ham = ham.jit()
ham_vg = jit(value_and_grad(ham))

# %%
n_pix_sqrt = 1000
x = np.linspace(-4, 4, n_pix_sqrt)
y = np.linspace(-4, 4, n_pix_sqrt)
xx = cartesian_product((x, y))
ham_everywhere = np.vectorize(ham, signature="(2)->()")(xx).reshape(
    n_pix_sqrt, n_pix_sqrt
)
plt.imshow(
    np.exp(-ham_everywhere.T),
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower"
)
plt.colorbar()
plt.title("target distribution")
plt.show()

# %%
n_mgvi_iterations = 30
n_samples = [2] * (n_mgvi_iterations - 10) + [2] * 5 + [10, 10, 10, 10, 100]
n_newton_iterations = [7] * (n_mgvi_iterations - 10) + [10] * 6 + 4 * [25]
absdelta = 1e-13

initial_position = np.array([1., 1.])
mkl_pos = 1e-2 * jft.Field(initial_position)

mgvi_positions = []

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mkl = jft.MetricKL(
        ham,
        mkl_pos,
        n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"absdelta": absdelta / 10.},
        hamiltonian_and_gradient=ham_vg
    )

    print("Minimizing...", file=sys.stderr)
    # TODO: Re-introduce a simplified version that works without fields
    opt_state = jft.minimize(
        None,
        x0=mkl_pos,
        method="newton-cg",
        options={
            "fun_and_grad": mkl.energy_and_gradient,
            "hessp": mkl.metric,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "cg_kwargs": {"name": None},
            "name": "N"
        }
    )
    mkl_pos = opt_state.x
    msg = f"Post MGVI Iteration {i}: Energy {ham(mkl_pos):2.4e}"
    print(msg, file=sys.stderr)
    mgvi_positions.append(mkl_pos)

# %%
n_geovi_iterations = 15
n_samples = [1] * (n_geovi_iterations - 10) + [2] * 5 + [10, 10, 10, 10, 100]
n_newton_iterations = [7] * (n_geovi_iterations - 10) + [10] * 6 + [25] * 4
absdelta = 1e-10

initial_position = np.array([1., 1.])
gkl_pos = 1e-2 * jft.Field(initial_position)

for i in range(n_geovi_iterations):
    print(f"geoVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    gkl = jft.GeoMetricKL(
        ham,
        gkl_pos,
        n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"absdelta": absdelta / 10.},
        non_linear_sampling_kwargs={
            "cg_kwargs": {
                "miniter": 0,
                "name": None
            },
            "maxiter": 20
        },
        hamiltonian_and_gradient=ham_vg
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=gkl_pos,
        method="newton-cg",
        options={
            "cg_kwargs": {
                "miniter": 0
            },
            "fun_and_grad": gkl.energy_and_gradient,
            "hessp": gkl.metric,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "cg_kwargs": {"name": None},
            "name": "N"
        }
    )
    gkl_pos = opt_state.x
    msg = f"Post geoVI Iteration {i}: Energy {ham(gkl_pos):2.4e}"
    print(msg, file=sys.stderr)

# %%
n_pix_sqrt = 200
x = np.linspace(-4.0, 4.0, n_pix_sqrt, endpoint=True)
y = np.linspace(-4.0, 4.0, n_pix_sqrt, endpoint=True)
X, Y = np.meshgrid(x, y)
XY = np.array([X, Y]).T
xy = XY.reshape((XY.shape[0] * XY.shape[1], 2))
es = np.exp(-lax.map(ham, xy)).reshape(XY.shape[:2]).T

# %%
mkl_b_space_smpls = np.array([(mkl_pos + smpl).val for smpl in mkl.samples])

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*mkl_b_space_smpls.T)
ax.plot(*mkl_pos, "rx")
plt.title("MGVI")
plt.show()

# %%
gkl_b_space_smpls = np.array([(gkl_pos + smpl).val for smpl in gkl.samples])

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*gkl_b_space_smpls.T)
ax.plot(*gkl_pos, "rx")
plt.title("GeoVI")
plt.show()

# %%
initial_position = np.array([1., 1.])

hmc_sampler = hmc.HMCChain(
    initial_position = 1e-2 * initial_position,
    potential_energy = ham,
    diag_mass_matrix = 1.,
    eps = 0.2,
    n_of_integration_steps = 64,
    rngseed = 42,
    compile = True,
    dbg_info = True,
)

(_last_pos, _key, hmc_samples, hmc_acceptance, hmc_unintegrated_momenta,
 hmc_momentum_samples, hmc_rejected_position_samples,
 hmc_rejected_momenta) = (hmc_sampler.generate_n_samples(100))

print(f"acceptance rate: {np.sum(hmc_acceptance)/len(hmc_acceptance)}")

# %%
b_space_smpls = hmc_samples
ax.scatter(*b_space_smpls.T)
#ax.plot(*gkl_pos, "rx")
plt.title("HMC (Metroplis-Hastings) samples")
plt.show()

# %%
initial_position = np.array([1., 1.])

hmc._DEBUG_TREE_END_IDXS = []
hmc._DEBUG_SUBTREE_END_IDXS = []
hmc._DEBUG_STORE = []

nuts_sampler = hmc.NUTSChain(
    initial_position = 1e-2 * initial_position,
    potential_energy = ham,
    diag_mass_matrix = 2.,
    eps = 0.2,
    maxdepth = 10,
    rngseed = 43,
    compile = True,
    dbg_info = True,
)

nuts_n_samples = []
ns_samples = [200, 1000, 1000000]
for n_samples in ns_samples:
    (_pos, _key, nuts_samples, nuts_momenta_before, nuts_momenta_after, nuts_depths,
     nuts_trees) = nuts_sampler.generate_n_samples(n_samples)
    nuts_n_samples.append(nuts_samples)

# %%
b_space_smpls = nuts_samples

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*b_space_smpls.T, s=2.)
#ax.plot(*gkl_pos, "rx")
plt.show()

# %%
plt.hist2d(*b_space_smpls.T, bins=[x,y], range=[[x.min(), x.max()], [y.min(), y.max()]])
plt.colorbar()

# %%
subplots = (3,2)
fig_width_pt = 426 # pt (a4paper, and such)
# fig_width_pt = 360 # pt
inches_per_pt = 1 / 72.27
fig_width_in = fig_width_pt * inches_per_pt
fig_height_in = fig_width_in * 1. * (subplots[0] / subplots[1])
fig_dims = (fig_width_in, fig_height_in)

fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(subplots[0], subplots[1], figsize=fig_dims, sharex=True, sharey=True)

ax1.set_title(r'$P(d=0|\xi_1, \xi_2) \cdot P(\xi_1, \xi_2)$')
xx = cartesian_product((x, y))
ham_everywhere = np.vectorize(ham, signature="(2)->()")(xx).reshape(
    n_pix_sqrt, n_pix_sqrt
)
ax1.imshow(
    np.exp(-ham_everywhere.T),
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower",
    #aspect='auto'
)
#ax1.colorbar()

ax1.set_ylim([-4., 4.])
ax1.set_xlim([-4., 4.])
#ax1.autoscale(enable=True, axis='y', tight=True) 
asp = float(np.diff(np.array(ax1.get_xlim()))[0] / np.diff(np.array(ax1.get_ylim()))[0])

smplmarkersize = .3
smplmarkercolor = 'k'

linewidths = 0.5
fontsize = 5
potlabels = False

ax2.set_title('MGVI')
mkl_b_space_smpls = np.array([(mkl_pos + smpl).val for smpl in mkl.samples])
contour = ax2.contour(X, Y, es, linewidths=linewidths)
ax2.clabel(contour, inline=True, fontsize=fontsize)
ax2.scatter(*mkl_b_space_smpls.T, s=smplmarkersize, c=smplmarkercolor)
ax2.plot(*mkl_pos, "rx")
#ax2.set_aspect(asp)

ax3.set_title('geoVI')
gkl_b_space_smpls = np.array([(gkl_pos + smpl).val for smpl in gkl.samples])
contour = ax3.contour(X, Y, es, linewidths=linewidths)
ax3.clabel(contour, inline=True, fontsize=fontsize)
ax3.scatter(*gkl_b_space_smpls.T, s=smplmarkersize, c=smplmarkercolor)
ax3.plot(*gkl_pos, "rx")
#ax3.set_aspect(asp)

for i in range(3):
    eval('ax' + str(i + 1)).set_ylabel(r'$\xi_2$')
ax3.set_xlabel(r'$\xi_1$')
ax6.set_xlabel(r'$\xi_1$')

for N, samples, ax in list(zip(ns_samples, nuts_n_samples, [ax4, ax5, ax6]))[:2]:
    ax.set_title(f"NUTS {N=}")
    contour = ax.contour(X, Y, es, linewidths=linewidths)
    #ax.clabel(contour, inline=True, fontsize=fontsize)
    ax.scatter(*samples.T, s=smplmarkersize, c=smplmarkercolor)

fig.tight_layout()
fig.savefig("pinch.pdf", bbox_inches='tight')

#
# NUTS HISTOGRAM
#

_h, _xedges, _yedges = np.histogram2d(*nuts_samples.T, bins=[x,y], range=[[x.min(), x.max()], [y.min(), y.max()]])

ax6.imshow(
    _h.T,
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower"
)

ax6.set_title(f'NUTS, {len(nuts_samples):.0E} samples')

fig.tight_layout()
fig.savefig("pinch.pdf", bbox_inches='tight')

print("final plot saved as pinch.pdf")