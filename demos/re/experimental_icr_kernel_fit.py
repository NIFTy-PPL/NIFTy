# %%
from typing import Optional

import jax
import nifty8.re as jft
import numpy as np
import jax.numpy as jnp
import healpy as hp
import matplotlib.pyplot as plt
from functools import partial
from nifty8.re.num.unique import amend_unique_
from nifty8.re.refine.util import (
    get_cov_from_loc,
    refinement_matrices,
    RefinementMatrices,
)
from nifty8.re.refine.healpix_field import cov_sqrt_hp
from jax import random, config, vmap
from jax.tree_util import tree_map

config.update("jax_enable_x64", True)


def _matrices_tol_prep(
    chart,
    depth: int,
    *,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    mat_buffer_size: Optional[int] = None,
    _verbosity: int = 0,
):
    """Basically copy of nifty8.re.refine.healpix_field._matrices_tol

    Builds unique indices and index maps using the local distance matrix of each
    refinment step. Instead of evaluating the kernel returns the map and unique
    distances to be used for evaluation later.
    """
    dist_mat_from_loc = get_cov_from_loc(lambda x: x, None)
    if atol is not None and rtol is not None and mat_buffer_size is None:
        raise TypeError("must specify `mat_buffer_size`")

    def dist_mat(lvl, idx_hp, idx_r):
        # `idx_r` is the left-most radial pixel of the to-be-refined slice
        # Extend `gc` and `gf` radially
        gc, gf = chart.get_coarse_fine_pair((idx_hp, idx_r), lvl)
        assert gf.shape[0] == chart.fine_size ** (chart.ndim + 1)

        coord = jnp.concatenate((gc, gf), axis=0)
        return dist_mat_from_loc(coord, coord)

    idx_map, dists = [], []
    for lvl in range(depth):
        pix_hp_idx = jnp.arange(chart.shape_at(lvl)[0])
        assert chart.ndim == 2
        pix_r_off = jnp.arange(chart.shape_at(lvl)[1] - chart.coarse_size + 1)
        # Map only over the radial axis b/c that is irregular anyways simply
        # due to the HEALPix geometry and manually scan over the HEALPix axis
        # to only save unique values
        vdist = jax.vmap(partial(dist_mat, lvl), in_axes=(None, 0))

        # Successively amend the duplicate-free distance/covariance matrices
        d = jax.eval_shape(vdist, pix_hp_idx[0], pix_r_off)
        u = jnp.full((mat_buffer_size,) + d.shape, jnp.nan, dtype=d.dtype)

        def scanned_amend_unique(u, pix):
            d = vdist(pix, pix_r_off)
            if _verbosity > 1:
                # Magic code to move up curser by one line and delete whole line
                msg = "\x1b[1A\x1b[2K{pix}/{n}"
                jax.debug.print(msg, pix=pix, n=pix_hp_idx[-1])
            return amend_unique_(u, d, axis=0, atol=atol, rtol=rtol)

        if _verbosity > 1:
            jax.debug.print("")
        u, inv = jax.lax.scan(scanned_amend_unique, u, pix_hp_idx)
        # Cut away the placeholder for preserving static shapes
        n = np.unique(inv).size
        if n >= u.shape[0] or not np.all(np.isnan(u[n:])):
            raise ValueError("`mat_buffer_size` too small")
        u = u[:n]
        dists.append(u)
        idx_map.append(inv)

    return idx_map, dists

def eval_kernel(kerfunc, dists, idx_map, chart):
    """Evaluates kernel at unique distances builds `RefinementMatrices`

    Using the index map and unique distance matrices obtained from
    `_matrices_tol_prep`, evaluate `kernelfunc` and build refinement matrices.
    `Idx_map` is then used to broadcast matrices to all indices on grid during
    application.
    """
    def ref_mat(cov):
        olf, ks = refinement_matrices(
            cov,
            chart.fine_size ** (chart.ndim + 1),
            coerce_fine_kernel=False,
        )
        if chart.ndim > 1:
            olf = olf.reshape(
                chart.fine_size**2,
                chart.fine_size,
                chart.coarse_size**2,
                chart.coarse_size,
            )
        return olf, ks

    opt_lin_filter, kernel_sqrt = [], []
    for uu in dists:
        # Finally, all distance/covariance matrices are assembled and we
        # can map over them to construct the refinement matrices as usual
        vmat = vmap(vmap(ref_mat, in_axes=(0,)), in_axes=(0,))
        u = kerfunc(uu)
        olf, ks = vmat(u)

        opt_lin_filter.append(olf)
        kernel_sqrt.append(ks)

    cov_sqrt0 = cov_sqrt_hp(chart, kerfunc)
    return RefinementMatrices(opt_lin_filter, kernel_sqrt, cov_sqrt0, idx_map)


class VariableICR(jft.Model):
    def __init__(
        self,
        rField: jft.ChartedHPField,
        kernelparams: jft.Model,
        kernelfunc: callable,
        exkey: str = "xi",
        atol: float = 0.001,
        rtol: float = 0.01,
        mat_buffer_size: int = 1000,
    ):
        self._rf = rField
        self._kernel = kernelparams
        self._kfunc = kernelfunc
        self._chart = rField.chart
        self._exkey = str(exkey)
        rfinit = jft.Initializer(
            {
                self._exkey: tree_map(
                    lambda x: partial(jft.random_like, primals=x), self._rf.domain
                )
            }
        )
        idx_map, dists = _matrices_tol_prep(
            self.chart,
            self.chart.depth,
            atol=atol,
            rtol=rtol,
            mat_buffer_size=mat_buffer_size,
        )
        self._mats = partial(
            eval_kernel, dists=dists, idx_map=idx_map, chart=self._chart
        )
        super().__init__(init=self._kernel.init | rfinit)

    @property
    def chart(self):
        return self._chart

    def __call__(self, x):
        params = self._kernel(x)
        mats = self._mats(partial(self._kfunc, **params))
        return self._rf(x[self._exkey], mats)


# Parameter model for toy kernel
class Kerparams(jft.Model):
    def __init__(self):
        self._a = jft.LogNormalPrior(1.0, 0.5, name="a")
        self._sig = jft.LogNormalPrior(1.0, 0.5, name="sig")
        super().__init__(init=self._a.init | self._sig.init)

    def __call__(self, x):
        return {"a": self._a(x), "sig": self._sig(x)}


class ExpSignal(jft.Model):
    def __init__(self, signal):
        self.signal = signal
        super().__init__(self, init=self.signal.init)

    def __call__(self, x):
        return jnp.exp(self.signal(x))


class RandomResponse(jft.Model):
    def __init__(self, signal, key, frac=0.2):
        assert (frac < 1) and (frac > 0)
        totpix = jft.size(signal.target)
        npix = int(frac * totpix)
        self._pix = random.choice(key, totpix, (npix,), replace=False)
        self.signal = signal
        super().__init__(init=self.signal.init)

    def __call__(self, x):
        return self.signal(x).ravel()[self._pix]


# %%
# Define chart
nside = 8
min_size = 10
noiselevel = 0.1
scl = 1.6
idx0 = -1.4
key = random.PRNGKey(42)


def rg2cart(x, idx0, scl):
    """Transforms regular, points from a Euclidean space to irregular points in
    an cartesian coordinate system in 1D."""
    return jnp.exp(scl * x[0] + idx0)[np.newaxis, ...]


def cart2rg(x, idx0, scl):
    """Inverse of `rg2cart`."""
    return ((jnp.log(x[0]) - idx0) / scl)[np.newaxis, ...]

cc = jft.HEALPixChart(
    min_shape=(
        hp.nside2npix(nside),
        min_size,
    ),
    nonhp_rg2cart=partial(rg2cart, idx0=idx0, scl=scl),
    nonhp_cart2rg=partial(cart2rg, idx0=idx0, scl=scl),
    _coarse_size=3,
    _fine_size=2,
)
print(cc.shape)
rf = jft.ChartedHPField(cc)

# %%
# Model for kernel
params = Kerparams()


# Toy Gaussian kernel
def mykernel(dist, a, sig):
    return a * jnp.exp(-0.5 * (dist / sig) ** 2)


model = VariableICR(rf, params, mykernel)


key, responsek, noisek, gtk = random.split(key, 4)

# Signal, response, data, lh
mfrac = 0.2
signal = ExpSignal(model)
signal_response = RandomResponse(signal, responsek, frac=mfrac)

gtxi = signal_response.init(gtk)
gt = signal_response(gtxi)


Ninv = lambda x: x / noiselevel**2
Nsqinv = lambda x: x / noiselevel

data = gt + noiselevel * jft.random_like(noisek, signal_response.target)
lh = jft.Gaussian(data=data, noise_cov_inv=Ninv, noise_std_inv=Nsqinv)
lh = lh.amend(signal_response)

print(np.min(data), np.max(data))
print(np.min(gt), np.max(gt))

# %%
# Fit data and kernel
n_vi_iterations = 6
delta = 1e-5
n_samples = 4

key, initk, samplek = random.split(key, 3)
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(initk)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    key=samplek,
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M",
            absdelta=delta * jft.size(lh.domain),
            cg_kwargs=dict(name="MCG"),
            maxiter=20,
        )
    ),
    sample_mode="linear_resample",
    odir="results_icr",
    resume=False,
)

# %%
# Output Plotting
synth_density = signal(gtxi)
post_density = vmap(signal)(samples.samples)
j = jnp.zeros_like(synth_density).ravel()
j = j.at[signal_response._pix].set(1.0)
j = j.reshape(synth_density.shape)

m = post_density.mean(axis=0)
sa = post_density[0]

fig = plt.figure(figsize=(50, 10))
vmin = min(m.min() for m in synth_density.T)
vmax = max(m.max() for m in synth_density.T)

for i, (g, r, ss, jj) in enumerate(zip(synth_density.T, m.T, sa.T, j.T)):
    hp.mollview(
        g,
        title=f"Gt {i+1}",
        fig=fig.number,
        sub=(4, synth_density.shape[1], i + 1),
        cmap="viridis",
        min=vmin,
        max=vmax,
        nest=True,
        notext=True,
        cbar=False,
    )
    hp.mollview(
        r,
        title=f"Mean {i+1}",
        fig=fig.number,
        sub=(4, synth_density.shape[1], i + 1 + synth_density.shape[1]),
        cmap="viridis",
        min=vmin,
        max=vmax,
        nest=True,
        notext=True,
        cbar=False,
    )
    hp.mollview(
        ss,
        title=f"Samp {i+1}",
        fig=fig.number,
        sub=(4, synth_density.shape[1], i + 1 + 2 * synth_density.shape[1]),
        cmap="viridis",
        min=vmin,
        max=vmax,
        nest=True,
        notext=True,
        cbar=False,
    )
    hp.mollview(
        jj,
        title=f"R.T {i+1} (f={mfrac})",
        fig=fig.number,
        sub=(
            4,
            synth_density.shape[1],
            i + 1 + 3 * synth_density.shape[1],
        ),  # (nrows, ncols, index)
        cmap="viridis",
        nest=True,
        notext=True,
        cbar=False,
    )

gtparams = params(gtxi)
sampparams = list(params(s) for s in samples)
mean, std = jft.mean_and_std(sampparams)
print(gtparams)
print(jft.mean_and_std(sampparams))
fig.suptitle(
    (
        f'a: {gtparams['a']:.2f} vs {mean['a']:.2f} pm {std['a']:.2f}'
        + f'sig: {gtparams['sig']:.2f} vs {mean['sig']:.2f} pm {std['sig']:.2f}'
    )
)
plt.savefig("results_icr/results.png", dpi=300)
plt.show()

# %%
