# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

"""Fisher metric and NIFTy.re Likelihood for the noncentral chi-squared, any k.

The data are independent draws, for every integer k >= 1, with known
component variance sigma2:

    data / sigma2  ~  ncx2(k, lam),      lam = model / sigma2

Units: `data`, `sigma2` and the primals (the noise-free model powers) are all
in the same physical power units; the conversion into the dimensionless ncx2
units d = data / sigma2 and lam = model / sigma2 happens inside the likelihood.

Energy: the energy is -sum log f_k(d; lam), the negative log-density of the
rescaled data.  The Jacobian log(1/sigma2) of the rescaling data -> d does not
depend on the model and is dropped, so the energy differs from the negative
log-likelihood of `data` by a constant.

Fisher information
------------------
From the Poisson-mixture representation, df/dlam = (f_{k+2} - f_k)/2, hence

    d/dlam log f_k = (f_{k+2}/f_k - 1) / 2,
    F(lam) = 1/4 * Int f_k * (f_{k+2}/f_k - 1)**2 dd,

computed by quadrature with the densities at k and k + 2.  Anchors:

    F(0)        = 1/(2k)                                     exact
    F(lam->inf) = 1/(2(k+2lam)) + 1/(2(k+2lam)**2) + O(lam**-3)

The large-lam form follows from d = |x|**2, x ~ N(sqrt(lam) e, 1_k): the
information 1/(4 lam) in x minus what is lost with the direction of x on the
sphere S^{k-1} (von Mises-Fisher, Var(cos theta) ~ (k-1)/(2 kappa**2)),
i.e. F = 1/(4 lam) - (k-1)/(8 lam**2) + ..., which equals the form above to
this order.

R(lam) = sqrt(F) is tabulated on a uniform grid in x = log1p(lam) and
interpolated linearly; the transformation g(lam) = Int_0^lam R is the exact
antiderivative of that interpolant, so dg/dlam == R by construction.  Beyond
lam_max the leading term 1/sqrt(2(k+2lam)) is used.  The table depends on k;
it is built on first use for a given (k, lam_max, n), about 10 s (30 s for
k = 1, which needs a finer grid), and cached for the lifetime of the process.
No file is written.

Assumptions and limits:

- k is an integer >= 1; sigma2 is known and not inferred.
- data must be >= 0 for k = 2 and > 0 otherwise (the density is infinite
  at 0 for k = 1 and zero for k >= 3), and sigma2 > 0, checked at
  construction.  Negative primals are clamped to 0.
- Requires `jax_enable_x64`; without it a warning is issued at construction.
"""

import dataclasses
import functools
import warnings
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

from .likelihood import Likelihood
from .logger import logger
from .num.ncx2_jax import _check_k, make_ncx2_logpdf
from .tree_math import ShapeWithDtype

__all__ = ["FisherTable", "NoncentralChiSquared", "fisher_asymptotic"]

_LAM_MAX = 1e6
_N = 4096
# k = 1 (nu = -1/2, density ~ d**-1/2 at 0) is the only k whose sqrt(F) is so
# strongly curved near lam = 0 that n = 4096 misses the 1e-6 target: max
# relative error 4.7e-6 in the first grid interval, 2.95e-7 with n = 16384
# (error ~ h**2; k = 2: 7.1e-7, k = 3: 3.6e-7 with n = 4096)
_N_K1 = 16384


def _default_n(k):
    """Default table size for k (larger for k = 1, see _N_K1)."""
    return _N_K1 if k == 1 else _N


@functools.lru_cache(maxsize=None)
def _cached_logpdf(k):
    # one closure per k, shared by all likelihoods: `_logpdf` is then not part
    # of the pytree aux data and equal likelihoods hit the same jit cache entry
    return make_ncx2_logpdf(k, validate=False)


def _check_inputs(data, sigma2, k):
    # data are concrete at construction; leaves that are tracers (likelihood
    # built under jit) cannot be inspected and are skipped
    def concrete(tree):
        return [
            np.asarray(a)
            for a in jax.tree_util.tree_leaves(tree)
            if not isinstance(a, jax.core.Tracer)
        ]

    if any(np.any(a < 0) for a in concrete(data)):
        raise ValueError(
            "data must be non-negative (ncx2 draws are sums of "
            "squares; do not subtract the background)"
        )
    if k != 2 and any(np.any(a == 0) for a in concrete(data)):
        # f_k(0) is infinite for k = 1 and zero for k >= 3: the energy would
        # be -inf / +inf; d = 0 has probability zero unless data are clipped
        raise ValueError(
            f"data must be positive for k = {k} (the density is "
            f"{'infinite' if k == 1 else 'zero'} at 0)"
        )
    if any(np.any(~(a > 0)) for a in concrete(sigma2)):
        raise ValueError("sigma2 must be positive")


def fisher_asymptotic(k, lam):
    """Large-lam Fisher information to next-to-leading order.

    1/(2s) + 1/(2s**2) with s = k + 2 lam; the relative error is O(lam**-2).
    """
    s = k + 2.0 * lam
    return 1.0 / (2.0 * s) + 0.5 / s**2


def _fisher_quad(k, lam, n_panel=200, n_node=40, tail=1e-30):
    """F(lam) by Gauss-Legendre in u = sqrt(d).  lam may be an array."""
    from scipy import stats

    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    lp_k = jax.jit(_cached_logpdf(k))
    lp_k2 = jax.jit(_cached_logpdf(k + 2))
    nodes, wts = np.polynomial.legendre.leggauss(n_node)

    out = np.empty(lam.shape)
    for i, li in enumerate(lam):
        # integrate in u = sqrt(d): the peak is ~1 unit wide in u for every
        # lam, so uniform panels over the actual support resolve it
        # scipy's ncx2 needs lam > 0; 1e-12 moves the bounds negligibly
        lam_pos = max(li, 1e-12)
        lo = np.sqrt(max(stats.ncx2.ppf(tail, k, lam_pos), 0.0))
        # upper bound from k + 4: the integrand f_{k+2}**2 / f_k has a
        # heavier upper tail than f_k itself
        hi = np.sqrt(stats.ncx2.isf(tail, k + 4, lam_pos))
        edges = np.linspace(lo, hi, n_panel + 1)
        u = (
            (edges[:-1, None] + edges[1:, None]) / 2
            + (edges[1:, None] - edges[:-1, None]) / 2 * nodes
        ).ravel()
        wq = ((edges[1:, None] - edges[:-1, None]) / 2 * wts).ravel()
        d = u**2
        lv = np.full_like(d, li)
        a = np.asarray(lp_k(d, lv))
        b = np.asarray(lp_k2(d, lv))
        # (f_{k+2}/f_k - 1)**2 via expm1: at large lam the log-ratio is
        # O(1/sqrt(lam)) and the naive difference loses every digit.
        r = np.expm1(b - a)
        # Gauss nodes never hit u = 0, where f_1 diverges like 1/u; f * 2u
        # stays finite for every k
        out[i] = 0.25 * float(np.sum(wq * np.exp(a) * r**2 * 2 * u))
    return out


def _segment_integrals(R, x0, h):
    # Int_{x_i}^{x_i + h} (a + b s) e^{x_i + s} ds for the piecewise-linear R,
    # written with expm1 (the plain differences cancel for small x)
    a, b = R[:-1], (R[1:] - R[:-1]) / h
    em1 = np.expm1(h)
    return np.exp(x0) * (a * em1 + b * (h * (em1 + 1.0) - em1))


@dataclasses.dataclass(frozen=True)
class FisherTable:
    """Interpolated sqrt(F) and its exact antiderivative, in x = log1p(lam)."""

    k: int
    x: np.ndarray  # uniform grid in log1p(lam)
    R: np.ndarray  # sqrt(F) at the grid points
    G: np.ndarray  # g(lam) = Int_0^lam sqrt(F) dlam', at grid points
    h: float

    @staticmethod
    def build(k, lam_max=_LAM_MAX, n=None, **quad):
        k = _check_k(k)
        n = _default_n(k) if n is None else n
        logger.info(
            f"NoncentralChiSquared: building the Fisher table for k = {k} "
            f"(n = {n}, lam_max = {lam_max:g}); this takes a few seconds"
        )
        x = np.linspace(0.0, np.log1p(lam_max), n)
        lam = np.expm1(x)
        F = _fisher_quad(k, lam, **quad)
        F[0] = 1.0 / (2.0 * k)  # analytic anchor at lam=0
        R = np.sqrt(F)
        h = float(x[1] - x[0])
        G = np.concatenate([[0.0], np.cumsum(_segment_integrals(R, x[:-1], h))])
        return FisherTable(k, x, R, G, h)

    @property
    def lam_max(self):
        return float(np.expm1(self.x[-1]))

    def save(self, path):
        np.savez(path, k=self.k, x=self.x, R=self.R, G=self.G, h=self.h)

    @staticmethod
    def load(path):
        with np.load(path) as f:
            return FisherTable(int(f["k"]), f["x"], f["R"], f["G"], float(f["h"]))

    def matches(self, k=None, lam_max=None, n=None):
        """True if the table agrees with every parameter that is not None."""
        return (
            (k is None or self.k == k)
            and (lam_max is None or abs(self.lam_max / lam_max - 1.0) < 1e-9)
            and (n is None or self.R.size == n)
        )

    @staticmethod
    def load_or_build(k, lam_max=_LAM_MAX, n=None, **quad):
        """Build the table for k, cached in memory per (k, lam_max, n).

        Never writes files; use `save` / `load` to keep a table across
        processes.  Custom quadrature settings bypass the cache.
        """
        k = _check_k(k)
        n = _default_n(k) if n is None else n
        if quad:
            return FisherTable.build(k, lam_max=lam_max, n=n, **quad)
        return _cached_table(k, float(lam_max), int(n))


@functools.lru_cache(maxsize=None)
def _cached_table(k, lam_max, n):
    return FisherTable.build(k, lam_max=lam_max, n=n)


class NoncentralChiSquared(Likelihood):
    """Noncentral chi-squared likelihood with integer k >= 1 and known sigma2.

    For data that are sums of `k` squared Gaussian components, each with known
    variance `sigma2`, whose squared means add up to the noise-free model
    power `s` (the primals):

    .. math ::
        d / \\sigma^2 \\sim \\chi'^2_k(\\lambda), \\quad
        \\lambda = s / \\sigma^2,

    elementwise.  The energy is the negative log-density of d / sigma2 (the
    model-independent Jacobian is dropped); metric, left_sqrt_metric and
    transformation use the Fisher information F(lam) from a table for this k.
    Requires `jax_enable_x64`.

    Parameters
    ----------
    data : tree-like structure of jnp.ndarray
        Measured powers in physical units, without background subtraction.
        Must be >= 0 for k = 2 and > 0 for every other k.
    sigma2 : tree-like structure of jnp.ndarray
        Variance of one Gaussian component, in the same units as `data`; > 0.
    k : int
        Degrees of freedom (number of squared Gaussian components), >= 1.
    table : FisherTable, optional
        Fisher table to use.  By default a table for `k` is built on first use
        (lam_max = 1e6, n = 4096; n = 16384 for k = 1) and cached in memory.
    table_path : str, optional
        Load the table from this `.npz` file instead (see `FisherTable.save`).
    lam_max, n : float, int, optional
        Table range and size.  Without `table`/`table_path` a table with these
        parameters is built (or taken from the cache); with a given table they
        must agree with it, otherwise ValueError is raised.
    sampling_dtype : dtype
        dtype of the domain and of the lsm tangents.

    `primals` are the noise-free model powers, in the same units as `data`.
    They must be non-negative; put that into the forward model (softplus/exp)
    rather than relying on it.  Negative primals are clamped to 0 in the
    energy and in the table lookup, i.e. they get the likelihood of a zero
    model; there the transformation is constant while left_sqrt_metric is
    not, so the two are only consistent for primals >= 0.
    """

    data: Any = dataclasses.field(metadata=dict(static=False))
    sigma2: Any = dataclasses.field(metadata=dict(static=False))
    d: Any = dataclasses.field(metadata=dict(static=False))
    tab_x: Any = dataclasses.field(metadata=dict(static=False))
    tab_R: Any = dataclasses.field(metadata=dict(static=False))
    tab_G: Any = dataclasses.field(metadata=dict(static=False))
    # static, hashable scalars: they end up in the pytree aux data and are
    # restored verbatim by `tree_unflatten`
    k: int = dataclasses.field(metadata=dict(static=True))
    h: float = dataclasses.field(metadata=dict(static=True))
    lam_max: float = dataclasses.field(metadata=dict(static=True))
    _n_tab: int = dataclasses.field(metadata=dict(static=True))
    _g_at_lam_max: float = dataclasses.field(metadata=dict(static=True))

    def __init__(
        self,
        data,
        sigma2,
        k,
        table=None,
        table_path=None,
        sampling_dtype=float,
        lam_max=None,
        n=None,
    ):
        k = _check_k(k)
        if not jax.config.jax_enable_x64:
            # measured against float64 scipy (maximum over k <= 1000;
            # 5.8e-5 for k <= 101)
            warnings.warn(
                "NoncentralChiSquared: jax_enable_x64 is off; per-observation "
                "log-density error up to 1.3e-4 (relative) for lam <= 1e7 and "
                "k <= 1000, growing with lam.",
                stacklevel=2,
            )
        _check_inputs(data, sigma2, k)
        self.data = data
        self.sigma2 = sigma2
        self.d = jax.tree_util.tree_map(lambda a, s: a / s, data, sigma2)
        self.k = k

        if table is not None and table_path is not None:
            raise ValueError("pass either `table` or `table_path`, not both")
        if table_path is not None:
            table = FisherTable.load(table_path)
        if table is None:
            tab = FisherTable.load_or_build(
                k, lam_max=_LAM_MAX if lam_max is None else lam_max, n=n
            )
        else:
            if not table.matches(lam_max=lam_max, n=n):
                raise ValueError(
                    f"table (lam_max = {table.lam_max:g}, n = {table.R.size}) "
                    f"conflicts with lam_max = {lam_max}, n = {n}"
                )
            tab = table
        if tab.k != k:
            raise ValueError(f"table was built for k = {tab.k}, need k = {k}")
        # dynamic fields: anything without static=False lands in the static aux
        # data of the pytree, and jnp arrays there are not hashable
        self.tab_x = jnp.asarray(tab.x)
        self.tab_R = jnp.asarray(tab.R)
        self.tab_G = jnp.asarray(tab.G)
        self.h = float(tab.h)
        self.lam_max = float(tab.lam_max)
        self._n_tab = int(tab.R.size)
        self._g_at_lam_max = float(tab.G[-1])

        shp = jax.tree_util.tree_map(
            lambda x: ShapeWithDtype(jnp.shape(x), sampling_dtype), data
        )
        super().__init__(domain=shp, lsm_tangents_shape=shp)

    @property
    def _logpdf(self):
        return _cached_logpdf(self.k)

    # ------------------------------------------------------------- lookup
    def _locate(self, lam):
        # NOT jnp.clip: at a boundary its JVP averages the two one-sided
        # derivatives, which silently halves d(transformation)/dlam at lam = 0
        # and at lam_max.  jnp.where has a well-defined one-sided gradient.
        lam_c = jnp.where(lam > self.lam_max, self.lam_max, lam)
        lam_c = jnp.where(lam_c < 0.0, 0.0, lam_c)
        x = jnp.log1p(lam_c)
        # clip is fine here: integer index, no gradient flows through it
        i = jnp.clip((x / self.h).astype(jnp.int32), 0, self._n_tab - 2)
        return x, i

    # Beyond the table only the leading term 1/sqrt(2(k+2lam)) is used, so that
    # d(g_asym)/dlam == R_asym holds exactly; the dropped correction is
    # 1/(2(k+2lam)) relative in R.
    def _sqrt_fisher_lam(self, lam):
        x, i = self._locate(lam)
        xi = self.tab_x[i]
        tab = self.tab_R[i] + (self.tab_R[i + 1] - self.tab_R[i]) * (x - xi) / self.h
        big = lam > self.lam_max
        lam_s = jnp.where(big, lam, self.lam_max)  # double-where
        asym = 1.0 / jnp.sqrt(2.0 * (self.k + 2.0 * lam_s))
        return jnp.where(big, asym, tab)

    def _transformation_lam(self, lam):
        x, i = self._locate(lam)
        xi = self.tab_x[i]
        a = self.tab_R[i]
        b = (self.tab_R[i + 1] - self.tab_R[i]) / self.h
        s = x - xi
        # e - e0 = e0*expm1(s) and (s-1)*e + e0 = e0*(s*e**s - expm1(s)) are
        # written with expm1: the plain differences cancel for small lam
        e0, em1 = jnp.exp(xi), jnp.expm1(s)
        tab = self.tab_G[i] + e0 * (a * em1 + b * (s * (em1 + 1.0) - em1))
        big = lam > self.lam_max
        lam_s = jnp.where(big, lam, self.lam_max)  # double-where
        g_asym = jnp.sqrt(0.5 * (self.k + 2.0 * lam_s))  # Int 1/sqrt(2(k+2lam))
        # offset so that g is continuous at lam_max
        c = self._g_at_lam_max - np.sqrt(0.5 * (self.k + 2.0 * self.lam_max))
        return jnp.where(big, c + g_asym, tab)

    # ------------------------------------------- Likelihood interface
    def energy(self, primals):
        lp = jax.tree_util.tree_map(
            lambda dd, p, s: self._logpdf(dd, p / s), self.d, primals, self.sigma2
        )
        return -sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.sum, lp)))

    def metric(self, primals, tangents):
        return jax.tree_util.tree_map(
            lambda p, t, s: self._sqrt_fisher_lam(p / s) ** 2 / s**2 * t,
            primals,
            tangents,
            self.sigma2,
        )

    def left_sqrt_metric(self, primals, tangents):
        return jax.tree_util.tree_map(
            lambda p, t, s: self._sqrt_fisher_lam(p / s) / s * t,
            primals,
            tangents,
            self.sigma2,
        )

    def transformation(self, primals):
        return jax.tree_util.tree_map(
            lambda p, s: self._transformation_lam(p / s), primals, self.sigma2
        )

    def normalized_residual(self, primals):
        # (d - E[d]) * sqrt(F(lam)) with E[d] = k + lam, in ncx2 units.  The
        # factor sqrt(F) is the one left_sqrt_metric uses; it equals
        # 1/sqrt(Var d) = 1/sqrt(2(k+2lam)) at lam = 0 and as lam -> inf, so
        # the residuals have unit variance there (for intermediate lam their
        # variance is F * Var(d)).  optimize_kl passes this to minisanity,
        # which reports the reduced chi**2 of these residuals.
        return jax.tree_util.tree_map(
            lambda p, dd, s: self._sqrt_fisher_lam(p / s) * (dd - (self.k + p / s)),
            primals,
            self.d,
            self.sigma2,
        )
