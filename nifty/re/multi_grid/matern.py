#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Philipp Frank, Laurin Soeding, Gordian Edenhofer

from dataclasses import field
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sp

from ..model import Model
from ..tree_math import norm
from ..prior import LogNormalPrior, NormalPrior

# For legacy Matern
from dataclasses import dataclass
from jax.tree_util import Partial, register_dataclass
from numpy import typing as npt
from scipy.special import j0, sici
from ..tree_math import zeros_like
from ..logger import logger

_RP1 = jnp.array(
    [
        -8.99971225705559398224e8,
        4.52228297998194034323e11,
        -7.27494245221818276015e13,
        3.68295732863852883286e15,
    ]
)
_RQ1 = jnp.array(
    [
        1.0,
        6.20836478118054335476e2,
        2.56987256757748830383e5,
        8.35146791431949253037e7,
        2.21511595479792499675e10,
        4.74914122079991414898e12,
        7.84369607876235854894e14,
        8.95222336184627338078e16,
        5.32278620332680085395e18,
    ]
)

_PP1 = jnp.array(
    [
        7.62125616208173112003e-4,
        7.31397056940917570436e-2,
        1.12719608129684925192e0,
        5.11207951146807644818e0,
        8.42404590141772420927e0,
        5.21451598682361504063e0,
        1.00000000000000000254e0,
    ]
)
_PQ1 = jnp.array(
    [
        5.71323128072548699714e-4,
        6.88455908754495404082e-2,
        1.10514232634061696926e0,
        5.07386386128601488557e0,
        8.39985554327604159757e0,
        5.20982848682361821619e0,
        9.99999999999999997461e-1,
    ]
)

_QP1 = jnp.array(
    [
        5.10862594750176621635e-2,
        4.98213872951233449420e0,
        7.58238284132545283818e1,
        3.66779609360150777800e2,
        7.10856304998926107277e2,
        5.97489612400613639965e2,
        2.11688757100572135698e2,
        2.52070205858023719784e1,
    ]
)
_QQ1 = jnp.array(
    [
        1.0,
        7.42373277035675149943e1,
        1.05644886038262816351e3,
        4.98641058337653607651e3,
        9.56231892404756170795e3,
        7.99704160447350683650e3,
        2.82619278517639096600e3,
        3.36093607810698293419e2,
    ]
)

_Z1 = 1.46819706421238932572e1
_Z2 = 4.92184563216946036703e1
_THPIO4 = 2.35619449019234492885  # 3*pi/4
_SQ2OPI = 0.79788456080286535588  # sqrt(2/pi)


def _j1_small(x):
    z = x * x
    w = jnp.polyval(_RP1, z) / jnp.polyval(_RQ1, z)
    w = w * x * (z - _Z1) * (z - _Z2)
    return w


def _j1_large_c(x):
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(_PP1, z) / jnp.polyval(_PQ1, z)
    q = jnp.polyval(_QP1, z) / jnp.polyval(_QQ1, z)
    xn = x - _THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * _SQ2OPI / jnp.sqrt(x)


def j1(x):
    """
    Bessel function of order one - using the implementation from CEPHES,
    translated to Jax.

    Pure JAX implementation for Bessel function from
    https://github.com/benjaminpope/sibylla/blob/main/notebooks/bessel_test.ipynb .
    """
    return jnp.sign(x) * jnp.where(
        jnp.abs(x) < 5.0, _j1_small(jnp.abs(x)), _j1_large_c(jnp.abs(x))
    )


@dataclass
class FourierIntegralGrid:
    mode_lengths: jax.Array
    mode_binbounds: jax.Array
    min_dist: float
    max_dist: float
    num: int
    ndim: int
    weights: jax.Array


register_dataclass(
    FourierIntegralGrid,
    data_fields=["mode_lengths", "mode_binbounds", "weights"],
    meta_fields=["num", "min_dist", "max_dist", "ndim"],
)


def make_integral_grid(
    min_dist: float, max_dist: float, num: int, ndim: int, normalize: bool = True
) -> FourierIntegralGrid:
    mode_lengths = np.geomspace(1.0 / max_dist, 1.0 / min_dist, num, endpoint=False)
    mode_lengths = np.insert(mode_lengths, 0, 0.0)
    # Binbounds
    lk = np.log(mode_lengths[1:])
    dlk = (np.log(max_dist) - np.log(min_dist)) / num
    lk = np.append(lk - 0.5 * dlk, lk[-1] + 0.5 * dlk)
    mode_binbounds = np.insert(np.exp(lk), 0, 0.0)
    # norm_weights
    wgt = None
    if normalize:
        if ndim == 1:
            fkr = sici(mode_binbounds * max_dist)[0]
        elif ndim == 2:
            fkr = 1.0 - j0(mode_binbounds * max_dist)
        elif ndim == 3:
            fkr = sici(mode_binbounds * max_dist)[0] - np.sin(mode_binbounds * max_dist)
        else:
            raise NotImplementedError
        wgt = fkr[1:] - fkr[:-1]
        if (ndim == 1) or (ndim == 3):
            wgt *= 2.0 / np.pi
    return FourierIntegralGrid(
        min_dist=min_dist,
        max_dist=max_dist,
        num=num,
        ndim=ndim,
        mode_lengths=mode_lengths,
        mode_binbounds=mode_binbounds,
        weights=wgt,
    )


def spectrum2covariance(
    fig: FourierIntegralGrid, spec: npt.NDArray, *, ref_distance=1.0, normalize=True
):
    fct = [np.pi, 2.0 * np.pi, 2.0 * np.pi**2]

    def cov(r: npt.NDArray) -> npt.NDArray:
        k = jnp.expand_dims(fig.mode_binbounds, tuple(i for i in range(len(r.shape))))
        r = r[..., jnp.newaxis]
        kr = r * k
        if fig.ndim == 1:
            fkr = jnp.sin(kr)
        elif fig.ndim == 2:
            fkr = kr * j1(kr)
        elif fig.ndim == 3:
            fkr = jnp.sin(kr) - kr * jnp.cos(kr)
        else:
            raise NotImplementedError
        res0 = (k[..., 1:] ** fig.ndim - k[..., :-1] ** fig.ndim) / fig.ndim
        resn0 = (fkr[..., 1:] - fkr[..., :-1]) / r**fig.ndim
        res = jnp.where(r < ref_distance * 1e-10, res0, resn0) / fct[fig.ndim - 1]
        res = jnp.tensordot(res, spec, axes=(-1, 0))
        if normalize:
            res /= (fig.weights * spec).sum()
        return res

    return cov


class MaternHarmonicCovariance(Model):
    scale: Union[Model, float] = field(metadata=dict(static=False))
    cutoff: Union[Model, float] = field(metadata=dict(static=False))
    loglogslope: Union[Model, float] = field(metadata=dict(static=False))
    _interp_dists: jnp.ndarray = field(metadata=dict(static=False))
    _fig: FourierIntegralGrid = field(metadata=dict(static=False))

    def __init__(
        self,
        scale: Union[tuple, Callable, float],
        cutoff: Union[tuple, Model, float],
        loglogslope: Union[tuple, Callable, float],
        *,
        ndim: int,
        n_integrate=2_048,
        n_interpolate=512,
        integration_dists_min_max=None,
        interpolation_dists_min_max=None,
        kind: str = "amplitude",
        prefix: str = "",
    ):
        """Compute the Matérn-kernel using its harmonic representation.

        See also
        --------
        `Causal, Bayesian, & non-parametric modeling of the SARS-CoV-2 viral
        load vs. patient's age`, Guardiani, Matteo and Frank, Philipp and Kostić,
        Andrija and Edenhofer, Gordian and Roth, Jakob and Uhlmann, Berit and
        Enßlin, Torsten, `<https://arxiv.org/abs/2105.13483>`_
        `<https://doi.org/10.1371/journal.pone.0275011>`_
        """

        logger.warning(
            "The experimental Matern kernel is deprecated and will be removed in a future version.\n"
            "We recommend using the new >MaternCovarianceModel< instead."
        )

        ref_distance = 1.0
        if isinstance(cutoff, (tuple, list)):
            ref_distance *= cutoff[0]
            cutoff = LogNormalPrior(*cutoff, name=prefix + "cutoff")
        elif isinstance(cutoff, Model):
            ref_distance *= cutoff(zeros_like(cutoff.domain))
        elif isinstance(cutoff, float):
            ref_distance *= cutoff
        else:
            raise TypeError(f"invalid `cutoff` specified; got '{cutoff!r}'")
        self.cutoff = cutoff
        self._ref_distance = ref_distance
        if isinstance(loglogslope, (tuple, list)):
            loglogslope = NormalPrior(*loglogslope, name=prefix + "loglogslope")
        elif not (callable(loglogslope) or isinstance(loglogslope, float)):
            raise TypeError(f"invalid `loglogslope` specified; got '{loglogslope!r}'")
        self.loglogslope = loglogslope
        if isinstance(scale, (tuple, list)):
            scale = LogNormalPrior(*scale, name=prefix + "scale")
        elif not (callable(scale) or isinstance(scale, float)):
            raise TypeError(f"invalid `scale` specified; got '{scale!r}'")
        self.scale = scale

        self.kind = kind
        self.ndim = ndim

        if integration_dists_min_max is None:
            integration_dists_min_max = tuple(
                np.array([1e-3, 1e4]) / self._ref_distance
            )
        if interpolation_dists_min_max is None:
            interpolation_dists_min_max = tuple(
                np.array([1e-4, 1e1]) / self._ref_distance
            )
        self._interp_dists = jnp.geomspace(*interpolation_dists_min_max, n_interpolate)
        min_dist, max_dist = integration_dists_min_max
        self._fig = make_integral_grid(
            min_dist, max_dist, n_integrate, ndim=self.ndim, normalize=False
        )

        super().__init__(
            domain=getattr(self.scale, "domain", {})
            | getattr(self.loglogslope, "domain", {})
            | getattr(self.cutoff, "domain", {})
        )

    def normalized_spectrum(self, x):
        cutoff = self.cutoff(x) if callable(self.cutoff) else self.cutoff
        loglogslope = (
            self.loglogslope(x) if callable(self.loglogslope) else self.loglogslope
        )

        ln_spectrum = (
            0.25 * loglogslope * jnp.log1p((self._fig.mode_lengths / cutoff) ** 2)
        )

        spectrum = jnp.exp(ln_spectrum)
        spectrum = spectrum.at[0].set(spectrum[1])
        if self.kind.lower() == "amplitude":
            spectrum = spectrum**2
        elif self.kind.lower() != "power":
            raise ValueError(f"invalid kind specified {self.kind!r}")
        return spectrum

    @staticmethod
    def _interp_cov(x, y, *, scale, distances, logcorr):
        r = jnp.linalg.norm(x - y, axis=0, ord=2)
        cov = jnp.exp(
            jnp.interp(r, distances, logcorr, left="extrapolate", right="extrapolate")
        )
        cov = jnp.where(r == 0.0, jnp.ones_like(r), cov)
        return scale * cov

    def __call__(self, x):
        scale = self.scale(x) if callable(self.scale) else self.scale
        spec = self.normalized_spectrum(x)
        corr_func = spectrum2covariance(
            self._fig, spec, ref_distance=self._ref_distance, normalize=False
        )

        corr = jax.vmap(corr_func)(self._interp_dists) / corr_func(jnp.array([0.0]))
        # This could perhaps go even lower, but the accuaracy is already quite
        # good and this is definitely numerically stable.
        # Restrict domain to ensure good numerics
        ref_scale = 1e-5
        mask = corr < ref_scale
        maxidx = jnp.argmax(mask) - 1
        # Take abs to ensure good logarithm
        ln_corr = jnp.log(jnp.abs(corr))
        # DIY interpolation that keeps the shapes static
        slope_at_maxidx = (ln_corr[maxidx - 1] - ln_corr[maxidx]) / (
            self._interp_dists[maxidx - 1] - self._interp_dists[maxidx]
        )
        ln_corr = jnp.where(
            ~mask,
            ln_corr,
            ln_corr[maxidx]
            + slope_at_maxidx * (self._interp_dists - self._interp_dists[maxidx]),
        )

        return Partial(
            self.__class__._interp_cov,
            scale=scale,
            distances=self._interp_dists,
            logcorr=ln_corr,
        )


# Below this line: New implementation of Matern kernel


def get_bessel_zeros(nu, N_zeros, n_iter=3):
    """
    Compute the first :math:`N_\\mathrm{zeros}` zeros of :math:`\\mathcal{J}_\\nu(x)` (Bessel function of the first kind).

    Args:
        nu : float
            Order of the Bessel function
        N_zeros : int
            Number of zeros to compute
        n_iter : int
            Number of Newton-Raphson refinement iterations
    Returns:
        zeros : array
            First :math:`N_\\mathrm{zeros}` zeros of :math:`\\mathcal{J}_\\nu(x)`
    """
    # Use SciPy for integer order
    if np.isclose(nu % 1, 0, atol=1e-6, rtol=0.0):
        return sp.jn_zeros(int(nu), N_zeros)
    # For non-integer order, use McMahon's asymptotic expansion, then refine with Newton-Raphson
    k = np.arange(1, N_zeros + 1)
    mu = 4 * nu**2
    a = (k + nu / 2 - 0.25) * np.pi
    zeros = a - (mu - 1) / (8 * a) - 4 * (mu - 1) * (7 * mu - 31) / (3 * (8 * a) ** 3)

    for _ in range(n_iter):
        zeros = zeros - sp.jv(nu, zeros) / sp.jvp(nu, zeros)

    return zeros


def default_h(Nint, nu, t_max=2.0):
    max_zero = get_bessel_zeros(nu, Nint)[-1]
    return t_max * np.pi / max_zero


def default_Nint(h, nu, t_max=2.0):
    # Use asymptotic formula for large N
    # h = t_max / (N+nu/2-0.25) -> N = t_max/h - nu/2 + 0.25
    return int(np.ceil(t_max / h - nu / 2 + 0.25))


class IsotropicPowerSpectrumTransform:
    """
    Transform between isotropic power spectrum P(k) and covariance
    kernel :math:`\\mathrm{Cov}(r)` in :math:`N_\\mathrm{dim}` dimensions.

    The general formula is:

    .. math ::
        \\mathrm{Cov}(r) = \\frac{1}{(2 \\pi)^{N_\\mathrm{dim}/2}} \\int_0^\\infty P(k) \\, k^{(N_\\mathrm{dim}-1)} \\frac{J_\\nu(k r)}{(k r)^\\nu} dk
    .. math ::
                         = \\frac{1}{(2 \\pi)^{N_\\mathrm{dim}/2}} \\int_0^\\infty P\\left(\\frac{x}{r}\\right) r^{-N_\\mathrm{dim}} x^{N_\\mathrm{dim}/2} J_\\nu(x) dx

    with :math:`x=kr` and :math:`\\nu = (N_\\mathrm{dim}-2)/2`.

    We use a modified version of Ogata quadrature
    (eqn. 5.2 in https://ems.press/journals/prims/articles/2319)
    to compute the integral.

    The initialisation precomputes the Ogata quadrature nodes and weights for given :math:`N_\\mathrm{dim}`.
    This needs to be done once at initialisation and depends on scipy.
    The application uses pure JAX for differentiability.

    Some advice on modelling power spectra: Without a high-k cutoff, it can quickly
    happen that the covariance function diverges. We therefore recommend to use power spectra
    that either decay sufficiently fast at high k, or have compact support.

    Args:
        Ndim : int
            Number of spatial dimensions
        Nint : int
            Number of integration nodes for Ogata quadrature. Chosen automatically if set to "auto" and h is provided.
        h : float
            Step-size. Controls accuracy. Chosen automatically if set to "auto" and Nint is provided.
    """

    def __init__(
        self, Ndim: int, Nint: Union[int, str] = 1024, h: Union[float, str] = "auto"
    ):
        # Input validation
        if not isinstance(Ndim, int) or Ndim < 1:
            raise ValueError("Ndim must be a positive integer.")
        if not (isinstance(h, (float, int)) and h > 0) and h != "auto":
            raise ValueError("h must be positive or 'auto'.")
        if not (isinstance(Nint, int) and Nint > 0) and Nint != "auto":
            raise ValueError("Nint must be a positive integer or 'auto'.")
        if h == "auto" and Nint == "auto":
            raise ValueError(
                "At least one of h or Nint must be specified (not 'auto')."
            )

        self.Ndim = Ndim
        self.nu = (Ndim - 2) / 2
        self.h = default_h(Nint, self.nu) if h == "auto" else h
        self.Nint = default_Nint(h, self.nu) if Nint == "auto" else Nint

        # Precompute nodes and weights for Ogata quadrature
        zeros = get_bessel_zeros(self.nu, self.Nint)
        xi = zeros / np.pi

        def psi(t):
            t = np.asarray(t)
            mask = t <= 10.0  # Prevent overflow
            res = t.copy()  # For large t, psi(t) = t
            t_masked = t[mask]
            res[mask] = t_masked * np.tanh(
                np.pi / 2 * np.exp(t_masked - 1.0 / t_masked)
            )
            return res

        def dpsi(t):
            t = np.asarray(t)
            mask = t <= 10.0  # Prevent overflow
            res = np.ones_like(t)  # For large t, dpsi/dt = 1
            t_masked = t[mask]
            exp_term = np.exp(t_masked - 1.0 / t_masked)
            th = np.tanh(np.pi / 2 * exp_term)
            res[mask] = th + t_masked * np.pi / 2 * exp_term * (1 - th**2) * (
                1 + 1.0 / t_masked**2
            )
            return res

        nodes = np.pi * psi(self.h * xi) / self.h
        w = sp.yv(self.nu, zeros) / sp.jv(self.nu + 1, zeros)
        dpsi_vals = dpsi(self.h * xi)
        J = sp.jv(
            self.nu, np.clip(nodes, 1e-30, None)
        )  # Clip to prevent numerical issues with J_nu(0) diverging
        prefactor = 1 / (2 * jnp.pi) ** (self.Ndim / 2)

        # Store as JAX arrays
        self.nodes = jnp.array(nodes)
        self.weights = jnp.array(
            prefactor * np.pi * w * dpsi_vals * J * nodes ** (self.Ndim / 2)
        )

    def pk_to_cov(
        self, P_func: Callable[[jnp.ndarray], jnp.ndarray], r: Union[float, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Compute covariance kernel from power spectrum.

        Parameters
        ----------
        P_func : callable
            Power spectrum P(k), must accept JAX arrays
        r : array
            Separation values at which to evaluate the covariance kernel

        Returns
        -------
        cov : array
            Covariance kernel values
        """
        if not callable(P_func):
            raise ValueError("P_func must be callable.")

        r = jnp.atleast_1d(jnp.asarray(r))
        Ndim = self.Ndim
        nodes = self.nodes
        weights = self.weights

        def cov_single(r_val):
            P_vals = P_func(nodes / r_val)
            integral = jnp.sum(weights * P_vals)
            return r_val ** (-Ndim) * integral

        return jnp.vectorize(cov_single)(r)

    def k_range_at_r(self, r):
        """Return (k_min, k_max) as functions of r."""
        return (self.nodes[0] / r, self.nodes[-1] / r)


class MaternCovarianceKernel(IsotropicPowerSpectrumTransform):
    """
    Computes the covariance kernel for a Matern-like power spectrum
    with a high-k cutoff. The power spectrum is given by:

    .. math ::
        P(k) = \\frac{1}{\\left(1 + \\left(\\alpha k\\right)^2\\right)^{\\beta / 2}} \\cdot e^{-(k / k_\\mathrm{cutoff})^2}

    where :math:`\\beta` (negloglogslope) controls the logarithmic slope at high k,
    :math:`\\alpha` (lengthscale) controls
    the correlation length scale, and kcutoff sets the high-k cutoff.

    Args:
        Ndim : int
            Number of spatial dimensions
        r_min : float
            Minimum r value for covariance kernel interpolation. This should
            correspond to the smallest length scale you want to resolve.
        r_max : float
            Maximum r value for covariance kernel interpolation. This should
            correspond to the largest length scale you want to resolve.
        Ninterp : int
            Number of interpolation/evaluation points for the covariance kernel.
        jitter : float
            Small value added to the kernel at r=0 for numerical stability. Increase
            this if you encounter numerical issues with Cholesky decompositions.
        Nint : int
            Number of integration nodes for Ogata quadrature. Chosen automatically if
            set to "auto" and h is provided.
        h : float
            Step-size. Controls accuracy. Chosen automatically if set to "auto" and Nint
            is provided.
        enforce_nonnegativity : bool
            If True, the covariance values are clipped to be non-negative. This is not
            mathematically necessary but can help with numerical stability, especially
            when used together with enforce_monotonicity.
        enforce_monotonicity : bool
            If True, the covariance values are post-processed to ensure they are
            non-increasing with r. This is not mathematically necessary but can help
            with numerical stability, especially when using ICR or graphgp.
    """

    def __init__(
        self,
        Ndim: int,
        r_min: float,
        r_max: float,
        Ninterp: int = 256,
        jitter: float = 1.0e-5,
        Nint: int = 1024,
        h: float = "auto",
        enforce_nonnegativity: bool = True,
        enforce_monotonicity: bool = True,
    ):
        # Input validation
        if not isinstance(r_min, (float, int)) or r_min <= 0:
            raise ValueError("r_min must be a positive scalar.")
        if not isinstance(r_max, (float, int)) or r_max <= r_min:
            raise ValueError("r_max must be a positive scalar greater than r_min.")
        if not isinstance(Ninterp, int) or Ninterp <= 0:
            raise ValueError("Ninterp must be a positive integer.")
        if not (isinstance(jitter, (float, int)) and jitter >= 0):
            raise ValueError("jitter must be a non-negative scalar.")
        if not isinstance(enforce_nonnegativity, bool):
            raise ValueError("enforce_nonnegativity must be a boolean.")
        if not isinstance(enforce_monotonicity, bool):
            raise ValueError("enforce_monotonicity must be a boolean.")

        super().__init__(Ndim=Ndim, Nint=Nint, h=h)
        self.jitter = jitter
        self.enforce_nonnegativity = enforce_nonnegativity
        self.enforce_monotonicity = enforce_monotonicity
        self.cov_rs_no_zero = jnp.geomspace(r_min, r_max, Ninterp)

    @staticmethod
    def Pk(k, *, lengthscale, negloglogslope, kcutoff):
        """
        Matern-like power spectrum with high-k cutoff:

        .. math ::
            P(k) = \\frac{1}{\\left(1 + \\left(\\alpha k\\right)^2\\right)^{\\beta / 2}} \\cdot e^{-(k / k_\\mathrm{cutoff})^2}

        Args:
            k : array
                Wavenumber values
            lengthscale : float
                Length scale parameter of the Matern-like power spectrum
            negloglogslope : float
                Log-log slope parameter of the Matern-like power spectrum
            kcutoff : float
                High-k cutoff parameter of the Matern-like power spectrum
        Returns:
            P(k) : array
                Power spectrum values
        """
        return (
            1.0
            / (1.0 + (k * lengthscale) ** 2) ** (negloglogslope / 2.0)
            * jnp.exp(-((k / kcutoff) ** 2))
        )

    def _normalisation_factor(self, lengthscale, negloglogslope, kcutoff):
        """
        Compute the normalisation factor, that is the unnormalised covariance at r=0.
        In this case, the equation simplifies to:

        .. math ::
            \\mathrm{Cov}(0) = \\frac{1}{(2 \\pi)^{N/2}} \\cdot \\frac{1}{\\Gamma(N/2)} \\cdot \\frac{1}{2^{N/2-1}} \\cdot \\int_0^\\infty P(k) \\, k^{N-1} dk
            = \\frac{1}{(2 \\pi)^{N/2}} \\cdot \\frac{1}{\\Gamma(N/2)} \\cdot \\frac{1}{2^{N/2-1}} \\cdot \\int_{-\\infty}^\\infty P\\left(e^u\\right) e^{N u} du

        with :math:`u = \\log(k)`, :math:`du = dk/k`.

        Args:
            lengthscale : float
                Length scale parameter of the Matern-like power spectrum
            negloglogslope : float
                Log-log slope parameter of the Matern-like power spectrum
            kcutoff : float
                High-k cutoff parameter of the Matern-like power spectrum
        Returns:
            norm_factor : float
                Normalisation factor for the covariance kernel at r=0
        """
        prefactor = 1.0 / (
            (2.0 * jnp.pi) ** (self.Ndim / 2.0)
            * jax.scipy.special.gamma(self.Ndim / 2.0)
            * 2.0 ** (self.Ndim / 2.0 - 1.0)
        )

        def integrand(u):
            k = jnp.exp(u)
            Pk = self.Pk(
                k,
                lengthscale=lengthscale,
                negloglogslope=negloglogslope,
                kcutoff=kcutoff,
            )
            return Pk * k**self.Ndim

        u_min = jnp.log(1.0e-10) / self.Ndim
        u_max = jnp.log(4.0 * kcutoff)
        u_array = jnp.linspace(u_min, u_max, self.Nint)
        integral = jnp.trapezoid(integrand(u_array), u_array)
        return prefactor * integral

    def get_covariance_kernel_pair(
        self, variance, lengthscale, negloglogslope, kcutoff
    ):
        """
        Returns (cov_rs, cov_vals) where cov_rs are the r values (including r=0) and
        cov_vals are the corresponding covariance values.

        Note: The covariance is normalised by cov(0) = variance.
        Afterwards, jitter is added at r = 0 for numerical stability.

        Args:
            variance: float
                Scaling factor defining cov(0) = variance * (1 + jitter)
            lengthscale : float
                Length scale parameter of the Matern-like power spectrum
            negloglogslope : float
                Log-log slope parameter of the Matern-like power spectrum
            kcutoff : float
                High-k cutoff parameter of the Matern-like power spectrum
        Returns:
            cov_rs : array
                r values including r=0
            cov_vals : array
                Corresponding covariance values
        """
        cov_fn = jax.tree_util.Partial(
            self.Pk,
            lengthscale=lengthscale,
            negloglogslope=negloglogslope,
            kcutoff=kcutoff,
        )
        cov_vals = self.pk_to_cov(cov_fn, r=self.cov_rs_no_zero)
        cov_vals /= self._normalisation_factor(lengthscale, negloglogslope, kcutoff)
        # Add jitter at r = 0
        cov_vals = jnp.append(1.0 + self.jitter, cov_vals)
        cov_rs = jnp.append(0.0, self.cov_rs_no_zero)
        cov_vals *= variance
        if self.enforce_nonnegativity:
            cov_vals = jnp.maximum(cov_vals, 0.0)
        if self.enforce_monotonicity:
            cov_vals = jnp.minimum.accumulate(cov_vals)
        return (cov_rs, cov_vals)

    def get_covariance_kernel_interpolator(
        self, variance, lengthscale, negloglogslope, kcutoff
    ):
        """
        Returns a function cov_fn(r) that interpolates the covariance kernel
        for given Matern-like power spectrum parameters.

        Note: The covariance is normalised by cov(0) = variance at r = 0.
        Afterwards, jitter is added at r = 0 for numerical stability.

        Args:
            variance: float
                Scaling factor defining cov(0) = variance * (1 + jitter)
            lengthscale : float
                Length scale parameter of the Matern-like power spectrum
            negloglogslope : float
                Log-log slope parameter of the Matern-like power spectrum
            kcutoff : float
                High-k cutoff parameter of the Matern-like power spectrum
        Returns:
            cov_fn : callable
                Function that takes r values and returns linearly interpolated covariance values
        """
        cov_rs, cov_vals = self.get_covariance_kernel_pair(
            variance, lengthscale, negloglogslope, kcutoff
        )

        def cov_fn(r, _variance, _cov_rs, _cov_vals, _jitter):
            return jnp.where(
                r == 0.0,
                _variance * (1.0 + _jitter) * jnp.ones_like(r),
                jnp.interp(r, _cov_rs, _cov_vals),
            )

        return jax.tree_util.Partial(
            cov_fn,
            _variance=variance,
            _cov_rs=cov_rs[1:],
            _cov_vals=cov_vals[1:],
            _jitter=self.jitter,
        )


class MaternCovarianceModel(MaternCovarianceKernel, Model):
    """
    NIFTy-wrapper for the Matern covariance kernel.
    Computes the covariance kernel for a Matern-like power spectrum
    with a high-k cutoff. The power spectrum is given by:

    .. math ::
        P(k) = \\frac{1}{\\left(1 + \\left(\\alpha k\\right)^2\\right)^{\\beta / 2}} \\cdot e^{-(k / k_\\mathrm{cutoff})^2}

    where :math:`\\beta` (negloglogslope) controls the logarithmic slope at high k,
    :math:`\\alpha` (lengthscale) controls the correlation length scale,
    and :math:`k_\\mathrm{cutoff}` sets the high-k cutoff.

    Args:
        Ndim: int
            Number of spatial dimensions
        r_min: float
            Minimum r value for covariance kernel interpolation. This should
            correspond to the smallest length scale you want to resolve.
        r_max: float
            Maximum r value for covariance kernel interpolation. This should
            correspond to the largest length scale you want to resolve.
        variance: float or tuple or :class:`Model`
            This sets the variance of the Gaussian process, i.e. :math:`\\mathrm{Cov}(0) = variance * (1 + jitter)`.
            If a float is provided, it is treated as a constant. If a tuple (mean, stddev) is provided,
            it is treated as a log-normal prior. If a :class:`Model` is provided, it is used directly and must
            have a scalar target.
        lengthscale: float or tuple or :class:`Model`
            This sets roughly the largest coherent structure size in the Gaussian process. If a float
            is provided, it is treated as a constant. If a tuple (mean, stddev) is provided, it is treated
            as a log-normal prior. If a :class:`Model` is provided, it is used directly and must have a scalar target.
        negloglogslope: float or tuple or :class:`Model`
            This controls the logarithmic slope of the power spectrum at intemediate to high k. If a float
            is provided, it is treated as a constant. If a tuple (mean, stddev) is provided, it is treated
            as a log-normal prior. If a :class:`Model` is provided, it is used directly and must have a scalar target.
        kcutoff: float or tuple or :class:`Model` or "auto"
            This sets the high-k cutoff of the power spectrum. If a float is provided, it is treated as a constant.
            If a tuple (mean, stddev) is provided, it is treated as a log-normal prior. If a :class:`Model` is provided,
            it is used directly and must have a scalar target. If "auto", it is set to :math:`\\pi/r_\\mathrm{min}`.
        Ninterp: int
            Number of interpolation/evaluation points for the covariance kernel.
        jitter: float
            Small value added to the kernel at r=0 for numerical stability. Increase
            this if you encounter numerical issues with Cholesky decompositions.
        Nint: int or "auto"
            Number of integration nodes for Ogata quadrature. Chosen automatically if set to "auto" and h is provided.
        h: float or "auto"
            Step-size. Controls accuracy. Chosen automatically if set to "auto" and Nint is provided.
        prefix: str
            Prefix for the names of the priors if tuple inputs are provided. This is useful to distinguish
            multiple instances of MaternCovarianceModel. The final prior names will be prefix + attribute name,
            e.g. "MaternCovarianceModel_lengthscale".
        mode: str
            Mode of operation. If "ICR", the model is intended to be used with ICR and the __call__ method will return
            a callable covariance kernel function. If "graphgp", the model is intended to be used with GraphGP and
            the __call__ method will return a tuple (cov_rs, cov_vals) with the r values and corresponding covariance
            values for interpolation.
        enforce_nonnegativity : bool
            If True, the covariance values are clipped to be non-negative. This is not
            mathematically necessary but can help with numerical stability, especially
            when used together with enforce_monotonicity.
        enforce_monotonicity : bool
            If True, the covariance values are post-processed to ensure they are
            non-increasing with r. This is not mathematically necessary but can help
            with numerical stability, especially when using ICR or graphgp.
    """

    variance: Callable = field(metadata=dict(static=False))
    lengthscale: Callable = field(metadata=dict(static=False))
    negloglogslope: Callable = field(metadata=dict(static=False))
    kcutoff: Callable = field(metadata=dict(static=False))
    nodes: jnp.ndarray = field(metadata=dict(static=False))
    weights: jnp.ndarray = field(metadata=dict(static=False))
    cov_rs_no_zero: jnp.ndarray = field(metadata=dict(static=False))

    def __init__(
        self,
        Ndim: int,
        r_min: float,
        r_max: float,
        variance: Union[float, tuple, Model],
        lengthscale: Union[float, tuple, Model],
        negloglogslope: Union[float, tuple, Model],
        kcutoff: Union[float, tuple, Model, str] = "auto",
        Ninterp: int = 256,
        jitter: float = 1.0e-5,
        Nint: Union[int, str] = 1024,
        h: Union[float, str] = "auto",
        prefix: str = "MaternCovarianceModel_",
        mode: str = "ICR",
        enforce_nonnegativity: bool = True,
        enforce_monotonicity: bool = True,
    ):
        # Input validation
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a string.")
        if not isinstance(mode, str) or mode not in ["ICR", "graphgp"]:
            raise ValueError("mode must be either 'ICR' or 'graphgp'.")
        self.mode = mode

        MaternCovarianceKernel.__init__(
            self,
            Ndim=Ndim,
            r_min=r_min,
            r_max=r_max,
            Ninterp=Ninterp,
            jitter=jitter,
            Nint=Nint,
            h=h,
            enforce_nonnegativity=enforce_nonnegativity,
            enforce_monotonicity=enforce_monotonicity,
        )

        # Initialise Matern priors
        if kcutoff == "auto":
            kcutoff = np.pi / r_min

        for attr, prior_type in [
            ("variance", "LN"),
            ("lengthscale", "LN"),
            ("negloglogslope", "LN"),
            ("kcutoff", "LN"),
        ]:
            value = locals()[attr]
            if isinstance(value, (float, int)):
                value = jax.tree_util.Partial(lambda x, v: v, v=float(value))
            elif isinstance(value, tuple):
                if not len(value) == 2:
                    raise ValueError(
                        f"If a tuple is provided for {attr}, it must have length 2 (mean, stddev)."
                    )
                if not all(isinstance(v, (float, int)) for v in value):
                    raise ValueError(
                        f"If a tuple is provided for {attr}, both mean and stddev must be floats."
                    )
                prior = {"N": NormalPrior, "LN": LogNormalPrior}[prior_type]
                value = prior(*value, name=prefix + attr)
            elif isinstance(value, Model):
                assert (
                    value.target.shape == ()
                ), f"Model for {attr} must have scalar target (shape ())."
            else:
                if attr == "kcutoff":
                    raise ValueError(
                        f"{attr} must be a float, a len-2 tuple, a `Model`, or 'auto'."
                    )
                else:
                    raise ValueError(
                        f"{attr} must be a float, a len-2 tuple, or a `Model`."
                    )
            setattr(self, attr, value)

        Model.__init__(
            self,
            domain=getattr(self.variance, "domain", {})
            | getattr(self.lengthscale, "domain", {})
            | getattr(self.negloglogslope, "domain", {})
            | getattr(self.kcutoff, "domain", {}),
        )

    def __call__(self, x):
        if self.mode == "ICR":
            return self._call_ICR(x)
        else:
            return self._call_graphgp(x)

    def _call_ICR(self, x):
        _fun = self.get_covariance_kernel_interpolator(
            self.variance(x),
            self.lengthscale(x),
            self.negloglogslope(x),
            self.kcutoff(x),
        )

        def wrapper(x, y, fun):
            r = norm(x - y)
            return fun(r)

        return jax.tree_util.Partial(wrapper, fun=_fun)

    def _call_graphgp(self, x):
        return self.get_covariance_kernel_pair(
            self.variance(x),
            self.lengthscale(x),
            self.negloglogslope(x),
            self.kcutoff(x),
        )
