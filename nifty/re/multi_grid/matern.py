#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Laurin Soeding

from dataclasses import field
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sp

from ..model import Model
from ..tree_math import norm
from ..prior import LogNormalPrior, NormalPrior


def get_bessel_zeros(nu, N_zeros, n_iter=3):
    """
    Compute the first N_zeros zeros of J_nu(x) (Bessel function of the first kind).

    Args:
        nu : float
            Order of the Bessel function
        N_zeros : int
            Number of zeros to compute
        n_iter : int
            Number of Newton-Raphson refinement iterations
    Returns:
        zeros : array
            First N_zeros zeros of J_nu(x)
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
    kernel cov(r) in Ndim dimensions.

    The general formula is:

    cov(r) = 1/(2*pi)^{Ndim/2} * int_0^infty P(k) k^{Ndim-1} J_nu(kr)/(kr)^nu dk
           = 1/(2*pi)^{Ndim/2} * int_0^infty P(x/r)*r^{-Ndim} x^{Ndim/2} J_nu(x) dx
    with x=kr and nu = (Ndim-2)/2.

    We use a modified version of Ogata quadrature
    (eqn. 5.2 in https://ems.press/journals/prims/articles/2319)
    to compute the integral.

    The initialisation precomputes the Ogata quadrature nodes and weights for given Ndim.
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
        self, Ndim: int, Nint: Union[int, str] = 512, h: Union[float, str] = "auto"
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


class _MaternCovarianceKernel(IsotropicPowerSpectrumTransform):
    """
    Computes the covariance kernel for a Matern-like power spectrum
    with a high-k cutoff. The power spectrum is given by:

    P(k) = 1 / (1 + (k * lengthscale)^2)^(negloglogslope / 2) * exp(- (k / kcutoff)^2)

    where negloglogslope controls the logarithmic slope at high k, lengthscale controls
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
        Ninterp: int = 128,
        jitter: float = 1.0e-5,
        Nint: int = 512,
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
        Matern-like power spectrum with high-k cutoff. Equation:

        P(k) = 1 / (1 + (k * lengthscale)^2)^(negloglogslope/2) * exp(- (k / kcutoff)^2)

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

        cov(0) = 1/(2*pi)^{N/2} * 1/gamma(N/2) * 1/2^{N/2-1} * int_0^infty P(k) k^{N-1} dk
        = 1/(2*pi)^{N/2} * 1/gamma(N/2) * 1/2^{N/2-1} * int_{-infty}^infty P(exp(u)) exp(N*u) du

        with u = log(k), du = dk/k.

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


class MaternCovarianceKernel(_MaternCovarianceKernel, Model):
    """
    NIFTy-wrapper for the Matern covariance kernel.
    Computes the covariance kernel for a Matern-like power spectrum
    with a high-k cutoff. The power spectrum is given by:

    P(k) = 1 / (1 + (k * lengthscale)^2)^(negloglogslope / 2) * exp(- (k / kcutoff)^2)

    where negloglogslope controls the logarithmic slope at high k, lengthscale controls
    the correlation length scale, and kcutoff sets the high-k cutoff.

    Args:
        Ndim: int
            Number of spatial dimensions
        r_min: float
            Minimum r value for covariance kernel interpolation. This should
            correspond to the smallest length scale you want to resolve.
        r_max: float
            Maximum r value for covariance kernel interpolation. This should
            correspond to the largest length scale you want to resolve.
        variance: float or tuple or `Model`
            This sets the variance of the Gaussian process, i.e. cov(0) = variance * (1 + jitter).
            If a float is provided, it is treated as a constant. If a tuple (mean, stddev) is provided,
            it is treated as a log-normal prior. If a `Model` is provided, it is used directly and must
            have a scalar target.
        lengthscale: float or tuple or `Model`
            This sets roughly the largest coherent structure size in the Gaussian process. If a float
            is provided, it is treated as a constant. If a tuple (mean, stddev) is provided, it is treated
            as a log-normal prior. If a `Model` is provided, it is used directly and must have a scalar target.
        negloglogslope: float or tuple or `Model`
            This controls the logarithmic slope of the power spectrum at intemediate to high k. If a float
            is provided, it is treated as a constant. If a tuple (mean, stddev) is provided, it is treated
            as a log-normal prior. If a `Model` is provided, it is used directly and must have a scalar target.
        kcutoff: float or tuple or `Model` or "auto"
            This sets the high-k cutoff of the power spectrum. If a float is provided, it is treated as a constant.
            If a tuple (mean, stddev) is provided, it is treated as a log-normal prior. If a `Model` is provided,
            it is used directly and must have a scalar target. If "auto", it is set to pi/r_min.
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
            multiple instances of MaternCovarianceKernel. The final prior names will be prefix + attribute name,
            e.g. "MaternCovarianceKernel_lengthscale".
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
        Ninterp: int = 128,
        jitter: float = 1.0e-5,
        Nint: Union[int, str] = 512,
        h: Union[float, str] = "auto",
        prefix: str = "MaternCovarianceKernel_",
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

        _MaternCovarianceKernel.__init__(
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
