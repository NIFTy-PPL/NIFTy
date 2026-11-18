# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

"""Log-density of the noncentral chi-squared distribution in JAX, any integer k.

    f_k(d; lam) = 1/2 exp(-(d+lam)/2) (d/lam)**(nu/2) I_nu(sqrt(lam d)),
    nu = k/2 - 1,   k = 1, 2, 3, ...

so nu is an integer (even k) or a half-integer (odd k).  The density is
written with the entire function

    Itilde_nu(u) = I_nu(sqrt u) / (sqrt(u)/2)**nu,     Itilde_nu(0) = 1/Gamma(nu+1),

    f_k(d; lam) = 1/2 exp(-(d+lam)/2) (d/2)**nu Itilde_nu(lam d),

and every branch is a function of u = lam d, guarded by the double-where idiom,
so that reverse-mode AD stays finite at lam = 0.  (`jnp.where` evaluates both
branches and their gradients; "double-where" first replaces the input of the
unused branch by a harmless value, so that branch can never produce inf/NaN
that would leak into the gradient as 0 * inf = NaN.)

`jax.scipy.special` only provides i0e and i1e, so log Itilde_nu is assembled
from four pieces, chosen at trace time from the static nu:

- u < 1: power series in u (positive terms, no cancellation), for every nu;
- nu in {-1/2, 0, 1/2, 1}: closed forms (cosh, sinh) or i0e / i1e;
- 3/2 <= nu < NU_DEBYE: a long power series up to z = sqrt(u) < Z_HANKEL(nu)
  and the Hankel large-argument expansion above (it terminates, i.e. is exact
  up to exp(-2z), for half-integer nu);
- nu >= NU_DEBYE: Debye's uniform asymptotic expansion in 1/nu.

For large u the logpdf uses -(d+lam)/2 + z = -(sqrt d - sqrt lam)**2 / 2 and
the scaled function log Itilde_nu(u) - z, which avoids cancelling two large
terms.  All thresholds were chosen by comparison with mpmath at 30 digits;
log Itilde_nu agrees with it to <= 1.1e-14 for k <= 1000 and z in [0, 3e6].

For k >= 5 the lam-derivative is not obtained by autodiff through these
branches but from the analytic score (f_{k+2}/f_k - 1)/2 (custom JVP), which
makes the energy gradient up to ~50 times cheaper.

Requires `jax_enable_x64`.
"""

import functools
import math
from fractions import Fraction

import jax
import numpy as np
from jax import numpy as jnp
from jax.custom_derivatives import SymbolicZero
from jax.scipy.special import i0e, i1e

__all__ = ["make_ncx2_logpdf", "log_itilde"]

# below u = lam*d = 1 the short series is used for every nu
_U_SWITCH = 1.0
_N_SERIES = 12
# Debye's expansion from this nu on (20 terms: <= 2e-15 at nu = 10, z >= 1;
# 25 terms are worse, the series is asymptotic); below it the long series for
# z < _Z_HANKEL and Hankel above (<= 2e-16 for z >= 20 and nu <= 9.5).
NU_DEBYE = 10.0
_N_DEBYE = 20  # number of U_k terms (k = 0 ... _N_DEBYE - 1)
_N_HANKEL = 40  # maximal number of Hankel terms
_Z_HANKEL = 30.0


def _z_hankel(nu):
    """z above which the Hankel expansion is used (3/2 <= nu < NU_DEBYE)."""
    return _Z_HANKEL


def _n_long_series(nu):
    # terms of the series peak at m ~ z/2; cover z < _z_hankel with margin
    zh = _z_hankel(nu)
    return int(zh / 2 + 12 * math.sqrt(zh / 2 + 1) + 30)


def _nu_of_k(k):
    return k / 2.0 - 1.0


# ------------------------------------------------------------ power series
def _log_series_sum(u, nu, n_terms):
    """log(Gamma(nu+1) Itilde_nu(sqrt(u))), the series normalised to start at 1."""
    t = u / 4.0
    # the m = 0 term is 1/Gamma(nu+1); factor it out so the sum starts at 1
    acc = jnp.ones_like(u)
    term = jnp.ones_like(u)
    for m in range(1, n_terms):
        term = term * t / (m * (m + nu))
        acc = acc + term
    return jnp.log(acc)


def _log_series(u, nu, n_terms):
    """log Itilde_nu(sqrt(u)) = log sum_m (u/4)**m / (m! Gamma(m+nu+1))."""
    return _log_series_sum(u, nu, n_terms) - math.lgamma(nu + 1.0)


# ------------------------------------------------------------ Hankel
def _hankel_coeffs(nu):
    """a_k(nu) with I_nu(z) ~ e^z / sqrt(2 pi z) sum_k (-1)^k a_k / z^k."""
    a = [1.0]
    for j in range(1, _N_HANKEL):
        a.append(a[-1] * (4.0 * nu * nu - (2 * j - 1) ** 2) / (8.0 * j))
        if a[-1] == 0.0:  # half-integer nu: series terminates
            break
    return a


def _log_scaled_hankel(z, nu):
    """log Itilde_nu(z**2) - z from the Hankel expansion (large z)."""
    a = _hankel_coeffs(nu)
    w = 1.0 / z
    s = jnp.zeros_like(z)
    for ak in reversed([(-1) ** j * c for j, c in enumerate(a)]):
        s = s * w + ak
    return jnp.log(s) - 0.5 * jnp.log(2.0 * math.pi * z) - nu * jnp.log(z / 2.0)


# ------------------------------------------------------------ Debye
def _debye_polys(n):
    """Coefficients (in p) of Debye's polynomials U_0 ... U_{n-1}, exact.

    U_{k+1}(p) = p**2 (1 - p**2) U_k'(p) / 2 + 1/8 Int_0^p (1 - 5 t**2) U_k(t) dt
    """
    polys = [[Fraction(1)]]
    for _ in range(n - 1):
        u = polys[-1]
        deg = len(u) + 2
        new = [Fraction(0)] * (deg + 1)
        for j, c in enumerate(u):  # p^2 (1 - p^2) U' / 2
            if j:
                new[j + 1] += c * j / 2
                new[j + 3] -= c * j / 2
        for j, c in enumerate(u):  # 1/8 Int (1 - 5t^2) U
            new[j + 1] += c / (8 * (j + 1))
            new[j + 3] -= 5 * c / (8 * (j + 3))
        while new and new[-1] == 0:
            new.pop()
        polys.append(new)
    return [np.array([float(c) for c in p]) for p in polys]


_DEBYE = _debye_polys(_N_DEBYE)


def _log_scaled_debye(z, nu):
    """log Itilde_nu(z**2) - z from Debye's uniform expansion (nu large)."""
    root = jnp.sqrt(nu * nu + z * z)
    p = nu / root
    # nu*eta - z = (sqrt(nu^2+z^2) - z) + nu log(z / (nu + sqrt(nu^2+z^2))),
    # with sqrt(nu^2+z^2) - z = nu^2 / (root + z): the plain difference
    # cancels for z >> nu
    nu_eta_m_z = nu * nu / (root + z) + nu * jnp.log(z / (nu + root))
    s = jnp.zeros_like(z)
    for coeffs in reversed(_DEBYE):
        uk = jnp.zeros_like(z)
        for c in reversed(coeffs):
            uk = uk * p + c
        s = s / nu + uk
    # I_nu(z) ~ exp(nu eta) / (sqrt(2 pi nu) (1 + x^2)^(1/4)) sum U_k / nu^k,
    # (1 + x^2)^(1/4) = (root/nu)^(1/2)
    log_i = nu_eta_m_z - 0.5 * jnp.log(2.0 * math.pi * root) + jnp.log(s)
    return log_i - nu * jnp.log(z / 2.0)


# ------------------------------------------------------------ dispatch
def _log_scaled_large(z, nu):
    """log Itilde_nu(z**2) - z for z >= 1 (static nu)."""
    # nu is a Python number: the branch is chosen at trace time and only that
    # one is compiled.  Everything is scaled by e^-z (no overflow at large z).
    # nu = -1/2 ... 1 (k = 1 ... 4): exact closed forms / i0e, i1e
    if nu == -0.5:
        # Itilde = cosh(z) / sqrt(pi)
        return jnp.log1p(jnp.exp(-2.0 * z)) - math.log(2.0) - 0.5 * math.log(math.pi)
    if nu == 0.0:
        return jnp.log(i0e(z))
    if nu == 0.5:
        # Itilde = 2 sinh(z) / (z sqrt(pi))
        return jnp.log1p(-jnp.exp(-2.0 * z)) - jnp.log(z) - 0.5 * math.log(math.pi)
    if nu == 1.0:
        return jnp.log(i1e(z)) - jnp.log(z / 2.0)
    # large nu: Debye is uniform in z; the Hankel expansion would need z >> nu^2
    if nu >= NU_DEBYE:
        return _log_scaled_debye(z, nu)
    # 3/2 <= nu < NU_DEBYE: the long series needs ~z/2 + O(sqrt z) terms, so
    # above Z_HANKEL the (there rounding-accurate) Hankel expansion is used
    zh = _z_hankel(nu)
    big = z >= zh
    z_h = jnp.where(big, z, zh)  # double-where
    z_s = jnp.where(big, zh, z)  # double-where
    ser = _log_series(z_s * z_s, nu, _n_long_series(nu)) - z_s
    return jnp.where(big, _log_scaled_hankel(z_h, nu), ser)


def log_itilde(u, nu):
    """log of I_nu(sqrt(u)) / (sqrt(u)/2)**nu, smooth and finite at u = 0.

    nu must be an integer or a half-integer >= -1/2 (static Python number).
    """
    u = jnp.where(u < 0.0, 0.0, u)
    small = u < _U_SWITCH
    u_s = jnp.where(small, u, 0.0)  # double-where
    u_l = jnp.where(small, _U_SWITCH, u)
    z = jnp.sqrt(u_l)
    return jnp.where(
        small, _log_series(u_s, nu, _N_SERIES), _log_scaled_large(z, nu) + z
    )


def _check_k(k):
    if isinstance(k, bool) or not float(k).is_integer() or k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")
    return int(k)


def _make_core(k):
    """log f_k(d, lam) for float arrays, negative d and lam clamped to 0.

    Differentiated by plain autodiff (reference for the custom JVP below).
    """
    nu = _nu_of_k(k)
    # Python floats: jnp values created here would be tracers if the factory
    # is first called under jit, and would leak through the closure
    log_half = math.log(2.0)

    def core(d, lam):
        d_c = jnp.where(d < 0.0, 0.0, d)
        lam_c = jnp.where(lam < 0.0, 0.0, lam)

        u = lam_c * d_c
        small = u < _U_SWITCH

        u_s = jnp.where(small, u, 0.0)  # double-where
        core_s = -(d_c + lam_c) / 2.0 + _log_series(u_s, nu, _N_SERIES)

        # large u: -(d+lam)/2 + sqrt(lam d) == -(sqrt d - sqrt lam)**2 / 2;
        # the left side cancels two large terms, the right side does not
        u_l = jnp.where(small, _U_SWITCH, u)  # double-where
        # dummy lam for the unused branch, consistent with u_l = lam_l * d
        lam_l = jnp.where(small, _U_SWITCH / jnp.where(d_c > 0.0, d_c, 1.0), lam_c)
        z = jnp.sqrt(u_l)
        r = jnp.sqrt(d_c) - jnp.sqrt(lam_l)
        core_l = -r * r / 2.0 + _log_scaled_large(z, nu)

        out = -log_half + jnp.where(small, core_s, core_l)
        # k = 2 (nu = 0): no (d/2)**nu factor, the density is finite at d = 0;
        # otherwise d = 0 gives d**nu = inf (k = 1) or 0 (k >= 3), set exactly
        if nu != 0.0:
            d_pos = jnp.where(d_c > 0.0, d_c, 1.0)  # double-where
            out = out + nu * (jnp.log(d_pos) - log_half)
            edge = jnp.inf if nu < 0.0 else -jnp.inf
            out = jnp.where(d_c > 0.0, out, edge)
        return out

    return core


def _make_log_ratio(k):
    """log(f_{k+2}/f_k)(d, lam) for clamped d > 0, lam >= 0.

    Written out instead of subtracting two logpdfs, so that the terms common
    to both, -(d+lam)/2 and nu*log(d/2), cancel exactly rather than leaving
    their rounding error (visible where the score is ~0, i.e. d ~ k):

        log f_{k+2}/f_k = log(d/2) + log Itilde_{nu+1}(u) - log Itilde_nu(u)
          = log(d/k) + S_{nu+1}(u) - S_nu(u)          u < 1 (series sums S,
                                                       Gamma(nu+2)/Gamma(nu+1)
                                                       = k/2)
          = log(d/2) + L_{nu+1}(z) - L_nu(z)          u >= 1 (scaled, the +z
                                                       cancels)
    """
    nu = _nu_of_k(k)

    def log_ratio(d_c, lam_c):
        u = lam_c * d_c
        small = u < _U_SWITCH
        u_s = jnp.where(small, u, 0.0)  # double-where
        r_s = _log_series_sum(u_s, nu + 1.0, _N_SERIES) - _log_series_sum(
            u_s, nu, _N_SERIES
        )
        u_l = jnp.where(small, _U_SWITCH, u)  # double-where
        z = jnp.sqrt(u_l)
        r_l = (
            math.log(k / 2.0)
            + _log_scaled_large(z, nu + 1.0)
            - _log_scaled_large(z, nu)
        )
        d_pos = jnp.where(d_c > 0.0, d_c, 1.0)  # double-where
        out = jnp.log(d_pos / k) + jnp.where(small, r_s, r_l)
        return jnp.where(d_c > 0.0, out, -jnp.inf)  # f_{k+2}(0) = 0

    return log_ratio


def _make_fast_core(k):
    """`_make_core(k)` with the lam-derivative given analytically.

    Reverse-mode autodiff through the Bessel branches (Debye polynomials, long
    series, both sides of every `where`) costs up to ~100 times the function
    itself.  The score is known in closed form,

        d/dlam log f_k = (f_{k+2}/f_k - 1) / 2 = expm1(log f_{k+2}/f_k) / 2,

    see `_make_log_ratio`.  It is set to 0 where the autodiff derivative of
    `_make_core` is 0 by construction: lam < 0 (clamped) and, for k != 2,
    d = 0 (constant edge value +-inf).  The d-derivative is never needed by
    the likelihood and falls back to autodiff.
    """
    core = _make_core(k)
    log_ratio = _make_log_ratio(k)

    @jax.custom_jvp
    def fast(d, lam):
        return core(d, lam)

    @functools.partial(fast.defjvp, symbolic_zeros=True)
    def _fast_jvp(primals, tangents):
        d, lam = primals
        d_dot, lam_dot = tangents
        out = core(d, lam)
        t_out = jnp.zeros_like(out)
        if not isinstance(lam_dot, SymbolicZero):
            d_c = jnp.where(d < 0.0, 0.0, d)
            lam_c = jnp.where(lam < 0.0, 0.0, lam)
            score = 0.5 * jnp.expm1(log_ratio(d_c, lam_c))
            if k != 2:
                score = jnp.where(d_c > 0.0, score, 0.0)
            score = jnp.where(lam < 0.0, 0.0, score)
            t_out = t_out + score * lam_dot
        if not isinstance(d_dot, SymbolicZero):
            t_out = t_out + jax.jvp(lambda x: core(x, lam), (d,), (d_dot,))[1]
        return out, t_out

    return fast


def _make_logpdf(core, validate):
    def logpdf(d, lam):
        d = jnp.asarray(d, dtype=float)
        lam = jnp.asarray(lam, dtype=float)
        # custom_jvp needs tangents of the same shape as the inputs
        d, lam = jnp.broadcast_arrays(d, lam)
        out = core(d, lam)
        if validate:
            out = jnp.where(d < 0.0, -jnp.inf, out)
            out = jnp.where(lam < 0.0, jnp.nan, out)
        return out

    return logpdf


def _make_ncx2_logpdf_autodiff(k, validate=True):
    """`make_ncx2_logpdf` without the analytic score (reference for tests)."""
    return _make_logpdf(_make_core(_check_k(k)), validate)


def make_ncx2_logpdf(k, validate=True):
    """Return logpdf(d, lam) for the noncentral chi-squared with `k` dof.

    Parameters
    ----------
    k : int
        Degrees of freedom, any integer >= 1.
    validate : bool
        If True, return -inf for d < 0 and NaN for lam < 0 instead of
        silently clamping.  Jit-safe either way (implemented with `where`).

    Returns
    -------
    logpdf : callable
        ``logpdf(d, lam)`` with broadcasting over both arguments.  At d = 0
        it returns +inf for k = 1 (the density diverges), log(1/2) - lam/2 for
        k = 2 and -inf for k >= 3.  For k >= 5 the derivative in lam is the
        analytic score (f_{k+2}/f_k - 1)/2 (see `_make_fast_core`).
    """
    k = _check_k(k)
    # k <= 4 (closed forms, i0e/i1e): autodiff is cheaper than two function
    # evaluations (gradient 2 - 10 ns vs 6 - 18 ns per element); from k = 5 on
    # the analytic score is 4 - 50 times faster (k = 5: 36 -> 9 ns,
    # k = 50: 790 -> 15 ns; N = 1e6, CPU)
    core = _make_core(k) if k <= 4 else _make_fast_core(k)
    return _make_logpdf(core, validate)
