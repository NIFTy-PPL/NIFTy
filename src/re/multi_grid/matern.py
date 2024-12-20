#!/usr/bin/env python3

# Author: Laurin Soeding

from dataclasses import field
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial

from ..model import Model
from ..prior import LogNormalPrior, NormalPrior


def si(x):
    """Calculates the integral of sin(t)/t from 0 to x.

    Following https://en.wikipedia.org/wiki/Trigonometric_integral#Efficient_evaluation
    and https://github.com/GalSim-developers/GalSim/blob/releases/2.6/src/math/Sinc.cpp
    this functions implements formulae by Rowe et al. (2015).
    """
    x2 = x * x

    def g(_x):
        """
        Chebyshev-Pade approximation of 1/y g(1/sqrt(y)) from 0..1/4^2
        leads to the following formula for g(x),
        which is also accurate to better than 1.e-16 for x > 4.
        """
        _y = 1.0 / (_x * _x)
        return (
            _y
            * (
                1.0
                + _y
                * (
                    8.1359520115168615e2
                    + _y
                    * (
                        2.35239181626478200e5
                        + _y
                        * (
                            3.12557570795778731e7
                            + _y
                            * (
                                2.06297595146763354e9
                                + _y
                                * (
                                    6.83052205423625007e10
                                    + _y
                                    * (
                                        1.09049528450362786e12
                                        + _y
                                        * (
                                            7.57664583257834349e12
                                            + _y
                                            * (
                                                1.81004487464664575e13
                                                + _y
                                                * (
                                                    6.43291613143049485e12
                                                    + _y * (-1.36517137670871689e12)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            / (
                1.0
                + _y
                * (
                    8.19595201151451564e2
                    + _y
                    * (
                        2.40036752835578777e5
                        + _y
                        * (
                            3.26026661647090822e7
                            + _y
                            * (
                                2.23355543278099360e9
                                + _y
                                * (
                                    7.87465017341829930e10
                                    + _y
                                    * (
                                        1.39866710696414565e12
                                        + _y
                                        * (
                                            1.17164723371736605e13
                                            + _y
                                            * (
                                                4.01839087307656620e13
                                                + _y * (3.99653257887490811e13)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

    def f(_x):
        """
        Chebyshev-Pade approximation of 1/sqrt(y) f(1/sqrt(y)) from 0..1/4^2
        leads to the following formula for g(x),
        which is also accurate to better than 1.e-16 for x > 4.
        """
        _y = 1.0 / (_x * _x)
        return (
            1.0
            + _y
            * (
                7.44437068161936700618e2
                + _y
                * (
                    1.96396372895146869801e5
                    + _y
                    * (
                        2.37750310125431834034e7
                        + _y
                        * (
                            1.43073403821274636888e9
                            + _y
                            * (
                                4.33736238870432522765e10
                                + _y
                                * (
                                    6.40533830574022022911e11
                                    + _y
                                    * (
                                        4.20968180571076940208e12
                                        + _y
                                        * (
                                            1.00795182980368574617e13
                                            + _y
                                            * (
                                                4.94816688199951963482e12
                                                + _y * (-4.94701168645415959931e11)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ) / (
            _x
            * (
                1.0
                + _y
                * (
                    7.46437068161927678031e2
                    + _y
                    * (
                        1.97865247031583951450e5
                        + _y
                        * (
                            2.41535670165126845144e7
                            + _y
                            * (
                                1.47478952192985464958e9
                                + _y
                                * (
                                    4.58595115847765779830e10
                                    + _y
                                    * (
                                        7.08501308149515401563e11
                                        + _y
                                        * (
                                            5.06084464593475076774e12
                                            + _y
                                            * (
                                                1.43468549171581016479e13
                                                + _y * (1.11535493509914254097e13)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

    def large_x2_approx(_x):
        """
        For |x| > 4, we use the asymptotic formula:
        si(x) = pi/2 - f(x) cos(x) - g(x) sin(x)
        where f(x) = int(sin(t)/(x+t),t=0..inf)
              g(x) = int(cos(t)/(x+t),t=0..inf)
        (asymptotic: f and g approach 1/x and 1/x^2 respectively as x -> inf. The formula as given is exact.)
        """
        fx = f(_x)
        gx = g(_x)

        sinx = jnp.sin(_x)
        cosx = jnp.cos(_x)
        return jnp.where(_x > 0.0, jnp.pi / 2.0, -jnp.pi / 2.0) - fx * cosx - gx * sinx

    def small_x2_approx(_x):
        """
        Here, Maple was used to calculate the Pade approximation for si(x), which is accurate
        to better than 1.e-16 for x < 4.
        """
        _x2 = _x * _x
        return (
            _x
            * (
                1.0
                + _x2
                * (
                    -4.54393409816329991e-2
                    + _x2
                    * (
                        1.15457225751016682e-3
                        + _x2
                        * (
                            -1.41018536821330254e-5
                            + _x2
                            * (
                                9.43280809438713025e-8
                                + _x2
                                * (
                                    -3.53201978997168357e-10
                                    + _x2
                                    * (
                                        7.08240282274875911e-13
                                        + _x2 * (-6.05338212010422477e-16)
                                    )
                                )
                            )
                        )
                    )
                )
            )
            / (
                1.0
                + _x2
                * (
                    1.01162145739225565e-2
                    + _x2
                    * (
                        4.99175116169755106e-5
                        + _x2
                        * (
                            1.55654986308745614e-7
                            + _x2
                            * (
                                3.28067571055789734e-10
                                + _x2
                                * (
                                    4.5049097575386581e-13
                                    + _x2 * (3.21107051193712168e-16)
                                )
                            )
                        )
                    )
                )
            )
        )

    return jnp.where(x2 > 16.0, large_x2_approx(x), small_x2_approx(x))


def sin_integral_antilinearly_exact(f, xs, k):
    """Compute integral of $f(x) \\cdot sin(kx)$ using antilinearly exact evaluation.

    That is, $f(x) \\approx c_1 + \\frac{c_2}{x}$ piecewise approximated and then
    analytically integrated.
    """
    x = xs
    kx = k * x
    f_values = f(x)

    dxinv = jnp.diff(1.0 / x)
    df = jnp.diff(f_values)

    si_kx = si(kx)
    cos_kx = jnp.cos(kx)
    c2 = df / dxinv
    c1 = f_values[:-1] - c2 / x[:-1]

    int_1 = c1 / k * (-jnp.diff(cos_kx))
    int_2 = c2 * jnp.diff(si_kx)
    result = jnp.sum(int_1 + int_2)
    return result


class MaternHarmonicCovariance(Model):
    scale: Union[Model, float] = field(metadata=dict(static=False))
    cutoff: Union[Model, float] = field(metadata=dict(static=False))
    loglogslope: Union[Model, float] = field(metadata=dict(static=False))
    _interpolation_log_dists: jnp.ndarray = field(metadata=dict(static=False))
    _integration_log_dists: jnp.ndarray = field(metadata=dict(static=False))

    def __init__(
        self,
        scale: Union[tuple, Callable, float],
        cutoff: Union[tuple, Callable, float],
        loglogslope: Union[tuple, Callable, float],
        *,
        ndim: int,
        n_integrate=2_000,
        n_interpolate=128,
        interpolation_dists_min_max=(1e-3, 1e2),
        integration_dists_min_max=(1e-3, 1e4),
        prefix: str = "",
    ):
        if isinstance(cutoff, (tuple, list)):
            cutoff = LogNormalPrior(*cutoff, name=prefix + "cutoff")
        elif not (callable(cutoff) or isinstance(cutoff, float)):
            raise TypeError(f"invalid `cutoff` specified; got '{cutoff!r}'")
        self.cutoff = cutoff
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

        self.ndim = ndim

        self._interpolation_log_dists = jnp.geomspace(
            *interpolation_dists_min_max, n_interpolate
        )
        self._integration_log_dists = jnp.geomspace(
            *integration_dists_min_max, n_integrate
        )

        super().__init__(
            domain=getattr(self.scale, "domain", {})
            | getattr(self.loglogslope, "domain", {})
            | getattr(self.cutoff, "domain", {})
        )

    @staticmethod
    def _cov(x, y, *, scale, distances, logcorr):
        r = jnp.linalg.norm(x - y, axis=0, ord=2)
        cov = jnp.exp(
            jnp.interp(r, distances, logcorr, left="extrapolate", right="extrapolate")
        )
        cov = jnp.where(r == 0.0, jnp.ones_like(r), cov)
        return scale * cov

    def __call__(self, x):
        scale = self.scale(x) if callable(self.scale) else self.scale
        cutoff = self.cutoff(x) if callable(self.cutoff) else self.cutoff
        loglogslope = (
            self.loglogslope(x) if callable(self.loglogslope) else self.loglogslope
        )
        # NOTE, emulate an ndim-harmonic transform by decrementing the slope. This
        # is technically not the same but at least for the Matern kernel it is similar.
        loglogslope = loglogslope - self.ndim

        def integral(r):
            r = jnp.abs(r) + 1.0e-5  # avoid numerical nonsense

            def f(k):
                power = 1.0 / (1.0 + (k * cutoff) ** 2) ** (-loglogslope / 2.0)
                return 1.0 / (2.0 * jnp.pi**2) * k / r * power

            return sin_integral_antilinearly_exact(f, self._integration_log_dists, r)

        correlations = jax.vmap(integral)(self._interpolation_log_dists) / integral(0.0)

        # This could perhaps go even lower, but the accuaracy is already quite
        # good and this is definitely numerically stable.
        # Restrict domain to ensure good numerics
        mask = correlations < 1.0e-5
        maxidx = jnp.argmax(mask) - 1
        # take abs to ensure good logarithm
        correlations = jnp.abs(correlations)
        logcorr = jnp.log(correlations)
        # DIY interpolation that keeps the shapes static
        slope_at_maxidx = (logcorr[maxidx - 1] - logcorr[maxidx]) / (
            self._interpolation_log_dists[maxidx - 1]
            - self._interpolation_log_dists[maxidx]
        )
        logcorr = jnp.where(
            ~mask,
            logcorr,
            logcorr[maxidx]
            + slope_at_maxidx
            * (self._interpolation_log_dists - self._interpolation_log_dists[maxidx]),
        )

        return Partial(
            self.__class__._cov,
            scale=scale,
            distances=self._interpolation_log_dists,
            logcorr=logcorr,
        )
