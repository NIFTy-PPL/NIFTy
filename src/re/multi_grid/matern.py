#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Philipp Frank, Laurin Soeding, Gordian Edenhofer

from dataclasses import dataclass, field
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial, register_dataclass
from numpy import typing as npt
from scipy.special import j0, sici

from ..model import Model
from ..prior import LogNormalPrior, NormalPrior
from ..tree_math import zeros_like

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
        n_integrate=2_000,
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
                self._ref_distance * np.array([1e-3, 1e4])
            )
        if interpolation_dists_min_max is None:
            interpolation_dists_min_max = tuple(
                self._ref_distance * np.array([1e-3, 1e2])
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
