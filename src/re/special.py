# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Margret Westerkamp, Vincent Eberle, Philipp Frank

import numpy as np
import jax.numpy as jnp
from typing import Any
from jax import vmap
from .gauss_markov import wiener_process
from .model import LazyModel, Model, VModel
from dataclasses import field


def get_freqs(lim_freqs, N_freqs, base=np.log):
    freqmin = base(lim_freqs[0])
    freqmax = base(lim_freqs[1])
    if freqmax <= freqmin:
        raise ValueError(f"Frequencies of invalid range [{freqmin}, {freqmax}[")
    return (freqmax - freqmin) / N_freqs * np.arange(N_freqs) + freqmin


def wp_with_drift(offst, drft, dev, frequencies):
    df = frequencies[1:] - frequencies[:-1]
    f0 = frequencies - frequencies[0]
    return wiener_process(dev, offst, sigma=1, dt=df) + drft * f0


def MFSkyModel(
    offset: LazyModel,
    drift: LazyModel,
    deviations: LazyModel,
    in_axes: Any = None,
    out_axes=0,
    N_freqs: int = None,
    lim_freqs: tuple = None,
    base=np.log,
    _freqs=None,
):
    shp = np.broadcast_shapes(
        offset.target.shape, drift.target.shape, deviations.target.shape
    )
    freqs = get_freqs(lim_freqs, N_freqs, base=base) if _freqs is None else _freqs
    N_freqs = freqs.size
    if in_axes is not None:
        deviations = VModel(deviations, N_freqs - 1, in_axes=in_axes, out_axes=-1)
    tot_shape = shp[:out_axes] + (N_freqs,) + shp[out_axes:]

    def apply(x):
        return vmap(wp_with_drift, in_axes=(0,) * 3 + (None,), out_axes=~out_axes)(
            jnp.broadcast_to(offset(x), shp).ravel(),
            jnp.broadcast_to(drift(x), shp).ravel(),
            jnp.broadcast_to(deviations(x), shp + (N_freqs - 1,)).reshape(
                (offset.target.size, -1)
            ),
            freqs,
        ).reshape(tot_shape)

    model = Model(apply, init=offset.init | drift.init | deviations.init)
    model.offset = offset
    model.drift = drift
    model.deviations = deviations
    model.frequencies = freqs
    return model
