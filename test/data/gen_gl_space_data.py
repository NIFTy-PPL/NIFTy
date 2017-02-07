#!/usr/bin/env python
import itertools
import numpy as np
import libsharp_wrapper_gl as gl

# deterministic
np.random.seed(42)


def distance_array(nlat, nlon, latitude, longitude):
    lat = latitude * (np.pi / (nlat - 1))
    lon = longitude * (2 * np.pi / (nlon - 1))
    # Vincenty formula: https://en.wikipedia.org/wiki/Great-circle_distance
    # phi_1, lambda_1 = lat, lon
    # phi_2, lambda_2 = 0
    numerator = np.sqrt((np.cos(0) * np.sin(lon - 0))**2 +
                        ((np.cos(lat) * np.sin(0)) -
                         (np.sin(lat) * np.cos(0) * np.cos(lon - 0)))**2)
    denominator = (
        np.sin(lat) * np.sin(0)) + (np.cos(lat) * np.cos(0) * np.cos(lon - 0))
    return np.arctan(numerator/denominator)


# for GLSpace(nlat=2, nlon=3)
da_0 = np.array(
        [distance_array(2, 3, *divmod(idx, 3)) for idx in np.arange(6)])

# for GLSpace(nlat=2, nlon=3)
weight_0 = np.array(list(itertools.chain.from_iterable(
    itertools.repeat(x, 3) for x in gl.vol(2))))
w_0_x = np.random.rand(6)
w_0_res = w_0_x * weight_0

weight_1 = np.array(list(itertools.chain.from_iterable(
    itertools.repeat(x, 3) for x in gl.vol(2))))
weight_1 = weight_1.reshape([1, 1, 6])
w_1_x = np.random.rand(32, 16, 6)
w_1_res = w_1_x * weight_1

# write everything to disk
np.savez(
    'gl_space', da_0=da_0, w_0_x=w_0_x, w_0_res=w_0_res, w_1_x=w_1_x,
    w_1_res=w_1_res)
