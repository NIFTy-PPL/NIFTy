#!/usr/bin/env python
from __future__ import division
import numpy as np

# deterministic
np.random.seed(42)


def distance_array_helper(index_arr, lmax):
    if index_arr <= lmax:
        index_half = index_arr
    else:
        if (index_arr - lmax) % 2 == 0:
            index_half = (index_arr + lmax) / 2
        else:
            index_half = (index_arr + lmax + 1) / 2

    m = (
            np.ceil(((2 * lmax + 1) - np.sqrt((2 * lmax + 1)**2 -
                    8 * (index_half - lmax))) / 2)
        ).astype(int)

    return index_half - m * (2 * lmax + 1 - m) // 2


# for LMSpace(5)
da_0 = [distance_array_helper(idx, 5) for idx in np.arange(36)]

# random input for weight
w_0_x = np.random.rand(32, 16, 6)

# random input for hermitian
h_0_res_real = np.random.rand(32, 16, 6).astype(np.complex128)
h_0_res_imag = np.random.rand(32, 16, 6).astype(np.complex128)
h_0_x = h_0_res_real + h_0_res_imag * 1j

# write everything to disk
np.savez('lm_space', da_0=da_0, w_0_x=w_0_x, w_0_res=w_0_x, h_0_x=h_0_x,
         h_0_res_real=h_0_res_real, h_0_res_imag=h_0_res_imag)
