#!/usr/bin/env python
from __future__ import division
import numpy as np

# deterministic
np.random.seed(42)

# for now directly the kindex
# RGSpace((4, 4), harmonic=True)
da_0 = np.array([0, 1.0, 1.41421356, 2., 2.23606798, 2.82842712])

# power 1
w_0_x = np.random.rand(32, 16, 6)
# RGSpace((4, 4), harmonic=True)
# using rho directly
weight_0 = np.array([1, 4, 4, 2, 4, 1])
weight_0 = weight_0.reshape([1, 1, 6])
w_0_res = w_0_x * weight_0


# write everything to disk
np.savez('power_space',
         da_0=da_0,
         w_0_x=w_0_x,
         w_0_res=w_0_res)
