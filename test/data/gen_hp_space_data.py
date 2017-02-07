#!/usr/bin/env python
import numpy as np
import healpy as hp

# deterministic
np.random.seed(42)

# for HPSpace(nside=2)
da_0 = np.array([np.arccos(hp.pix2vec(2, idx)[0]) for idx in np.arange(48)])

# for HPSpace(nside=2)
w_0_x = np.random.rand(48)
w_0_res = w_0_x * ((4 * np.pi) / 48)
w_1_res = w_0_x * (((4 * np.pi) / 48)**2)

# write everything to disk
np.savez('hp_space', da_0=da_0, w_0_x=w_0_x, w_0_res=w_0_res, w_1_x=w_0_x,
         w_1_res=w_1_res)
