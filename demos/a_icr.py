# %%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import nifty8.re as jft
from nifty8.re.multi_grid import grid_impl as gi
from nifty8.re.multi_grid.correlated_field import ICRCorrelatedField

# %%

shape = (32, 32)

# TODO: increase grid size and remove desired_size0
grid = gi.SimpleOpenGrid(min_shape=shape, desired_size0=6)
cf = ICRCorrelatedField(
    grid,
    kernel=dict(
        kind="matern",
        scale=(1.0, 0.2),
        cutoff=(1e-1, 1e-2),
        loglogslope=(-6.0, 0.5),
    ),
)

# %%
res = cf(cf.init(jax.random.PRNGKey(42)))
plt.imshow(res)
