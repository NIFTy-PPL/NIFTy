import numpy as np
import jax.numpy as jnp
from functools import partial
from .indexing import OpenGrid, OpenGridAtLevel, MGrid, MGridAtLevel, HEALPixGrid


def logaritmic_grid(depth, rshape0, rlim):
    gr_r = OpenGrid(shape0=(rshape0,), splits=(2,) * depth, padding=(1,) * depth)
    grd = gr_r.at(gr_r.depth)
    Nr = grd.size
    ls = grd.index2coord(np.arange(grd.size))
    print("Log-grid size:", Nr)

    def l_to_r(l, lmi, lma, rmi, rma):
        dl = lma - lmi
        lp = (l - lmi) / dl
        b = np.log(rmi)
        a = np.log(rma) - b
        return jnp.exp(a * lp + b)

    def r_to_l(r, lmi, lma, rmi, rma):
        dl = lma - lmi
        b = np.log(rmi)
        a = np.log(rma) - b

        lp = (jnp.log(r) - b) / a
        return lp * dl + lmi

    f_rl = partial(l_to_r, lmi=ls[0], lma=ls[-1], rmi=rlim[0], rma=rlim[1])
    f_lr = partial(r_to_l, lmi=ls[0], lma=ls[-1], rmi=rlim[0], rma=rlim[1])

    assert np.allclose(ls, f_lr(f_rl(ls)))
    assert np.allclose(rlim[0], f_rl(ls[0]))
    assert np.allclose(rlim[1], f_rl(ls[-1]))

    class MyLGrid(OpenGridAtLevel):
        def coord2index(self, coord):
            coord = f_lr(coord)
            return super().coord2index(coord)

        def index2coord(self, index):
            coord = super().index2coord(index)
            return f_rl(coord)

        def index2volume(self, index):
            coords = super().index2coord(index + jnp.array([-0.5, 0.5])[:, None])
            return jnp.prod(coords[1] - coords[0], axis=0)

    return OpenGrid(
        shape0=gr_r.shape0, splits=gr_r.splits, padding=gr_r.padding, atLevel=MyLGrid
    )


def get_hplogr_grid(depth, rshape0, nside0, rlim):
    gr_r = logaritmic_grid(depth, rshape0, rlim)
    gr_hp = HEALPixGrid(nside0=nside0, depth=depth)
    print("HP Nside:", gr_hp.at(gr_hp.depth).nside)

    class MyMGrid(MGridAtLevel):
        def index2coord(self, index, **kwargs):
            coords = super().index2coord(index, **kwargs)
            return coords[:1] * coords[1:]

        def coord2index(self, coord, **kwargs):
            assert coord.shape[0] == 3
            r = jnp.linalg.norm(coord, axis=0)[jnp.newaxis, ...]
            coord = jnp.concatenate((r, coord / r), axis=0)
            return super().coord2index(coord, **kwargs)

    gr = MGrid(gr_r, gr_hp, atLevel=MyMGrid)
    print("Base size:", gr.at(0).size)
    return gr
